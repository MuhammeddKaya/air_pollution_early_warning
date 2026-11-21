# -*- coding: utf-8 -*-
"""
data.py — Veri yükleme, ön işleme, öznitelik seti seçimi, pencereleme ve bölme.
Ön işlem hattı ilk makaleyle tutarlıdır (imputasyon, döngüsel kodlama, normalizasyon).

Gerçek implementasyonla gelen kısımlar: döngüsel kodlama, pencereleme, zaman-sıralı bölme,
train-only normalizasyon, rolling-origin üreteç, öznitelik-seti maskeleme.
TODO ile bırakılanlar: gelişmiş imputasyon stratejileri (istenirse).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import config as C


# 16 yönlü pusula → açı (derece), 'C' (rüzgarsız) → NaN/0 vektör
_WIND_ANGLE = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5, "C": np.nan,
}


# ─────────────────────────────────────────────────────────────
# 1) Yükleme
# ─────────────────────────────────────────────────────────────
def load_raw(path: str, sheet: str = "Sheet1") -> pd.DataFrame:
    """Excel veya CSV'den ham veriyi yükler ve zaman damgası oluşturur."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)
    # zaman damgası
    dt = pd.to_datetime(dict(
        year=df["YIL"], month=df["AY"], day=df["GUN"], hour=df["SAAT"]
    ), errors="coerce")
    df = df.assign(_dt=dt).sort_values("_dt").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# 2) Ön işleme
# ─────────────────────────────────────────────────────────────
def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makale 1 stratejisi:
      - meteoroloji/sürekli: doğrusal interpolasyon (önceki/sonraki)
      - yağış: 0 ile doldur
      - rüzgar yönü (kategorik): ffill/bfill
      - kirleticiler: kısa boşluklar interpolasyon
    """
    df = df.copy()
    if "TOPLAM_YAGIS" in df:
        df["TOPLAM_YAGIS"] = df["TOPLAM_YAGIS"].fillna(0.0)
    if C.WIND_DIR_COL in df:
        df[C.WIND_DIR_COL] = df[C.WIND_DIR_COL].ffill().bfill()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    return df


def correct_outliers(df: pd.DataFrame, col: str = C.TARGET,
                     window: int = 24, k: float = 2.0) -> pd.DataFrame:
    """Kayan ortalama ± k·std sınırına kırpma (Makale 1 yöntemi)."""
    df = df.copy()
    mu = df[col].rolling(window, min_periods=1, center=True).mean()
    sd = df[col].rolling(window, min_periods=1, center=True).std().fillna(0)
    lo, hi = mu - k * sd, mu + k * sd
    df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def encode_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Zaman ve rüzgar yönü için sin/cos döngüsel kodlama."""
    df = df.copy()
    for col, period in C.CYCLE_PERIODS.items():
        if col in df:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    if C.WIND_DIR_COL in df:
        ang = df[C.WIND_DIR_COL].map(_WIND_ANGLE)
        rad = np.deg2rad(ang)
        df["WIND_sin"] = np.sin(rad).fillna(0.0)   # 'C' → 0 vektör
        df["WIND_cos"] = np.cos(rad).fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────
# 3) Öznitelik seti seçimi (full | corr | causal)
# ─────────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame, feature_set: str,
                    causal_parents: Optional[List[str]] = None) -> List[str]:
    """
    Öznitelik sütun listesini döndürür.
      - 'full'  : tüm sürekli + kodlanmış öznitelikler
      - 'corr'  : hedefle |korelasyon| eşiğini geçenler (Makale 1 mantığı)
      - 'causal': PCMCI/CCM'den gelen nedensel ebeveynler (causal.py doldurur)
    """
    engineered = [c for c in df.columns
                  if c.endswith(("_sin", "_cos"))]
    base = [c for c in (C.POLLUTANTS + C.MET_NASA_COLS) if c in df.columns]
    full = sorted(set(base + engineered))

    if feature_set == "full":
        feats = full
    elif feature_set == "corr":
        corr = df[list(dict.fromkeys(full + [C.TARGET]))].corr()[C.TARGET].abs()  # PM2.5 full içinde zaten var → yinelenen sütun hatasını önler
        feats = [c for c in full if corr.get(c, 0) >= 0.15]  # eşik ayarlanabilir
    elif feature_set == "causal":
        if not causal_parents:
            raise ValueError("causal öznitelik seti için causal_parents gerekli (bkz. causal.py)")
        feats = [c for c in causal_parents if c in df.columns]
    else:
        raise ValueError(f"bilinmeyen feature_set: {feature_set}")

    # hedef her zaman girişte (otoregresif); istemezseniz çıkarın
    if C.TARGET not in feats:
        feats = [C.TARGET] + feats
    return feats


# ─────────────────────────────────────────────────────────────
# 4) Normalizasyon (yalnız train'e fit)
# ─────────────────────────────────────────────────────────────
@dataclass
class Scaler:
    mins: np.ndarray
    maxs: np.ndarray
    cols: List[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        rng = np.where((self.maxs - self.mins) == 0, 1.0, self.maxs - self.mins)
        return (x - self.mins) / rng

    def inverse_col(self, x: np.ndarray, col: str) -> np.ndarray:
        i = self.cols.index(col)
        return x * (self.maxs[i] - self.mins[i]) + self.mins[i]


def fit_scaler(train_df: pd.DataFrame, cols: List[str]) -> Scaler:
    arr = train_df[cols].to_numpy(dtype=float)
    return Scaler(mins=np.nanmin(arr, 0), maxs=np.nanmax(arr, 0), cols=list(cols))


# ─────────────────────────────────────────────────────────────
# 5) Pencereleme
# ─────────────────────────────────────────────────────────────
def make_windows(values: np.ndarray, target_idx: int,
                 lookback: int, horizons: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    values: (T, F) normalize edilmiş matris.
    Döndürür:
      X: (N, lookback, F)
      Y: (N, H)  — her ufuk için hedef (max(horizons) adım ileri)
    """
    Hmax = max(horizons)
    T, F = values.shape
    N = T - lookback - Hmax + 1
    if N <= 0:
        raise ValueError("yetersiz veri: lookback + horizon toplam uzunluğu aşıyor")
    X = np.zeros((N, lookback, F), dtype=np.float32)
    Y = np.zeros((N, len(horizons)), dtype=np.float32)
    for i in range(N):
        X[i] = values[i:i + lookback]
        for j, h in enumerate(horizons):
            Y[i, j] = values[i + lookback + h - 1, target_idx]
    return X, Y


# ─────────────────────────────────────────────────────────────
# 6) Zaman-sıralı bölme + rolling-origin
# ─────────────────────────────────────────────────────────────
def time_split(n: int, train_frac: float, val_frac: float) -> Tuple[slice, slice, slice]:
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return slice(0, a), slice(a, b), slice(b, n)


def rolling_origin(n: int, initial: int, step: int, horizon: int):
    """Kayan-başlangıç değerlendirme için (train_end, test_slice) üretir."""
    start = initial
    while start + horizon <= n:
        yield start, slice(start, min(start + step, n))
        start += step


def split_windows(n_rows: int, lookback: int, hmax: int, mode: str,
                  train_frac: float, val_frac: float, block_hours: int = 720):
    """
    Pencere indekslerini train/val/test olarak ayırır (sızıntısız).
    Pencere i: girdi satırları [i, i+L), hedefler en geç satır i+L+Hmax-1.
      chrono : hedefler kendi kümesinin zaman aralığında kalır (girdi geçmişi serbest).
      blocked: veri 'block_hours' bloklarına bölünür (k%7==2 → val, k%7==1 → test, kalan train; desen yalnızca takvime bakılarak seçildi: Ara/Oca/Şub her kümede ≥350 satır);
               pencerenin TÜM aralığı aynı kümenin bloklarında olmalı (embargo) →
               her mevsim (kış inversiyonu dahil) her kümede temsil edilir.
    Döndürür: (idx dict {train,val,test} tam sayı dizileri, satır etiketi dizisi 0/1/2).
    """
    N = n_rows - lookback - hmax + 1
    i = np.arange(N)
    start, end = i, i + lookback + hmax - 1
    row_lab = np.zeros(n_rows, dtype=int)            # 0 train, 1 val, 2 test
    if mode == "chrono":
        a, b = int(n_rows * train_frac), int(n_rows * (train_frac + val_frac))
        row_lab[a:b], row_lab[b:] = 1, 2
        lab = np.where(end < a, 0, np.where((start >= a) & (end < b), 1,
                       np.where(start >= b, 2, -1)))
    elif mode == "blocked":
        k = np.arange(n_rows) // block_hours
        row_lab = np.where(k % 7 == 2, 1, np.where(k % 7 == 1, 2, 0))
        same = row_lab[start] == row_lab[end]
        # aradaki bloklar da aynı kümede mi (blok > aralık olduğundan en çok 2 blok)
        mid = row_lab[(start + end) // 2] == row_lab[start]
        lab = np.where(same & mid, row_lab[start], -1)
    else:
        raise ValueError(mode)
    return {n: i[lab == v] for n, v in (("train", 0), ("val", 1), ("test", 2))}, row_lab


# ─────────────────────────────────────────────────────────────
# 7) Uçtan uca hazırlık
# ─────────────────────────────────────────────────────────────
def prepare(cfg: C.ExperimentConfig,
            causal_parents: Optional[List[str]] = None) -> Dict:
    """
    Tüm hattı çalıştırır ve model-hazır dizileri döndürür.
    Not: eşik/metrik hesapları ORİJİNAL birimde yapılır → scaler ile geri dönülür.
    """
    df = load_raw(cfg.data.path, cfg.data.sheet)
    df = impute(df)
    df = correct_outliers(df, C.TARGET)
    df = encode_cyclical(df)

    feats = select_features(df, cfg.data.feature_set, causal_parents)
    df = df.dropna(subset=feats).reset_index(drop=True)

    # sızıntısız bölme (pencere indeksleri) ve train satırlarına göre normalizasyon
    n_rows = len(df)
    L, Hmax = cfg.data.lookback, max(cfg.data.horizons)
    splits, row_lab = split_windows(n_rows, L, Hmax, cfg.data.split_mode,
                                    cfg.data.train_frac, cfg.data.val_frac,
                                    cfg.data.block_hours)
    scaler = fit_scaler(df[row_lab == 0], feats)
    values = scaler.transform(df[feats].to_numpy(dtype=float))
    target_idx = feats.index(C.TARGET)

    X, Y = make_windows(values, target_idx, L, cfg.data.horizons)
    origin_rows = np.arange(len(X)) + L - 1      # pencere i'nin tahmin anı (df satırı)

    return dict(
        X=X, Y=Y, feats=feats, target_idx=target_idx, scaler=scaler,
        splits=splits, origin_rows=origin_rows,
        horizons=cfg.data.horizons, df=df,
    )
