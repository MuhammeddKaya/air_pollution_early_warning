# -*- coding: utf-8 -*-
"""
config.py — Merkezi konfigürasyon.
Yön 1 proje planındaki deney matrisiyle (Tablo M1) birebir örtüşür.
Tüm yollar/eşikler/ufuklar buradan yönetilir.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Değişken adları (veri setindeki Türkçe sütunlar)
# ─────────────────────────────────────────────────────────────
TARGET = "PM2.5"

POLLUTANTS = ["PM10", "PM2.5", "SO2", "NO2", "NOX", "NO", "O3"]

# Döngüsel kodlanacak zaman sütunları (sin/cos)
TIME_COLS = ["AY", "GUN", "SAAT"]          # yıl trend olarak ayrı tutulabilir
CYCLE_PERIODS = {"AY": 12, "GUN": 31, "SAAT": 24}

WIND_DIR_COL = "RUZGAR_YONU"               # kategorik (N, NNE, ... , C) → açı → sin/cos

# Sürekli meteoroloji / NASA POWER değişkenleri (örnek; veriye göre güncelleyin)
MET_NASA_COLS = [
    "NISPI_NEM_%", "SICAKLIK", "TOPLAM_YAGIS", "RUZGAR_HIZI",
    "2_m_ozgul_nem(g/kg)", "yuzey_basinc (kPa)", "2_m_ruzg_hiz(m/s)",
    "10_m_ruzg_hiz(m/s)", "tum_gy_irradiance_kisa_dalga(MJ/saat)",
    "tum_gokyuzu_UV_indeksi(ölçeksiz)", "2_m_ciy_noktasi(°C)",
]


@dataclass
class DataConfig:
    path: str = "data/last_dataset_Cok_Onemli.xlsx"   # veya .csv
    sheet: str = "Sheet1"
    datetime_cols: tuple = ("YIL", "AY", "GUN", "SAAT")
    lookback: int = 72          # geçmiş pencere L (saat)
    horizons: tuple = (8, 24, 72)   # tahmin ufukları H (saat)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # 'chrono': kronolojik bölme | 'blocked': 30 günlük bloklar, her mevsim her kümede (kış dahil)
    split_mode: str = "blocked"
    block_hours: int = 720
    # test_frac = 1 - train - val
    # Öznitelik setleri: 'full' | 'corr' | 'causal'
    feature_set: str = "full"


@dataclass
class ThresholdConfig:
    """Aşım eşikleri (µg/m³) — orijinal (fiziksel) birimde uygulanır."""
    who_agg: float = 15.0                    # DSÖ 2021 24-saat kılavuz
    who_interim: tuple = (75.0, 50.0, 37.5, 25.0)   # IT-1..IT-4
    national: Optional[float] = None         # ulusal sınır/hedef (varsa doldurun)
    data_driven_pct: tuple = (90.0, 95.0, 99.0)     # yerel yüzdelikler


@dataclass
class ModelConfig:
    backbone: str = "gru"          # 'lstm' | 'gru' | 'tft' | 'patchtst' | 'naive'
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    # kuantil regresyon için hedef kuantiller
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class UQConfig:
    method: str = "conformal"      # 'ensemble' | 'mc_dropout' | 'quantile' | 'conformal' | 'diffusion_tsfm'
    ensemble_size: int = 5
    mc_samples: int = 50
    conformal_alpha: float = 0.1   # 1 - alpha = hedef kapsama (ör. %90)


@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 10             # early stopping
    batch_size: int = 64
    lr: float = 1e-3
    optimizer: str = "adam"        # 'adam' | 'rmsprop'
    seeds: tuple = (0, 1, 2, 3, 4) # çoklu tohum (ensemble + varyans)
    device: str = "cuda"           # TRUBA: 'cuda'; yerel test: 'cpu'


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    thr: ThresholdConfig = field(default_factory=ThresholdConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uq: UQConfig = field(default_factory=UQConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "results"

    # Deney matrisi (Tablo M1) — çalıştırılacak kombinasyonlar
    backbones: tuple = ("lstm", "gru", "tft")
    uq_methods: tuple = ("ensemble", "mc_dropout", "quantile", "conformal")
    feature_sets: tuple = ("full", "corr", "causal")
    transfer_stations: tuple = ()   # ek istasyon dosya yolları (dış geçerlilik)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
