# -*- coding: utf-8 -*-
"""
uq.py — Belirsizlik niceleme yöntemleri ve öngörü dağılımının üretilmesi.

Her yöntem, test kümesi için ORİJİNAL birimde bir öngörü temsili döndürür:
  - 'samples': (N, H, S) örnek matrisi (ensemble/mc-dropout/diffusion)
  - veya 'quantiles': (N, H, Q) + quantile seviyeleri (quantile/conformal)
Bu temsil exceedance.py ve metrics.py tarafından tüketilir.

Gerçek: ensemble/mc-dropout çıkarım mantığı, split-conformal (CQR) kalibrasyonu.
Stub (TODO): eğitim döngüsü (train_one) ve diffusion/TSFM entegrasyonu.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional

try:
    import torch
except Exception:
    torch = None


# ─────────────────────────────────────────────────────────────
# Eğitim döngüsü (stub — betik iskeleti)
# ─────────────────────────────────────────────────────────────
def train_one(model, Xtr, Ytr, Xva, Yva, cfg_train, head: str, quantiles=None, seed: int = 0):
    """
    Tek modeli early stopping ile eğitir; en iyi val ağırlıklarını geri yükler.
    head: 'point' (MSE) | 'gaussian' (NLL) | 'quantile' (pinball).
    Girdi/hedefler normalize birimdedir. Dönüş: eğitilmiş model (eval modunda).
    """
    assert torch is not None, "torch gerekli"
    import copy
    from .models import pinball_loss, gaussian_nll

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(cfg_train.device if (cfg_train.device != "cuda"
                       or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)

    Xtr_t, Ytr_t = torch.as_tensor(Xtr), torch.as_tensor(Ytr)
    Xva_t, Yva_t = torch.as_tensor(Xva).to(dev), torch.as_tensor(Yva).to(dev)

    def loss_fn(out, y):
        if head == "gaussian":
            mu, logvar = out
            return gaussian_nll(mu, logvar.clamp(-10, 10), y)
        if head == "quantile":
            return pinball_loss(out, y, quantiles)
        return torch.nn.functional.mse_loss(out, y)

    opt_cls = torch.optim.RMSprop if cfg_train.optimizer == "rmsprop" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg_train.lr)

    g = torch.Generator().manual_seed(seed)
    n = Xtr_t.shape[0]
    best, best_state, bad = float("inf"), None, 0
    for _ in range(cfg_train.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg_train.batch_size):
            idx = perm[i:i + cfg_train.batch_size]
            xb, yb = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva_t), Yva_t))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg_train.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()


# ─────────────────────────────────────────────────────────────
# 1) Deep Ensemble
# ─────────────────────────────────────────────────────────────
def deep_ensemble_predict(models: List, X, scaler, target_col: str,
                          n_samples: int = 200) -> Dict:
    """
    M Gaussian-başlıklı model → karışım dağılımından örnekleme.
    Döndürür: {'samples': (N,H,S)} orijinal birimde.
    """
    assert torch is not None, "torch gerekli"
    mus, logvars = [], []
    xb = torch.as_tensor(X)
    for m in models:
        m.eval()
        with torch.no_grad():
            mu, logvar = m(xb)
        mus.append(mu.cpu().numpy()); logvars.append(logvar.cpu().numpy())
    mus = np.stack(mus, 0)                 # (M,N,H)
    sig = np.exp(0.5 * np.stack(logvars))  # (M,N,H)
    M, N, H = mus.shape
    s_per = max(1, n_samples // M)
    samples = np.concatenate([
        mus[m][..., None] + sig[m][..., None] * np.random.randn(N, H, s_per)
        for m in range(M)
    ], axis=-1)                            # (N,H,~S)  normalize birimde
    return {"samples": _inv(samples, scaler, target_col)}


# ─────────────────────────────────────────────────────────────
# 2) MC Dropout
# ─────────────────────────────────────────────────────────────
def mc_dropout_predict(model, X, scaler, target_col: str, n_samples: int = 50) -> Dict:
    """Çıkarımda dropout AÇIK; T ileri geçiş → örnekler."""
    assert torch is not None, "torch gerekli"
    model.train()  # dropout açık kalsın (BatchNorm varsa ayrıca ele alın)
    xb = torch.as_tensor(X)
    outs = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(xb)
            mu = out[0] if isinstance(out, tuple) else out
            outs.append(mu.cpu().numpy())
    samples = np.stack(outs, -1)           # (N,H,T)
    return {"samples": _inv(samples, scaler, target_col)}


# ─────────────────────────────────────────────────────────────
# 3) Kuantil Regresyon
# ─────────────────────────────────────────────────────────────
def quantile_predict(model, X, scaler, target_col: str, quantiles) -> Dict:
    """QuantileHead → (N,H,Q) kuantil tahminleri (orijinal birimde)."""
    assert torch is not None, "torch gerekli"
    model.eval()
    with torch.no_grad():
        q = model(torch.as_tensor(X)).cpu().numpy()   # (N,H,Q) normalize
    return {"quantiles": _inv(q, scaler, target_col),
            "levels": np.asarray(quantiles)}


# ─────────────────────────────────────────────────────────────
# 4) Conformal Prediction — CQR (Conformalized Quantile Regression)
# ─────────────────────────────────────────────────────────────
def cqr_calibrate(cal_quant: np.ndarray, cal_y: np.ndarray,
                  levels: np.ndarray, alpha: float) -> np.ndarray:
    """
    Split-conformal düzeltmesi (Romano ve ark., CQR).
    cal_quant: (Nc,H,Q) kalibrasyon kuantil tahminleri (orijinal birim)
    cal_y:     (Nc,H)   kalibrasyon gerçekleri
    levels:    (Q,)     kuantil seviyeleri (alt ve üst uçları içermeli)
    Döndürür: her ufuk için düzeltme payı q_hat (H,) — alt/üst aralığa eklenir.
    """
    lo_i, hi_i = int(np.argmin(levels)), int(np.argmax(levels))
    lo = cal_quant[..., lo_i]              # (Nc,H)
    hi = cal_quant[..., hi_i]
    # konformite skoru: aralığın dışına ne kadar çıkıldığı
    score = np.maximum(lo - cal_y, cal_y - hi)   # (Nc,H)
    Nc = score.shape[0]
    k = int(np.ceil((Nc + 1) * (1 - alpha)))
    k = min(max(k, 1), Nc)
    q_hat = np.sort(score, axis=0)[k - 1]        # (H,)
    return q_hat


def cqr_apply(test_quant: np.ndarray, levels: np.ndarray, q_hat: np.ndarray) -> Dict:
    """CQR düzeltmesini test kuantillerine uygular → garantili kapsama aralığı."""
    lo_i, hi_i = int(np.argmin(levels)), int(np.argmax(levels))
    lo = test_quant[..., lo_i] - q_hat[None, :]
    hi = test_quant[..., hi_i] + q_hat[None, :]
    return {"interval": np.stack([lo, hi], axis=-1),   # (N,H,2)
            "quantiles": test_quant, "levels": levels}


# ─────────────────────────────────────────────────────────────
# 5) Naive taban
# ─────────────────────────────────────────────────────────────
def naive_predict(df, horizons, target_col: str, seasonal_period: int = 24,
                  lookback: int = 72) -> Dict:
    """
    Persistence ve mevsimsel-naive nokta tahminleri (orijinal birim).
    data.make_windows ile aynı hizalama: pencere i, df satırları [i, i+lookback) girdisini
    kullanır; h-adım ileri hedef satır i+lookback+h-1'dir.
      persistence[i,h]    = y[i+lookback-1]
      seasonal_naive[i,h] = y[i+lookback-1+h - m*ceil(h/m)]  (hedefle aynı saat, son gözlenen gün)
    Döndürür: {'persistence': (N,H), 'seasonal_naive': (N,H)} — N, make_windows ile aynı.
    """
    y = df[target_col].to_numpy(dtype=float)
    H = len(horizons)
    N = len(y) - lookback - max(horizons) + 1
    if N <= 0:
        raise ValueError("yetersiz veri")
    i = np.arange(N)
    last = i + lookback - 1
    pers = np.repeat(y[last][:, None], H, axis=1)
    seas = np.empty((N, H))
    for j, h in enumerate(horizons):
        k = int(np.ceil(h / seasonal_period))
        seas[:, j] = y[last + h - seasonal_period * k]
    return {"persistence": pers, "seasonal_naive": seas}


# ─────────────────────────────────────────────────────────────
# yardımcı
# ─────────────────────────────────────────────────────────────
def _inv(arr: np.ndarray, scaler, target_col: str) -> np.ndarray:
    """Normalize edilmiş hedefi orijinal birime döndürür (min-maks tersi)."""
    if scaler is None:
        return arr
    return scaler.inverse_col(arr, target_col)
