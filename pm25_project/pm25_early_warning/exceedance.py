# -*- coding: utf-8 -*-
"""
exceedance.py — Öngörü dağılımını sağlık-eşiği aşım olasılığına çevirir.

Girdi temsili (uq.py'den):
  - 'samples'   : (N,H,S)
  - 'quantiles' : (N,H,Q) + levels
Çıktı: her eşik τ için P(y_{t+h} > τ) ve pencere içi P(max_h y > τ).
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


def prob_from_samples(samples: np.ndarray, tau: float) -> np.ndarray:
    """P(y>τ) = örneklerin τ'yu aşma oranı. samples:(N,H,S) → (N,H)."""
    return (samples > tau).mean(axis=-1)


def prob_from_quantiles(quant: np.ndarray, levels: np.ndarray, tau: float) -> np.ndarray:
    """
    Kuantil tahminlerinden P(y>τ): τ'nun kuantil eğrisi üzerindeki yerini
    doğrusal enterpolasyonla olasılığa çevirir. quant:(N,H,Q) → (N,H).
    """
    order = np.argsort(levels)
    lv = np.asarray(levels)[order]
    q = quant[..., order]                    # (N,H,Q) artan seviyelerle
    N, H, Q = q.shape
    out = np.empty((N, H), dtype=float)
    for i in range(N):
        for j in range(H):
            # τ, q[i,j] içinde nereye düşüyor → CDF(τ) ≈ enterpolasyon
            cdf = np.interp(tau, q[i, j], lv, left=0.0, right=1.0)
            out[i, j] = 1.0 - cdf
    return out


def prob_from_gaussian(mu: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    """Gaussian kapalı-form: P(y>τ) = 1 - Φ((τ-μ)/σ)."""
    from math import erf, sqrt
    z = (tau - mu) / np.maximum(sigma, 1e-9)
    # vektörel Φ
    Phi = 0.5 * (1 + np.vectorize(lambda t: erf(t / sqrt(2)))(z))
    return 1.0 - Phi


def window_exceedance(prob_per_h: np.ndarray) -> np.ndarray:
    """
    Pencere içinde EN AZ BİR aşım olasılığı (bağımsızlık varsayımı — üst sınır):
      P(max_h y>τ) ≈ 1 - Π_h (1 - p_h).
    Not: saatler bağımlı olduğundan bu bir yaklaşımdır; alternatif olarak
    'samples' varsa doğrudan (max_h samples > τ) oranı kullanılmalıdır.
    """
    return 1.0 - np.prod(1.0 - prob_per_h, axis=1)


def window_exceedance_from_samples(samples: np.ndarray, tau: float) -> np.ndarray:
    """Örneklerden doğrudan (tercih edilen): P(herhangi ufukta aşım)."""
    return (samples.max(axis=1) > tau).mean(axis=-1)


def build_exceedance_table(pred: Dict, taus: List[float]) -> Dict[float, np.ndarray]:
    """
    pred: uq.py çıktısı ('samples' veya 'quantiles'+'levels').
    Döndürür: {tau: P(y>τ) (N,H)} ve ayrıca pencere olasılığı '_window' anahtarıyla.
    """
    table = {}
    for tau in taus:
        if "samples" in pred:
            p = prob_from_samples(pred["samples"], tau)
            pw = window_exceedance_from_samples(pred["samples"], tau)
        elif "quantiles" in pred:
            p = prob_from_quantiles(pred["quantiles"], pred["levels"], tau)
            pw = window_exceedance(p)
        else:
            raise ValueError("pred içinde 'samples' veya 'quantiles' bekleniyor")
        table[tau] = p
        table[(tau, "window")] = pw
    return table
