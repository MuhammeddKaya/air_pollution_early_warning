# -*- coding: utf-8 -*-
"""
models.py — Omurga modeller ve tahmin başlıkları.

Gerçek: LSTM/GRU + (a) nokta, (b) ortalama-varyans (Gaussian), (c) kuantil başlıkları.
Stub (TODO): TFT / PatchTST — kütüphane (pytorch-forecasting / neuralforecast) ile bağlanır.
"""
from __future__ import annotations
from typing import Tuple

try:
    import torch
    import torch.nn as nn
except Exception:  # skeleton compile-check ortamında torch olmayabilir
    torch = None
    nn = object  # type: ignore


# ─────────────────────────────────────────────────────────────
# Başlık tipleri
# ─────────────────────────────────────────────────────────────
class PointHead(nn.Module):
    """Tek nokta tahmini (ufuk sayısı kadar çıktı)."""
    def __init__(self, in_dim: int, n_h: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_h)

    def forward(self, z):
        return self.fc(z)                      # (B, H)


class GaussianHead(nn.Module):
    """Ortalama + log-varyans → Gaussian öngörü dağılımı (ensemble/mc-dropout ile)."""
    def __init__(self, in_dim: int, n_h: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, n_h)
        self.logvar = nn.Linear(in_dim, n_h)

    def forward(self, z):
        return self.mu(z), self.logvar(z)      # (B,H),(B,H)


class QuantileHead(nn.Module):
    """Her ufuk için Q kuantil çıktı (pinball kaybı ile)."""
    def __init__(self, in_dim: int, n_h: int, n_q: int):
        super().__init__()
        self.n_h, self.n_q = n_h, n_q
        self.fc = nn.Linear(in_dim, n_h * n_q)

    def forward(self, z):
        out = self.fc(z).view(-1, self.n_h, self.n_q)
        # monotonluk için kümülatif softplus (kuantil çaprazlaşmasını önler)
        base = out[..., :1]
        inc = torch.nn.functional.softplus(out[..., 1:])
        return torch.cat([base, base + torch.cumsum(inc, dim=-1)], dim=-1)  # (B,H,Q)


# ─────────────────────────────────────────────────────────────
# Omurga
# ─────────────────────────────────────────────────────────────
class RecurrentBackbone(nn.Module):
    """LSTM veya GRU omurga + seçilebilir başlık."""
    def __init__(self, n_feat: int, n_h: int, head: str = "gaussian",
                 kind: str = "gru", hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2, n_q: int = 5):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(n_feat, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)        # MC-dropout için çıkarımda açık tutulur
        self.head_type = head
        if head == "point":
            self.head = PointHead(hidden, n_h)
        elif head == "gaussian":
            self.head = GaussianHead(hidden, n_h)
        elif head == "quantile":
            self.head = QuantileHead(hidden, n_h, n_q)
        else:
            raise ValueError(head)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B, L, hidden)
        z = self.drop(out[:, -1])     # son adım
        return self.head(z)


def build_model(cfg_model, n_feat: int, n_h: int, head: str):
    """config.ModelConfig'ten model kurar."""
    if cfg_model.backbone in ("lstm", "gru"):
        return RecurrentBackbone(
            n_feat=n_feat, n_h=n_h, head=head, kind=cfg_model.backbone,
            hidden=cfg_model.hidden, layers=cfg_model.layers,
            dropout=cfg_model.dropout, n_q=len(cfg_model.quantiles),
        )
    elif cfg_model.backbone in ("tft", "patchtst"):
        # TODO: pytorch-forecasting (TFT) veya neuralforecast (PatchTST) ile bağla.
        #   - TFT doğal kuantil çıktısı verir → QuantileHead yerine kütüphane kuantilleri.
        #   - Girdi/normalizasyon sözleşmesini data.prepare() ile hizala.
        raise NotImplementedError("TFT/PatchTST omurgası: kütüphane entegrasyonu (stub).")
    elif cfg_model.backbone == "naive":
        return None   # persistence/seasonal-naive → uq.py içinde ele alınır
    else:
        raise ValueError(cfg_model.backbone)


# ─────────────────────────────────────────────────────────────
# Kayıp fonksiyonları
# ─────────────────────────────────────────────────────────────
def pinball_loss(pred_q, target, quantiles):
    """
    pred_q: (B, H, Q), target: (B, H), quantiles: (Q,)
    Kuantil (pinball) kaybı ortalaması.
    """
    q = torch.as_tensor(quantiles, dtype=pred_q.dtype, device=pred_q.device)
    err = target.unsqueeze(-1) - pred_q          # (B,H,Q)
    loss = torch.maximum(q * err, (q - 1) * err)
    return loss.mean()


def gaussian_nll(mu, logvar, target):
    """Gaussian negatif log-olabilirlik (ortalama-varyans başlığı için)."""
    inv = torch.exp(-logvar)
    return 0.5 * (logvar + (target - mu) ** 2 * inv).mean()
