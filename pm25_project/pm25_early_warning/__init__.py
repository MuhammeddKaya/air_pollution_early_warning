# -*- coding: utf-8 -*-
"""
pm25_early_warning — Yön 1 deney iskeleti.
Kalibre olasılıksal PM2.5 aşım erken-uyarısı (Iğdır).

Modüller:
  config      — merkezi konfigürasyon / deney matrisi
  data        — yükleme, ön işleme, öznitelik setleri, pencereleme, bölme
  models      — LSTM/GRU omurga + nokta/gaussian/kuantil başlıkları (+TFT stub)
  uq          — deep ensemble, MC dropout, kuantil, conformal (CQR)
  exceedance  — öngörü dağılımı → aşım olasılığı P(y>τ)
  metrics     — nokta / olasılıksal / olay-tabanlı / karar metrikleri (+DM testi)
  causal      — PCMCI / CCM nedensel keşif (opsiyonel)
  run_experiment — uçtan uca akış
"""
__all__ = ["config", "data", "models", "uq", "exceedance", "metrics", "causal"]
__version__ = "0.1.0-skeleton"
