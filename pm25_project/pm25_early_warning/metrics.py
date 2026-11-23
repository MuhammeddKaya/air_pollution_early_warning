# -*- coding: utf-8 -*-
"""
metrics.py — Dört-eksenli değerlendirme metrikleri (saf numpy, gerçek implementasyon).

Aileler:
  (A) Nokta          : RMSE, MAE, R², MASE
  (B) Olasılıksal    : CRPS (gaussian/ensemble), pinball, PICP, MPIW, PIT, reliability
  (C) Olay/erken-uyarı: POD, FAR, CSI, F1, Brier, BSS, ROC-AUC, PR-AUC, lead time
  (D) Karar          : maliyet-kayıp (cost–loss) değer eğrisi
Ek: Diebold–Mariano testi.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# (A) NOKTA
# ═══════════════════════════════════════════════════════════════
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mae(y, yhat):  return float(np.mean(np.abs(y - yhat)))

def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mase(y, yhat, y_train, m: int = 24):
    """Ölçekten bağımsız hata; payda = eğitimde mevsimsel-naive MAE."""
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(mae(y, yhat) / denom) if denom > 0 else float("nan")


# ═══════════════════════════════════════════════════════════════
# (B) OLASILIKSAL
# ═══════════════════════════════════════════════════════════════
def crps_gaussian(y, mu, sigma):
    """CRPS kapalı form (Gneiting & Raftery 2007)."""
    from math import sqrt, pi
    sigma = np.maximum(sigma, 1e-9)
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + _erf_vec(z / np.sqrt(2)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * pi)
    return float(np.mean(sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(pi))))

def crps_ensemble(y, samples):
    """
    Örnek-tabanlı CRPS tahmincisi:
      CRPS = E|X - y| - 0.5 E|X - X'|
    y:(N,), samples:(N,S)
    """
    N, S = samples.shape
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)
    # E|X - X'| — O(S log S) sıralama ile
    ss = np.sort(samples, axis=1)
    idx = np.arange(1, S + 1)
    term2 = (2.0 / (S * S)) * np.sum((2 * idx - S - 1) * ss, axis=1)
    return float(np.mean(term1 - 0.5 * term2))

def pinball(y, q_pred, levels):
    """y:(N,), q_pred:(N,Q), levels:(Q,) → ortalama pinball kaybı."""
    err = y[:, None] - q_pred
    q = np.asarray(levels)[None, :]
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))

def picp(y, lo, hi):
    """Prediction Interval Coverage Probability — nominal ile karşılaştırın."""
    return float(np.mean((y >= lo) & (y <= hi)))

def mpiw(lo, hi):
    """Mean Prediction Interval Width (keskinlik; küçük daha iyi)."""
    return float(np.mean(hi - lo))

def pit_values(y, samples):
    """PIT = F(y); kalibre modelde ~Uniform(0,1). samples:(N,S)."""
    return np.mean(samples <= y[:, None], axis=1)

def reliability_curve(prob, event, n_bins: int = 10):
    """
    Güvenilirlik (kalibrasyon) eğrisi: tahmini olasılık vs gözlenen frekans.
    prob:(N,) [0,1], event:(N,) {0,1}. Döndürür (bin_merkez, gözlenen_frek, sayım).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    cx, fy, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        cx.append(prob[m].mean()); fy.append(event[m].mean()); cnt.append(int(m.sum()))
    return np.array(cx), np.array(fy), np.array(cnt)


# ═══════════════════════════════════════════════════════════════
# (C) OLAY / ERKEN-UYARI
# ═══════════════════════════════════════════════════════════════
def confusion(prob, event, thr: float = 0.5):
    """Olasılık → ikili karar (thr) → TP,FP,FN,TN."""
    pred = prob >= thr
    ev = event.astype(bool)
    TP = int(np.sum(pred & ev)); FP = int(np.sum(pred & ~ev))
    FN = int(np.sum(~pred & ev)); TN = int(np.sum(~pred & ~ev))
    return TP, FP, FN, TN

def event_scores(prob, event, thr: float = 0.5) -> Dict[str, float]:
    TP, FP, FN, TN = confusion(prob, event, thr)
    eps = 1e-12
    pod = TP / (TP + FN + eps)                 # hit rate / recall
    far = FP / (TP + FP + eps)                 # false alarm ratio
    csi = TP / (TP + FP + FN + eps)            # critical success index
    prec = TP / (TP + FP + eps)
    f1 = 2 * prec * pod / (prec + pod + eps)
    return dict(POD=pod, FAR=far, CSI=csi, F1=f1, TP=TP, FP=FP, FN=FN, TN=TN)

def brier(prob, event):
    return float(np.mean((prob - event) ** 2))

def brier_skill(prob, event):
    """BSS = 1 - BS/BS_clim (iklimsel taban = olay taban oranı)."""
    base = np.mean(event)
    bs_clim = np.mean((base - event) ** 2)
    return float(1 - brier(prob, event) / bs_clim) if bs_clim > 0 else float("nan")

def roc_auc(prob, event):
    """ROC-AUC (Mann–Whitney U ile; sklearn'e gerek yok)."""
    return _auc_rank(prob, event)

def pr_auc(prob, event):
    """
    PR-AUC, average precision (AP) olarak: AP = Σ (R_k − R_{k−1}) · P_k
    (sklearn.metrics.average_precision_score ile aynı tanım). Eğri recall=0'dan başlar;
    bağlı (eşit) olasılıklar tek eşik sayılır → 0/1 olasılıklı naive tabanlar için de doğru.
    Rastgele tahmincinin beklenen değeri ≈ taban oran (base_rate).
    """
    prob = np.asarray(prob, dtype=float)
    ev = np.asarray(event).astype(bool)
    n_pos = int(ev.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-prob, kind="mergesort")
    p_s, e_s = prob[order], ev[order]
    last = np.r_[np.where(np.diff(p_s) != 0)[0], len(p_s) - 1]   # her farklı eşiğin son indeksi
    tps = np.cumsum(e_s)[last]
    fps = (last + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

def lead_time(prob_series: np.ndarray, event_series: np.ndarray,
              thr: float = 0.5, max_lag: int = 72) -> Dict[str, float]:
    """
    Kazanılan uyarı süresi: her gerçek aşım olayı için, olaydan ÖNCE olasılığın
    thr'yi ilk geçtiği ana kadarki saat farkı.
    prob_series, event_series: (T,) zaman dizisi (tek ufuk/eşik için).
    Döndürür ortalama/medyan lead time (saat).
    """
    T = len(event_series)
    onsets = np.where((event_series[1:] == 1) & (event_series[:-1] == 0))[0] + 1
    leads = []
    for o in onsets:
        w0 = max(0, o - max_lag)
        warned = np.where(prob_series[w0:o] >= thr)[0]
        if warned.size:
            leads.append(o - (w0 + warned[0]))
        else:
            leads.append(0)   # zamanında uyarı yok
    leads = np.array(leads) if leads else np.array([np.nan])
    return dict(mean_lead=float(np.nanmean(leads)),
                median_lead=float(np.nanmedian(leads)),
                n_events=int(len(onsets)))


def lead_time_by_horizon(prob: np.ndarray, origin_rows: np.ndarray, y_all: np.ndarray,
                         tau: float, horizons, thr: float = 0.5) -> Dict[str, float]:
    """
    Kazanılan uyarı süresi (ufuk tabanlı, dürüst tanım).
    prob:(N,H) test pencerelerinin aşım olasılığı; origin_rows:(N,) tahmin anı (satır);
    y_all:(T,) tüm seride gerçek PM2.5 (orijinal birim).
    Olay başlangıcı r: y[r]>tau ve y[r-1]<=tau. r için, uyarı h saat önce (origin=r-h)
    p_h>=thr ise 'h saat önceden uyarıldı'. Süre = uyarı verilen EN UZUN ufuk.
    Yalnızca tüm ufuk-originleri test kümesinde olan olaylar sayılır (blok sınırı yanlılığı yok).
    Döndürür: n_events, detect_rate (>=1 ufukta uyarı), mean/median_lead_detected (saat; yalnızca
    yakalananlar), mean_lead_all (kaçırılan=0), ayrıca ufuk bazlı yakalama oranı.
    """
    pos = {int(o): i for i, o in enumerate(origin_rows)}
    onsets = np.where((y_all[1:] > tau) & (y_all[:-1] <= tau))[0] + 1
    leads, per_h = [], {h: [] for h in horizons}
    for r in onsets:
        idxs = [pos.get(int(r - h)) for h in horizons]
        if any(i is None for i in idxs):
            continue
        hit = [prob[i, j] >= thr for j, i in enumerate(idxs)]
        for h, ok in zip(horizons, hit):
            per_h[h].append(ok)
        leads.append(max([h for h, ok in zip(horizons, hit) if ok], default=0))
    leads = np.asarray(leads, dtype=float)
    out = dict(n_events=int(len(leads)))
    if len(leads) == 0:
        return {**out, "detect_rate": float("nan"), "mean_lead_detected": float("nan"),
                "median_lead_detected": float("nan"), "mean_lead_all": float("nan")}
    det = leads[leads > 0]
    out.update(detect_rate=float((leads > 0).mean()),
               mean_lead_detected=float(det.mean()) if len(det) else float("nan"),
               median_lead_detected=float(np.median(det)) if len(det) else float("nan"),
               mean_lead_all=float(leads.mean()))
    for h in horizons:
        out[f"detect_h{h}"] = float(np.mean(per_h[h])) if per_h[h] else float("nan")
    return out


# ═══════════════════════════════════════════════════════════════
# (D) KARAR — maliyet-kayıp
# ═══════════════════════════════════════════════════════════════
def cost_loss_value(prob, event, cost: float, loss: float, thr: Optional[float] = None):
    """
    Cost–loss karar modeli. Uyarınca önlem maliyeti C, kaçırılan olay kaybı L.
    Değer skoru: V = (E_clim - E_forecast) / (E_clim - E_perfect), 1=mükemmel.
    """
    ev = event.astype(bool)
    base = np.mean(ev)
    thr = thr if thr is not None else (cost / loss)   # optimal karar eşiği
    pred = prob >= thr
    # beklenen maliyetler
    E_f = np.mean(pred * cost + (~pred) * ev * loss)
    E_clim = min(cost, base * loss)                   # ya hep önlem ya hiç
    E_perf = base * cost                              # kusursuz tahmin
    denom = (E_clim - E_perf)
    return float((E_clim - E_f) / denom) if denom != 0 else float("nan")

def cost_loss_curve(prob, event, ratios=np.linspace(0.02, 0.98, 25)):
    """Farklı C/L oranlarında değer skoru eğrisi (F7 için)."""
    return ratios, np.array([cost_loss_value(prob, event, r, 1.0) for r in ratios])


# ═══════════════════════════════════════════════════════════════
# İSTATİSTİKSEL TEST
# ═══════════════════════════════════════════════════════════════
def diebold_mariano(e1, e2, h: int = 1, power: int = 2):
    """
    İki modelin tahmin hatalarını karşılaştırır (DM testi).
    e1,e2: hata dizileri. Döndürür (DM istatistiği, ~p-değeri).
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    # h-adım için gecikmeli otokovaryans düzeltmesi (Newey-West basit)
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k])[0, 1]
        var += 2 * (1 - k / h) * cov
    dm = dbar / np.sqrt(var / n) if var > 0 else float("nan")
    # normal yaklaşım p-değeri (iki yönlü)
    p = 2 * (1 - _stdnorm_cdf(abs(dm)))
    return float(dm), float(p)


# ═══════════════════════════════════════════════════════════════
# yardımcılar
# ═══════════════════════════════════════════════════════════════
def _trapz(y, x):
    """Trapez integrali — NumPy 1.x (trapz) ve 2.x (trapezoid) uyumlu."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if fn is not None:
        return fn(y, x)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

def _erf_vec(x):
    from math import erf
    return np.vectorize(erf)(x)

def _stdnorm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))

def _auc_rank(prob, event):
    """AUC = P(prob(pos) > prob(neg)); rank tabanlı."""
    ev = event.astype(bool)
    pos, neg = prob[ev], prob[~ev]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob) + 1)
    r_pos = ranks[ev].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)
