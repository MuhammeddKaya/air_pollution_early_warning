# -*- coding: utf-8 -*-
"""
run_experiment.py — Deney matrisini (Tablo M1) yürüten ana akış.

Akış:  veri hazırla → (öznitelik seti × omurga × UQ) döngüsü → öngörü dağılımı →
       aşım olasılığı → dört-eksenli metrikler → sonuç tablosu → (transfer).

Bu bir İSKELETTİR: eğitim döngüsü (uq.train_one) ve TFT/PCMCI stubları doldurulunca
uçtan uca çalışır. Metrik/conformal/exceedance kısımları gerçek implementasyonludur.

Çalıştırma:
    python -m pm25_early_warning.run_experiment
"""
from __future__ import annotations
import os
import json
import itertools
import numpy as np

from . import config as C
from . import data as D
from . import models as M
from . import uq as U
from . import exceedance as EX
from . import metrics as MET


def _thresholds(cfg: C.ExperimentConfig, y_train) -> list:
    """Eşikleri (µg/m³) toplar. Veriye dayalı yüzdelikler YALNIZ train hedefinden hesaplanır."""
    t = [cfg.thr.who_agg, *cfg.thr.who_interim]
    if cfg.thr.national:
        t.append(cfg.thr.national)
    y = np.asarray(y_train, float)
    t += [float(np.percentile(y, p)) for p in cfg.thr.data_driven_pct]
    return sorted(set(round(x, 2) for x in t))


def torch_seed(seed):
    import torch
    torch.manual_seed(seed)


def _events_for_threshold(Y_true_orig: np.ndarray, tau: float) -> np.ndarray:
    """Gerçek aşım olayları (N,H) {0,1}."""
    return (Y_true_orig > tau).astype(int)


def run_single(cfg: C.ExperimentConfig, feature_set: str, backbone: str,
               uq_method: str, causal_parents=None, seed: int = 0) -> dict:
    """Tek (öznitelik × omurga × UQ) kombinasyonunu çalıştırır ve metrik döndürür."""
    # 1) Veri
    cfg.data.feature_set = feature_set
    cfg.model.backbone = backbone
    cfg.uq.method = uq_method
    prep = D.prepare(cfg, causal_parents=causal_parents)
    X, Y = prep["X"], prep["Y"]
    sp = prep["splits"]; scaler = prep["scaler"]
    Xtr, Ytr = X[sp["train"]], Y[sp["train"]]
    Xva, Yva = X[sp["val"]], Y[sp["val"]]
    Xte, Yte = X[sp["test"]], Y[sp["test"]]
    n_feat, n_h = X.shape[2], Y.shape[1]

    # 2) Model + UQ → öngörü temsili (test)
    head = {"ensemble": "gaussian", "mc_dropout": "gaussian",
            "quantile": "quantile", "conformal": "quantile"}.get(uq_method, "gaussian")

    # --- eğitim (stub: train_one doldurulacak) ---
    if uq_method == "ensemble":
        models = []
        for k in range(cfg.uq.ensemble_size):
            m = M.build_model(cfg.model, n_feat, n_h, head)
            m = U.train_one(m, Xtr, Ytr, Xva, Yva, cfg.train, head, seed=seed * 100 + k)
            models.append(m)
        np.random.seed(seed)
        pred = U.deep_ensemble_predict(models, Xte, scaler, C.TARGET)
    elif uq_method == "mc_dropout":
        m = M.build_model(cfg.model, n_feat, n_h, head)
        m = U.train_one(m, Xtr, Ytr, Xva, Yva, cfg.train, head, seed=seed)
        torch_seed(seed)
        pred = U.mc_dropout_predict(m, Xte, scaler, C.TARGET, cfg.uq.mc_samples)
    elif uq_method in ("quantile", "conformal"):
        m = M.build_model(cfg.model, n_feat, n_h, head)
        m = U.train_one(m, Xtr, Ytr, Xva, Yva, cfg.train, head,
                        quantiles=cfg.model.quantiles, seed=seed)
        pred = U.quantile_predict(m, Xte, scaler, C.TARGET, cfg.model.quantiles)
        if uq_method == "conformal":
            cal = U.quantile_predict(m, Xva, scaler, C.TARGET, cfg.model.quantiles)
            cal_y = scaler.inverse_col(Yva, C.TARGET)
            q_hat = U.cqr_calibrate(cal["quantiles"], cal_y,
                                    cal["levels"], cfg.uq.conformal_alpha)
            pred.update(U.cqr_apply(pred["quantiles"], pred["levels"], q_hat))
    else:
        raise ValueError(uq_method)

    # 3) Metrikler (orijinal birimde)
    Yte_orig = scaler.inverse_col(Yte, C.TARGET)
    Ytr_orig = scaler.inverse_col(Ytr, C.TARGET)
    res = {"feature_set": feature_set, "backbone": backbone, "uq": uq_method, "seed": seed,
           "point": {}, "prob": {}, "event": {}}

    # nokta tahmini (dağılım medyanı/ortalaması)
    if "samples" in pred:
        yhat = np.median(pred["samples"], axis=-1)          # (N,H)
    else:
        mid = int(np.argmin(np.abs(pred["levels"] - 0.5)))
        yhat = pred["quantiles"][..., mid]

    # olasılıksal metrikler (tüm UQ yöntemleri için; nominal aralık %90 = 0.05–0.95)
    if "samples" in pred:
        S = pred["samples"]
        lo_all, hi_all = np.quantile(S, 0.05, axis=-1), np.quantile(S, 0.95, axis=-1)
    else:
        lv = pred["levels"]
        lo_all = pred["quantiles"][..., int(np.argmin(lv))]
        hi_all = pred["quantiles"][..., int(np.argmax(lv))]
    for j, h in enumerate(prep["horizons"]):
        yt, yh = Yte_orig[:, j], yhat[:, j]
        res["point"][h] = dict(RMSE=MET.rmse(yt, yh), MAE=MET.mae(yt, yh),
                               R2=MET.r2(yt, yh),
                               MASE=MET.mase(yt, yh, Ytr_orig[:, j]))
        pr = {}
        if "samples" in pred:
            pr["CRPS"] = MET.crps_ensemble(yt, pred["samples"][:, j, :])
        else:
            # CRPS ≈ 2 × ortalama pinball (kuantil skoru; 5 seviye → kaba yaklaşım)
            pr["CRPS_qs"] = 2 * MET.pinball(yt, pred["quantiles"][:, j, :], pred["levels"])
            pr["pinball"] = MET.pinball(yt, pred["quantiles"][:, j, :], pred["levels"])
        pr["PICP90_raw"] = MET.picp(yt, lo_all[:, j], hi_all[:, j])
        pr["MPIW90_raw"] = MET.mpiw(lo_all[:, j], hi_all[:, j])
        if "interval" in pred:        # CQR düzeltmeli aralık
            pr["PICP90_cqr"] = MET.picp(yt, pred["interval"][:, j, 0], pred["interval"][:, j, 1])
            pr["MPIW90_cqr"] = MET.mpiw(pred["interval"][:, j, 0], pred["interval"][:, j, 1])
        res["prob"][h] = pr

    # 4) Aşım olasılığı + olay/erken-uyarı (HER ufuk için) + ufuk-tabanlı uyarı süresi
    taus = _thresholds(cfg, Ytr_orig[:, 0])
    table = EX.build_exceedance_table(pred, taus)
    y_all = prep["df"][C.TARGET].to_numpy(float)
    o_te = prep["origin_rows"][sp["test"]]
    for tau in taus:
        p = table[tau]                     # (N,H)
        ev = _events_for_threshold(Yte_orig, tau)
        per_h = {}
        for j, h in enumerate(prep["horizons"]):
            pj, ej = p[:, j], ev[:, j]
            sc = MET.event_scores(pj, ej, thr=0.5)
            sc.update(base_rate=float(ej.mean()), BSS=MET.brier_skill(pj, ej),
                      ROC_AUC=MET.roc_auc(pj, ej), PR_AUC=MET.pr_auc(pj, ej))
            per_h[h] = sc
        res["event"][tau] = dict(per_horizon=per_h,
                                 lead=MET.lead_time_by_horizon(
                                     p, o_te, y_all, tau, prep["horizons"], thr=0.5))

    return res


def run_naive(cfg: C.ExperimentConfig) -> list:
    """Persistence ve mevsimsel-naive tabanları: aynı pencere/bölme, nokta + (ikili) olay metrikleri."""
    cfg.data.feature_set = "full"
    prep = D.prepare(cfg)
    sp, scaler = prep["splits"], prep["scaler"]
    hs = prep["horizons"]
    nv = U.naive_predict(prep["df"], hs, C.TARGET, lookback=cfg.data.lookback)
    Yte = scaler.inverse_col(prep["Y"][sp["test"]], C.TARGET)
    Ytr = scaler.inverse_col(prep["Y"][sp["train"]], C.TARGET)
    y_all = prep["df"][C.TARGET].to_numpy(float)
    o_te = prep["origin_rows"][sp["test"]]
    taus = _thresholds(cfg, Ytr[:, 0])
    out = []
    for name, full in nv.items():
        yhat = full[sp["test"]]
        res = {"feature_set": "full", "backbone": "naive", "uq": name, "seed": 0,
               "point": {}, "prob": {}, "event": {}}
        for j, h in enumerate(hs):
            res["point"][h] = dict(RMSE=MET.rmse(Yte[:, j], yhat[:, j]), MAE=MET.mae(Yte[:, j], yhat[:, j]),
                                   R2=MET.r2(Yte[:, j], yhat[:, j]),
                                   MASE=MET.mase(Yte[:, j], yhat[:, j], Ytr[:, j]))
        for tau in taus:
            p = (yhat > tau).astype(float)          # deterministik tahmin → 0/1 "olasılık"
            ev = _events_for_threshold(Yte, tau)
            per_h = {}
            for j, h in enumerate(hs):
                sc = MET.event_scores(p[:, j], ev[:, j], thr=0.5)
                sc.update(base_rate=float(ev[:, j].mean()), BSS=MET.brier_skill(p[:, j], ev[:, j]),
                          ROC_AUC=MET.roc_auc(p[:, j], ev[:, j]), PR_AUC=MET.pr_auc(p[:, j], ev[:, j]))
                per_h[h] = sc
            res["event"][tau] = dict(per_horizon=per_h,
                                     lead=MET.lead_time_by_horizon(p, o_te, y_all, tau, hs, thr=0.5))
        out.append(res)
    return out


# ─────────────────────────────────────────────────────────────
# İş listesi + komut satırı (TRUBA SLURM dizisi ile uyumlu)
# ─────────────────────────────────────────────────────────────
def job_list(backbones, feature_sets, uq_methods, seeds) -> list:
    """Sıralı iş listesi. 1. iş = naive tabanlar; sonrası omurga×öznitelik×UQ×tohum."""
    jobs = [("naive", "full", "naive", 0)]
    jobs += list(itertools.product(backbones, feature_sets, uq_methods, seeds))
    return jobs


def _save(out_dir: str, tag: str, obj) -> str:
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    path = os.path.join(out_dir, "runs", f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def run_job(cfg: C.ExperimentConfig, job) -> bool:
    """Tek işi çalıştırır; sonuç dosyası varsa atlar (yeniden başlatılabilir). Başarıyı döndürür."""
    backbone, fset, uq_method, seed = job
    tag = "naive" if backbone == "naive" else f"{backbone}_{fset}_{uq_method}_s{seed}"
    path = os.path.join(cfg.out_dir, "runs", f"{tag}.json")
    if os.path.exists(path):
        print(f"[SKIP] {tag} (sonuç var)"); return True
    try:
        r = run_naive(cfg) if backbone == "naive" else run_single(cfg, fset, backbone, uq_method, None, seed)
        _save(cfg.out_dir, tag, r)
        print(f"[OK] {tag}"); return True
    except Exception as e:
        import traceback
        _save(cfg.out_dir, tag + ".ERROR", {"error": repr(e), "trace": traceback.format_exc()})
        print(f"[ERR] {tag}: {e}"); return False


def cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="PM2.5 erken-uyarı deney koşucusu (Aşama A)")
    ap.add_argument("--data", default="data/last_dataset_Cok_Onemli.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backbones", default="lstm,gru")
    ap.add_argument("--features", default="full,corr")
    ap.add_argument("--uq", default="ensemble,mc_dropout,quantile,conformal")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ens", type=int, default=5, help="ensemble üye sayısı")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list-jobs", action="store_true", help="iş listesini yazdır ve çık")
    ap.add_argument("--job", type=int, default=None, help="yalnızca bu numaralı işi (1 tabanlı) çalıştır")
    a = ap.parse_args(argv)

    csv = lambda x: tuple(v for v in x.split(",") if v)
    bb, fs, uqs = csv(a.backbones), csv(a.features), csv(a.uq)
    seeds = tuple(int(v) for v in csv(a.seeds))
    bad = [b for b in bb if b not in ("lstm", "gru")] + [f for f in fs if f not in ("full", "corr")]
    if bad:
        raise SystemExit(f"Aşama A'da desteklenmeyen seçenek(ler): {bad} "
                         "(tft ve causal henüz yazılmadı — Aşama B)")
    jobs = job_list(bb, fs, uqs, seeds)
    if a.list_jobs:
        for i, j in enumerate(jobs, 1): print(i, *j)
        print(f"# toplam {len(jobs)} iş"); return

    cfg = C.default_config()
    cfg.data.path, cfg.out_dir = a.data, a.out
    cfg.model.hidden, cfg.model.layers = a.hidden, a.layers
    cfg.train.epochs, cfg.train.patience = a.epochs, a.patience
    cfg.train.batch_size, cfg.train.device = a.batch, a.device
    cfg.uq.ensemble_size = a.ens
    todo = [jobs[a.job - 1]] if a.job else jobs
    ok = [run_job(cfg, j) for j in todo]
    if not all(ok): raise SystemExit(1)


def main(cfg: C.ExperimentConfig = None):
    """Eski giriş noktası (geri uyumluluk): cfg'deki matrisi sırayla koşturur, results.json yazar."""
    cfg = cfg or C.default_config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    all_results = []
    for feature_set, backbone, uq_method in itertools.product(
            cfg.feature_sets, cfg.backbones, cfg.uq_methods):
        if backbone == "tft" or feature_set == "causal":
            continue
        try:
            all_results.append(run_single(cfg, feature_set, backbone, uq_method, None))
            print(f"[OK] {feature_set:7s} · {backbone:5s} · {uq_method}")
        except Exception as e:
            print(f"[ERR] {feature_set}·{backbone}·{uq_method}: {e}")
    with open(os.path.join(cfg.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=float)
    return all_results


if __name__ == "__main__":
    cli()
