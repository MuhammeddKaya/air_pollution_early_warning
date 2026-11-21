# -*- coding: utf-8 -*-
"""
aggregate.py — results/runs/*.json dosyalarını birleştirir, tohumlar üzerinden ort ± std alır.
Kullanım:  python aggregate.py results            (çıktı: results/summary_*.csv)
"""
import sys, glob, json, os
import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "results"
files = sorted(glob.glob(os.path.join(out, "runs", "*.json")))
runs = []
for f in files:
    if f.endswith(".ERROR.json"):
        print("HATA DOSYASI:", os.path.basename(f)); continue
    r = json.load(open(f, encoding="utf-8"))
    runs += r if isinstance(r, list) else [r]
print(f"{len(files)} dosya, {len(runs)} koşu")

pt, pr, ev, ld = [], [], [], []
for r in runs:
    key = dict(backbone=r["backbone"], features=r["feature_set"], uq=r["uq"], seed=r["seed"])
    for h, d in r["point"].items():  pt.append({**key, "h": int(h), **d})
    for h, d in r["prob"].items():   pr.append({**key, "h": int(h), **d})
    for tau, e in r["event"].items():
        for h, d in e["per_horizon"].items():
            ev.append({**key, "tau": float(tau), "h": int(h), **d})
        ld.append({**key, "tau": float(tau), **e["lead"]})

def summarize(rows, by, name):
    if not rows: return
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in by + ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(by)[num]
    res = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_seeds"))
    res.to_csv(os.path.join(out, f"summary_{name}.csv"))
    print("yazıldı:", f"summary_{name}.csv", len(res), "satır")

b = ["backbone", "features", "uq"]
summarize(pt, b + ["h"], "point")
summarize(pr, b + ["h"], "prob")
summarize(ev, b + ["tau", "h"], "event")
summarize(ld, b + ["tau"], "lead")
