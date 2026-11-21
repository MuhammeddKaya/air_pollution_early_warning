# -*- coding: utf-8 -*-
"""
causal.py — Nedensel keşif katmanı (Yön 1 · RQ4).

Amaç: meteoroloji/NASA → PM2.5 nedensel ebeveynlerini bulup 'causal' öznitelik
setini beslemek ve kış/yaz mevsimsel karşıtlığını raporlamak.

Yöntemler:
  - PCMCI+ (tigramite) — gecikmeli, koşullu-bağımsızlık tabanlı.
  - CCM (Convergent Cross Mapping) — doğrusal-olmayan ikili nedensel güç + surrogate.
Her ikisi de opsiyonel bağımlılık; yoksa net hata verir.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


def discover_pcmci(df, var_cols: List[str], target: str,
                   tau_max: int = 24, alpha: float = 0.05,
                   season: Optional[str] = None) -> Dict:
    """
    PCMCI+ ile nedensel ebeveynleri keşfeder.
    season: None | 'winter' | 'summer' (mevsimsel alt-küme).
    Döndürür: {'parents': [...], 'lags': {...}, 'graph': ...}
    TODO: tigramite entegrasyonu.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
    except Exception as e:
        raise ImportError(
            "PCMCI için 'tigramite' gerekli: pip install tigramite"
        ) from e

    sub = _season_subset(df, season)
    data = sub[[target] + [c for c in var_cols if c != target]].to_numpy(float)
    var_names = [target] + [c for c in var_cols if c != target]
    dataframe = pp.DataFrame(data, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=alpha)
    # hedefin (indeks 0) anlamlı ebeveynlerini derle
    parents, lags = [], {}
    val = results["graph"]
    # TODO: results yapısından hedef ebeveynlerini çıkarın (sürüm bağımlı).
    return {"parents": parents, "lags": lags, "raw": results, "season": season}


def ccm_strength(df, x_col: str, target: str, E: int = 3, tau: int = 1,
                 n_surrogate: int = 100, season: Optional[str] = None) -> Dict:
    """
    CCM ile x_col → target nedensel gücü (cross-map skill ρ) + surrogate testi.
    Döndürür: {'rho': float, 'p_surrogate': float}
    TODO: pyEDM veya elle CCM; surrogate (mevsimsel/twin) testi.
    """
    try:
        import pyEDM  # noqa: F401
    except Exception as e:
        raise ImportError("CCM için 'pyEDM' önerilir: pip install pyEDM") from e
    # TODO: pyEDM.CCM çağrısı + surrogate dağılımına karşı p-değeri.
    raise NotImplementedError("ccm_strength: pyEDM entegrasyonu (stub).")


def seasonal_causal_report(df, var_cols: List[str], target: str) -> Dict:
    """Kış vs yaz nedensel ebeveyn karşılaştırması (Tablo T5 / Şekil F8)."""
    out = {}
    for s in ("winter", "summer"):
        try:
            out[s] = discover_pcmci(df, var_cols, target, season=s)["parents"]
        except NotImplementedError:
            out[s] = []   # stub aşamasında boş
    out["stable"] = sorted(set(out.get("winter", [])) & set(out.get("summer", [])))
    out["winter_only"] = sorted(set(out.get("winter", [])) - set(out.get("summer", [])))
    return out


def _season_subset(df, season: Optional[str]):
    if season is None:
        return df
    if season == "winter":
        return df[df["AY"].isin([12, 1, 2])]
    if season == "summer":
        return df[df["AY"].isin([6, 7, 8])]
    raise ValueError(season)
