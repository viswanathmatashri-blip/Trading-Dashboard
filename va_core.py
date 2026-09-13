"""Exact VA trigger helpers copied from Options Simulator Bimodel.py (no Streamlit)."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

try:
    import scipy.stats as si
except Exception:
    si = None


VA_PLAYBOOK = {
    "M1S_WATCH": ("Range · VAH probe — price statistically above value, absorption forming", "WATCH SHORT (VAH fade)"),
    "M1S_ENTRY": ("Range · VAH mean-reversion SHORT — absorption + EFI divergence at/above VAH", "SHORT MEAN-REVERSION (Target: POC → VAL)"),
    "M1S_ADD": ("Range · VAH failed acceptance — close back inside value, ΔV still offered", "ADD SHORT (Retest VAH from below)"),
    "M1L_WATCH": ("Range · VAL probe — price statistically below value, demand absorbing", "WATCH LONG (VAL bounce)"),
    "M1L_ENTRY": ("Range · VAL mean-reversion LONG — absorption + EFI divergence at/below VAL", "LONG MEAN-REVERSION (Target: POC → VAH)"),
    "M1L_ADD": ("Range · VAL failed breakdown — close back inside value, ΔV still bid", "ADD LONG (Retest VAL from above)"),
    "M2L_WATCH": ("Trend · VAH break — price holding above value, force not yet confirmed", "WATCH LONG (VAH acceptance)"),
    "M2L_ENTRY": ("Trend · VAH acceptance LONG — density building above VAH + EFI expansion + ΔV bid", "LONG BREAKOUT (Hold above VAH / trail under box)"),
    "M2L_ADD": ("Trend · VAH retest from above — shallow pullback, ΔV stays non-negative", "ADD LONG (Retest old VAH / new box low)"),
    "M2S_WATCH": ("Trend · VAL break — price holding below value, force not yet confirmed", "WATCH SHORT (VAL acceptance)"),
    "M2S_ENTRY": ("Trend · VAL acceptance SHORT — density building below VAL + EFI expansion + ΔV offered", "SHORT BREAKDOWN (Hold below VAL / trail above box)"),
    "M2S_ADD": ("Trend · VAL retest from below — shallow bounce, ΔV stays non-positive", "ADD SHORT (Retest old VAL / new box high)"),
    "INSIDE": ("Inside value — no edge at the edge of the profile", "NO ENTRY"),
    "CHOP": ("Regime mixed / low efficiency — neither clean range nor trend", "NO ENTRY"),
}

LEVEL_GLYPH = {"++": "⇈", "+": "↑", "-": "↓", "--": "⇊", "=": "→"}


def _series_num(s):
    return pd.to_numeric(s, errors="coerce")


def _ols_slope_p(y):
    y = _series_num(y).dropna().astype(float)
    n = len(y)
    if n < 6:
        return 0.0, 1.0
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    yv = y.values - y.values.mean()
    den = float((x * x).sum())
    if den <= 1e-18:
        return 0.0, 1.0
    sl = float((x * yv).sum() / den)
    resid = yv - sl * x
    dof = n - 2
    se = float(np.sqrt((resid * resid).sum() / max(dof, 1) / den))
    if se <= 1e-18:
        return sl, 0.0
    t = sl / se
    try:
        p = float(2.0 * si.t.sf(abs(t), dof)) if si is not None else float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
    except Exception:
        p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
    return sl, p


def _zscore(last, mean, sd):
    sd = float(sd) if sd and sd == sd else 0.0
    if sd <= 1e-12:
        return 0.0
    return float((last - mean) / sd)


def _signed_delta(d: pd.DataFrame) -> pd.Series:
    if "volume" not in d.columns:
        if "open" in d.columns:
            return _series_num(d["close"]) - _series_num(d["open"])
        return _series_num(d["close"]).diff().fillna(0.0)
    o = _series_num(d["open"]) if "open" in d.columns else _series_num(d["close"]).shift(1)
    sgn = np.sign(_series_num(d["close"]) - o).fillna(0.0)
    return _series_num(d["volume"]).fillna(0.0) * sgn


def _atr_pct(d: pd.DataFrame, n: int = 14) -> float:
    h = _series_num(d["high"] if "high" in d.columns else d["close"])
    l = _series_num(d["low"] if "low" in d.columns else d["close"])
    c = _series_num(d["close"])
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(n).mean()) if tr.notna().any() else 0.0
    px = float(c.iloc[-1]) if c.notna().any() else 1.0
    return atr / max(px, 1.0)


def _efficiency_ratio(d: pd.DataFrame, n: int = 20) -> float:
    c = _series_num(d["close"]).dropna()
    if len(c) < max(8, n // 2):
        return 0.0
    w = c.tail(n)
    net = abs(float(w.iloc[-1] - w.iloc[0]))
    path = float(w.diff().abs().sum())
    return net / max(path, 1e-9)


def _value_density(d: pd.DataFrame, val, vah) -> float:
    if val is None or vah is None or "volume" not in d.columns:
        return 0.0
    c = _series_num(d["close"])
    v = _series_num(d["volume"]).fillna(0.0)
    tot = float(v.sum())
    if tot <= 0:
        return 0.0
    inside = ((c >= float(val)) & (c <= float(vah))).astype(float)
    return float((v * inside).sum() / tot)


def _persist_side(series, thresh, side="above", bars=3) -> bool:
    s = _series_num(series).dropna()
    if len(s) < bars:
        return False
    tail = s.iloc[-bars:]
    if side == "above":
        return bool((tail > thresh).sum() >= max(2, bars - 1))
    return bool((tail < thresh).sum() >= max(2, bars - 1))


def _box_tight(d: pd.DataFrame, n: int = 8, atr_frac: float = 0.85) -> bool:
    if len(d) < n:
        return False
    sl = d.tail(n)
    h = float(_series_num(sl["high"] if "high" in sl.columns else sl["close"]).max())
    l = float(_series_num(sl["low"] if "low" in sl.columns else sl["close"]).min())
    atr = _atr_pct(d) * max(float(_series_num(d["close"]).iloc[-1]), 1.0)
    return (h - l) <= max(atr_frac * atr, 1.0)


def _level4(series, z_hi=1.10, z_mid=0.45) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return "-"
    last = float(s.iloc[-1])
    tail = s.tail(min(20, len(s)))
    sd = float(tail.std(ddof=1) or 0.0) or 1e-9
    z = (last - float(tail.mean())) / sd
    if last >= 0:
        return "++" if z >= z_hi else "+"
    return "--" if z <= -z_hi else "-"


def _price_level4(df) -> str:
    px = pd.to_numeric(df["close"], errors="coerce")
    last = float(px.iloc[-1])
    base = _level4(px)
    vwcol = "vwap_idx" if "vwap_idx" in df.columns else ("vwap" if "vwap" in df.columns else None)
    if vwcol:
        vw = float(df[vwcol].iloc[-1])
        if last >= vw and base in ("-", "--"):
            base = "+"
        if last < vw and base in ("+", "++"):
            base = "-"
        stretch = (last - vw) / max(abs(vw), 1.0)
        if stretch > 0.003 and base == "+":
            base = "++"
        if stretch < -0.003 and base == "-":
            base = "--"
    return base


def compute_session_volume_profile(df: pd.DataFrame, bin_step: float = 2.0, prominence_factor: float = 0.35) -> dict:
    empty = {"ok": False}
    if df is None or df.empty or "volume" not in df.columns:
        return empty
    lo = float(min(df["low"].min(), df["close"].min()))
    hi = float(max(df["high"].max(), df["close"].max()))
    if hi <= lo:
        hi = lo + bin_step
    bin_step = float(bin_step) if bin_step and bin_step > 0 else 5.0
    lo = np.floor(lo / bin_step) * bin_step
    hi = np.ceil(hi / bin_step) * bin_step
    edges = np.arange(lo, hi + bin_step * 0.5, bin_step)
    if len(edges) < 3:
        edges = np.array([lo, lo + bin_step, lo + 2 * bin_step])
    n_bins = len(edges) - 1
    vol_at = np.zeros(n_bins, dtype=float)
    for _, r in df.iterrows():
        v = float(r.get("volume") or 0)
        if v <= 0:
            continue
        l = float(r["low"]); h = float(r["high"])
        if h <= l:
            h = l + 1e-6
        span = h - l
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            overlap = max(0.0, min(h, b) - max(l, a))
            if overlap > 0:
                vol_at[i] += v * (overlap / span)
    mids = (edges[:-1] + edges[1:]) / 2.0
    tot = float(vol_at.sum())
    if tot <= 0:
        return empty
    poc_i = int(np.argmax(vol_at))
    wmean = float(np.sum(mids * vol_at) / tot)
    wstd = float(np.sqrt(max(np.sum(vol_at * (mids - wmean) ** 2) / tot, 0.0)))

    def _expand(seed_i, frac):
        t = float(vol_at.sum())
        tgt = frac * t
        lo_i = hi_i = int(np.clip(seed_i, 0, n_bins - 1))
        acc = float(vol_at[lo_i])
        guard = 0
        while acc < tgt and guard < n_bins + 2:
            guard += 1
            left = float(vol_at[lo_i - 1]) if lo_i > 0 else -1.0
            right = float(vol_at[hi_i + 1]) if hi_i < n_bins - 1 else -1.0
            if right < 0 and left < 0:
                break
            if right >= left:
                hi_i = min(hi_i + 1, n_bins - 1)
                acc += float(vol_at[hi_i])
            else:
                lo_i = max(lo_i - 1, 0)
                acc += float(vol_at[lo_i])
        return {"vah": float(mids[hi_i]), "val": float(mids[lo_i]), "poc": float(mids[seed_i])}

    va70 = _expand(poc_i, 0.70)
    va80 = _expand(poc_i, 0.80)
    # extra fat nodes: local volume peaks away from POC
    zones = []
    poc_vol = float(vol_at[poc_i]) or 1.0
    for i in range(2, n_bins - 2):
        if vol_at[i] <= vol_at[i - 1] or vol_at[i] <= vol_at[i + 1]:
            continue
        if vol_at[i] < 0.38 * poc_vol:
            continue
        if abs(i - poc_i) < 6:
            continue
        loc = _expand(i, 0.70)
        zones.append({
            "tag": f"VA{len(zones)+2}",
            "vah": loc["vah"],
            "val": loc["val"],
            "poc": loc["poc"],
        })
        if len(zones) >= 4:
            break
    return {
        "ok": True,
        "mids": mids,
        "vol": vol_at,
        "poc": float(mids[poc_i]),
        "poc_vol": float(vol_at[poc_i]),
        "wmean": wmean,
        "wstd": wstd,
        "vah1": wmean + wstd,
        "val1": wmean - wstd,
        "vah15": wmean + 1.5 * wstd,
        "val15": wmean - 1.5 * wstd,
        "vah": float(va70["vah"]),
        "val": float(va70["val"]),
        "vah80": float(va80["vah"]),
        "val80": float(va80["val"]),
        "zones": zones,
        "bin_step": bin_step,
    }


def classify_market_regime(dfi: pd.DataFrame, data: dict = None) -> dict:
    out = {
        "regime": "CHOP", "score_range": 0.0, "score_trend": 0.0,
        "eff": 0.0, "atr_pct": 0.0, "density": 0.0, "gex_sign": 0,
        "vwap_z": 0.0, "note": "",
    }
    if dfi is None or getattr(dfi, "empty", True) or len(dfi) < 12:
        return out
    d = dfi.copy()
    px = _series_num(d.get("spot_px", d["close"]))
    last = float(px.iloc[-1])
    atrp = _atr_pct(d)
    eff = _efficiency_ratio(d, 20)
    vp = {}
    try:
        src = d
        if "spot_px" in d.columns and d["spot_px"].notna().sum() >= 8:
            src = pd.DataFrame({
                "open": _series_num(d.get("open", d["close"])),
                "high": _series_num(d.get("high", d["close"])),
                "low": _series_num(d.get("low", d["close"])),
                "close": _series_num(d["spot_px"] if "spot_px" in d.columns else d["close"]),
                "volume": _series_num(d["volume"]) if "volume" in d.columns else 1.0,
            })
        vp = compute_session_volume_profile(src, bin_step=2.0, prominence_factor=0.35)
    except Exception:
        vp = {}
    vah = vp.get("vah") if isinstance(vp, dict) else None
    val = vp.get("val") if isinstance(vp, dict) else None
    dens = _value_density(d, val, vah)
    vw = None
    if "vwap_idx" in d.columns:
        vw = float(_series_num(d["vwap_idx"]).iloc[-1])
    elif "vwap" in d.columns:
        vw = float(_series_num(d["vwap"]).iloc[-1])
    std = float(_series_num(d["vwap_std"]).iloc[-1]) if "vwap_std" in d.columns else float(px.tail(20).std(ddof=1) or 0)
    vz = _zscore(last, vw, std) if vw is not None else 0.0
    gex = 0.0
    if isinstance(data, dict):
        try:
            gex = float(data.get("total_net_gex_oi") or 0)
        except Exception:
            gex = 0.0
    gex_sign = 1 if gex > 0 else (-1 if gex < 0 else 0)
    sess_hi = float(_series_num(d.get("high", d["close"])).max())
    sess_lo = float(_series_num(d.get("low", d["close"])).min())
    sess_rng = max(sess_hi - sess_lo, 1e-6)
    atr_abs = atrp * max(last, 1.0)
    realized_vs_atr = sess_rng / max(atr_abs * math.sqrt(max(len(d), 1) / 14.0), 1e-6)
    s_range = 0.0
    s_trend = 0.0
    s_range += 1.2 if dens >= 0.62 else (0.4 if dens >= 0.50 else -0.4)
    s_range += 1.0 if eff < 0.22 else (0.2 if eff < 0.32 else -0.6)
    s_range += 0.7 if abs(vz) < 1.10 else -0.3
    s_range += 0.6 if gex_sign > 0 else (-0.2 if gex_sign < 0 else 0.1)
    s_range += 0.4 if realized_vs_atr < 1.15 else -0.3
    s_trend += 1.2 if eff >= 0.38 else (0.4 if eff >= 0.28 else -0.5)
    s_trend += 1.0 if dens <= 0.48 else (0.2 if dens <= 0.58 else -0.6)
    s_trend += 0.7 if abs(vz) >= 1.15 else -0.2
    s_trend += 0.6 if gex_sign < 0 else (-0.15 if gex_sign > 0 else 0.05)
    s_trend += 0.5 if realized_vs_atr >= 1.25 else -0.2
    if s_range >= 1.4 and s_range >= s_trend + 0.35:
        regime = "RANGE"
    elif s_trend >= 1.4 and s_trend >= s_range + 0.35:
        regime = "TREND"
    else:
        regime = "CHOP"
    out.update({
        "regime": regime, "score_range": round(s_range, 2), "score_trend": round(s_trend, 2),
        "eff": round(eff, 3), "atr_pct": round(atrp * 100.0, 3), "density": round(dens, 3),
        "gex_sign": gex_sign, "vwap_z": round(vz, 2),
        "vah": float(vah) if vah else None, "val": float(val) if val else None,
        "poc": float(vp.get("poc")) if isinstance(vp, dict) and vp.get("poc") else None,
        "vp": vp if isinstance(vp, dict) else {},
        "note": f"{regime} dens={dens:.2f} eff={eff:.2f} zVWAP={vz:+.2f} GEX={'+' if gex_sign>0 else ('-' if gex_sign<0 else '0')}",
    })
    return out


def _loc_vs_value(last, vah, val, std, atr_abs):
    if vah is None or val is None:
        return "UNKNOWN", 0.0
    buf = max(0.35 * float(std or 0), 0.35 * float(atr_abs or 0), 1.0)
    z_h = (last - float(vah)) / max(float(std) or 1.0, 1e-6)
    z_l = (float(val) - last) / max(float(std) or 1.0, 1e-6)
    if last >= float(vah) + buf:
        return "ABOVE_VAH", z_h
    if last <= float(val) - buf:
        return "BELOW_VAL", z_l
    if last > float(vah):
        return "VAH_EDGE", z_h
    if last < float(val):
        return "VAL_EDGE", z_l
    return "INSIDE", 0.0


def classify_va_setup(dfi: pd.DataFrame, data: dict = None) -> dict:
    empty = {
        "ok": False, "arrows": "→ → → →", "micro": "", "action": "NO ENTRY",
        "key": ("=", "=", "=", "="), "hover": "", "regime": "CHOP",
        "model": "", "price": "=", "delta": "=", "efi": "=", "cvd": "=",
        "obv": "=", "efi_zero": False, "efi_note": "",
    }
    if dfi is None or getattr(dfi, "empty", True) or len(dfi) < 12:
        return empty
    d = dfi.copy()
    px = _series_num(d["spot_px"] if "spot_px" in d.columns else d["close"])
    last = float(px.iloc[-1])
    delta_s = _signed_delta(d)
    efi = _series_num(d["efi13"]) if "efi13" in d.columns else pd.Series(dtype=float)
    cvd = _series_num(d["cvd"]) if "cvd" in d.columns else pd.Series(dtype=float)
    d_arr = _level4(delta_s)
    e_arr = _level4(efi) if len(efi) else "-"
    c_arr = _level4(cvd) if len(cvd) else "-"
    p_arr = _price_level4(d) if "close" in d.columns else "+"

    reg = classify_market_regime(d, data)
    vah, val, poc = reg.get("vah"), reg.get("val"), reg.get("poc")
    std = float(_series_num(d["vwap_std"]).iloc[-1]) if "vwap_std" in d.columns else float(px.tail(20).std(ddof=1) or 0)
    atr_abs = _atr_pct(d) * max(last, 1.0)
    loc, loc_z = _loc_vs_value(last, vah, val, std, atr_abs)

    efi_sl, efi_p = _ols_slope_p(efi.tail(12)) if len(efi) else (0.0, 1.0)
    px_sl, px_p = _ols_slope_p(px.tail(12))
    dv_tail = delta_s.tail(5)
    dv_sum = float(dv_tail.sum()) if len(dv_tail) else 0.0
    dv_abs = float(dv_tail.abs().sum()) if len(dv_tail) else 1.0
    absorb_up = (px_sl > 0 and px_p < 0.12) and (dv_sum <= 0.15 * max(dv_abs, 1.0))
    absorb_dn = (px_sl < 0 and px_p < 0.12) and (dv_sum >= -0.15 * max(dv_abs, 1.0))
    efi_div_up = px_sl > 0 and efi_sl <= 0
    efi_div_dn = px_sl < 0 and efi_sl >= 0
    efi_exp_up = efi_sl > 0 and (float(efi.iloc[-1]) if len(efi) else 0) > 0
    efi_exp_dn = efi_sl < 0 and (float(efi.iloc[-1]) if len(efi) else 0) < 0
    persist_h = _persist_side(px, float(vah) if vah else last + 1e9, "above", 3) if vah else False
    persist_l = _persist_side(px, float(val) if val else last - 1e9, "below", 3) if val else False
    tight = _box_tight(d, 8, 0.90)
    last_dv = float(delta_s.iloc[-1]) if len(delta_s) else 0.0
    inside_now = loc == "INSIDE"

    code = "CHOP"
    if reg["regime"] == "RANGE":
        if loc in ("ABOVE_VAH", "VAH_EDGE") and loc_z >= 0.35:
            if absorb_up and efi_div_up:
                code = "M1S_ENTRY"
            elif absorb_up or efi_div_up:
                code = "M1S_WATCH"
            else:
                code = "M1S_WATCH"
        elif loc in ("BELOW_VAL", "VAL_EDGE") and loc_z >= 0.35:
            if absorb_dn and efi_div_dn:
                code = "M1L_ENTRY"
            else:
                code = "M1L_WATCH"
        elif inside_now and vah and last < float(vah) and last_dv < 0 and px.iloc[-2] >= float(vah) * 0.999:
            code = "M1S_ADD"
        elif inside_now and val and last > float(val) and last_dv > 0 and px.iloc[-2] <= float(val) * 1.001:
            code = "M1L_ADD"
        else:
            code = "INSIDE"
    elif reg["regime"] == "TREND":
        if loc == "ABOVE_VAH" and persist_h and loc_z >= 0.45:
            if efi_exp_up and dv_sum > 0 and (tight or persist_h):
                code = "M2L_ENTRY"
            else:
                code = "M2L_WATCH"
        elif loc == "BELOW_VAL" and persist_l and loc_z >= 0.45:
            if efi_exp_dn and dv_sum < 0 and (tight or persist_l):
                code = "M2S_ENTRY"
            else:
                code = "M2S_WATCH"
        elif vah and last >= float(vah) and last_dv >= 0 and tight:
            code = "M2L_ADD"
        elif val and last <= float(val) and last_dv <= 0 and tight:
            code = "M2S_ADD"
        elif loc == "ABOVE_VAH":
            code = "M2L_WATCH"
        elif loc == "BELOW_VAL":
            code = "M2S_WATCH"
        else:
            code = "CHOP"
    else:
        if loc == "ABOVE_VAH" and loc_z >= 0.80 and absorb_up:
            code = "M1S_WATCH"
        elif loc == "BELOW_VAL" and loc_z >= 0.80 and absorb_dn:
            code = "M1L_WATCH"
        else:
            code = "CHOP"

    micro, action = VA_PLAYBOOK.get(code, VA_PLAYBOOK["CHOP"])
    glyphs = " ".join(LEVEL_GLYPH.get(k, "→") for k in (p_arr, d_arr, e_arr, c_arr))
    note = (
        f"{reg['note']} · {loc} z={loc_z:+.2f} · ΔΣ5={dv_sum:.0f} · "
        f"EFIsl={efi_sl:+.2f} p={efi_p:.2f} · Pxsl={px_sl:+.4f}"
    )
    return {
        "ok": True, "key": (p_arr, d_arr, e_arr, c_arr), "arrows": glyphs,
        "micro": micro + " · " + note, "action": action,
        "price": p_arr, "delta": d_arr, "efi": e_arr, "cvd": c_arr, "obv": d_arr,
        "hover": note, "efi_zero": False, "efi_note": note,
        "regime": reg["regime"], "model": code, "loc": loc, "loc_z": loc_z,
        "vah": vah, "val": val, "poc": poc,
    }


def pdec_session_history(dfi: pd.DataFrame, data: dict = None, min_bars: int = 16) -> list:
    out = []
    if dfi is None or dfi.empty:
        return out
    prev = None
    for i in range(len(dfi)):
        if i + 1 < min_bars:
            continue
        sl = dfi.iloc[: i + 1]
        try:
            rec = classify_va_setup(sl, data)
        except Exception:
            continue
        if not rec.get("ok"):
            continue
        act = rec.get("action") or ""
        if act == prev:
            continue
        prev = act
        t = sl["time_str"].iloc[-1] if "time_str" in sl.columns else str(i)
        out.append({"i": i, "t": str(t), "action": act, "model": rec.get("model")})
    return out
