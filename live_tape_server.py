#!/usr/bin/env python3
"""
Live Nifty tape sidecar — prices + VWAP + EFI + CVD + VA triggers only.

Run next to your Streamlit app (separate process):

    pip install smartapi-python pyotp websocket-client pandas numpy python-dotenv
    python live_tape_server.py

Open http://127.0.0.1:8765
Does not compute GEX / VEX / IV / heatmap.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import urllib.request
from collections import deque
from datetime import date
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from va_core import classify_va_setup, pdec_session_history, compute_session_volume_profile

load_dotenv()

# Laptop: 127.0.0.1:8765. Render: HOST=0.0.0.0 and PORT is set by Render.
HOST = os.getenv("TAPE_HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
PORT = int(os.getenv("PORT") or os.getenv("TAPE_PORT", "8765"))
BAR_SECONDS = int(os.getenv("TAPE_BAR_SECONDS", "180"))  # 3-min bars by default
NIFTY_TOKEN = os.getenv("NIFTY_TOKEN", "99926000")
FUT_TOKEN = os.getenv("FUT_TOKEN", "")
FUT_EXCHANGE_TYPE = int(os.getenv("FUT_EXCHANGE_TYPE", "2"))

# Same universe as Options Simulator INDEX_TOKEN_MAP
# ws_ex: Angel WS exchangeType 1 NSE, 2 NFO, 3 BSE, 4 BFO, 5 MCX
INDEX_MAP = {
    "NIFTY": {"spot": "99926000", "spot_ex": "NSE", "spot_ws": 1, "fut_ex": "NFO", "fut_ws": 2, "fut_type": "FUTIDX", "opt_ex": "NFO", "lot": 65},
    "BANKNIFTY": {"spot": "99926009", "spot_ex": "NSE", "spot_ws": 1, "fut_ex": "NFO", "fut_ws": 2, "fut_type": "FUTIDX", "opt_ex": "NFO", "lot": 30},
    "FINNIFTY": {"spot": "99926037", "spot_ex": "NSE", "spot_ws": 1, "fut_ex": "NFO", "fut_ws": 2, "fut_type": "FUTIDX", "opt_ex": "NFO", "lot": 60},
    "MIDCPNIFTY": {"spot": "99926074", "spot_ex": "NSE", "spot_ws": 1, "fut_ex": "NFO", "fut_ws": 2, "fut_type": "FUTIDX", "opt_ex": "NFO", "lot": 120},
    "SENSEX": {"spot": "99919000", "spot_ex": "BSE", "spot_ws": 3, "fut_ex": "BFO", "fut_ws": 4, "fut_type": "FUTIDX", "opt_ex": "BFO", "lot": 20},
    "GOLDM": {"spot": "", "spot_ex": "MCX", "spot_ws": 5, "fut_ex": "MCX", "fut_ws": 5, "fut_type": "FUTCOM", "opt_ex": "", "lot": 100},
    "CRUDEOIL": {"spot": "", "spot_ex": "MCX", "spot_ws": 5, "fut_ex": "MCX", "fut_ws": 5, "fut_type": "FUTCOM", "opt_ex": "", "lot": 100},
}
ACTIVE = {"name": os.getenv("TAPE_INDEX", "NIFTY")}
WS_HOLD = {"sws": None, "api": None, "spot": "", "fut": "", "cfg": INDEX_MAP["NIFTY"]}
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
GEX_SECONDS = int(os.getenv("TAPE_GEX_SECONDS", "3600"))  # hourly; ticks never run this
LOT = 65
GEX_WAKE = threading.Event()
API_HOLD = {"api": None}
INDEX_PDHL = {"pdh": None, "pdl": None}


def resolve_fut_token(name: str, cfg: dict) -> str:
    if FUT_TOKEN and name == "NIFTY":
        return str(FUT_TOKEN)
    try:
        with urllib.request.urlopen(SCRIP_MASTER_URL, timeout=30) as resp:
            rows = json.loads(resp.read().decode())
        today = date.today()
        best = None
        want_ex = cfg["fut_ex"]
        want_ty = cfg["fut_type"]
        for r in rows:
            if str(r.get("name", "")).upper() != name:
                continue
            if str(r.get("exch_seg", "")).upper() != want_ex:
                continue
            if str(r.get("instrumenttype", "")).upper() != want_ty:
                continue
            exp = pd.to_datetime(r.get("expiry"), format="%d%b%Y", errors="coerce")
            if pd.isna(exp) or exp.date() < today:
                continue
            tok = str(r.get("token") or "")
            if not tok:
                continue
            if best is None or exp.date() < best[0]:
                best = (exp.date(), tok, r.get("symbol"))
        if best:
            print(f"auto futures {name} {best[2]} expiry {best[0]} token {best[1]}")
            return best[1]
    except Exception as e:
        print("fut token lookup failed", e)
    return ""

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "live_tape.html"

# ---------- shared snapshot ----------
LOCK = threading.Lock()
SNAPSHOT = {
    "spot": None,
    "fut": None,
    "vwap": None,
    "efi": None,
    "cvd": None,
    "bar_tf": f"{BAR_SECONDS // 60} min",
    "bars": [],
    "va": {"action": "NO ENTRY", "micro": "starting"},
    "levels": {},
    "gex": {"ts": None, "net": None, "busy": False},
}
CLIENTS: set[asyncio.Queue] = set()
LOOP: asyncio.AbstractEventLoop | None = None


def _publish():
    if LOOP is None:
        return
    payload = json.dumps(SNAPSHOT, default=str)
    for q in list(CLIENTS):
        try:
            LOOP.call_soon_threadsafe(q.put_nowait, payload)
        except Exception:
            pass


# ---------- incremental session bars ----------
class SessionBars:
    def __init__(self, bar_seconds: int):
        self.bar_seconds = bar_seconds
        self.cur = None
        self.rows = []
        self.pv = 0.0
        self.vol = 0.0
        self.cvd = 0.0
        self.prev_close = None

    def _floor(self, ts: float) -> int:
        return int(ts) - (int(ts) % self.bar_seconds)

    def on_tick(self, ts: float, price: float, volume: float = 0.0, spot: float | None = None):
        if price <= 0:
            return
        t0 = self._floor(ts)
        signed = 0.0
        if self.prev_close is not None:
            signed = volume * (1.0 if price >= self.prev_close else -1.0)
        if self.cur is None or t0 != self.cur["t0"]:
            if self.cur is not None:
                self.rows.append(dict(self.cur))
                if len(self.rows) > 240:
                    self.rows = self.rows[-240:]
            self.cur = {
                "t0": t0,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "spot": spot if spot else price,
                "volume": 0.0,
                "delta": 0.0,
            }
        self.cur["high"] = max(self.cur["high"], price)
        self.cur["low"] = min(self.cur["low"], price)
        self.cur["close"] = price
        if spot:
            self.cur["spot"] = spot
        self.cur["volume"] += max(volume, 0.0)
        self.cur["delta"] += signed
        self.pv += price * max(volume, 0.0)
        self.vol += max(volume, 0.0)
        self.cvd += signed
        self.prev_close = price

    def seed_bar(self, t0: float, o: float, h: float, l: float, c: float, volume: float, spot: float | None = None):
        signed = 0.0
        if self.prev_close is not None:
            signed = volume * (1.0 if c >= self.prev_close else -1.0)
        self.rows.append({
            "t0": int(t0),
            "open": o, "high": h, "low": l, "close": c,
            "spot": spot if spot else c,
            "volume": max(volume, 0.0),
            "delta": signed,
        })
        self.pv += ((h + l + c) / 3.0) * max(volume, 0.0)
        self.vol += max(volume, 0.0)
        self.cvd += signed
        self.prev_close = c
        if len(self.rows) > 240:
            self.rows = self.rows[-240:]

    def vwap(self) -> float | None:
        if self.vol <= 0:
            return self.cur["close"] if self.cur else None
        return self.pv / self.vol

    def as_frame(self) -> pd.DataFrame:
        rows = list(self.rows)
        if self.cur:
            rows.append(dict(self.cur))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["vwap"] = np.nan
        # reconstruct running vwap from stored bars is approximate; use live vwap on last
        df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_pv = (df["typical"] * df["volume"]).cumsum()
        cum_v = df["volume"].cumsum().replace(0, np.nan)
        df["vwap"] = cum_pv / cum_v
        df["vwap"] = df["vwap"].ffill()
        raw_efi = df["close"].diff().fillna(0.0) * df["volume"]
        df["efi13"] = raw_efi.ewm(span=13, adjust=False).mean()
        df["cvd"] = df["delta"].cumsum()
        df["spot_px"] = df["spot"]
        return df


BARS = SessionBars(BAR_SECONDS)


# ---------- VA (same models as the Streamlit playbook, no GEX) ----------
VA_PLAYBOOK = {
    "M1S_WATCH": ("Range · VAH probe", "WATCH SHORT (VAH fade)"),
    "M1S_ENTRY": ("Range · VAH mean-reversion SHORT", "SHORT MEAN-REVERSION (Target: POC → VAL)"),
    "M1S_ADD": ("Range · VAH failed acceptance", "ADD SHORT (Retest VAH from below)"),
    "M1L_WATCH": ("Range · VAL probe", "WATCH LONG (VAL bounce)"),
    "M1L_ENTRY": ("Range · VAL mean-reversion LONG", "LONG MEAN-REVERSION (Target: POC → VAH)"),
    "M1L_ADD": ("Range · VAL failed breakdown", "ADD LONG (Retest VAL from above)"),
    "M2L_WATCH": ("Trend · VAH break", "WATCH LONG (VAH acceptance)"),
    "M2L_ENTRY": ("Trend · VAH acceptance LONG", "LONG BREAKOUT (Hold above VAH / trail under box)"),
    "M2L_ADD": ("Trend · VAH retest from above", "ADD LONG (Retest old VAH / new box low)"),
    "M2S_WATCH": ("Trend · VAL break", "WATCH SHORT (VAL acceptance)"),
    "M2S_ENTRY": ("Trend · VAL acceptance SHORT", "SHORT BREAKDOWN (Hold below VAL / trail above box)"),
    "M2S_ADD": ("Trend · VAL retest from below", "ADD SHORT (Retest old VAL / new box high)"),
    "INSIDE": ("Inside value", "NO ENTRY"),
    "CHOP": ("Regime mixed / low efficiency", "NO ENTRY"),
}


def _value_area(df: pd.DataFrame) -> dict:
    if df.empty or df["volume"].sum() <= 0:
        return {}
    px = df["close"].astype(float)
    step = 5.0
    lo, hi = float(px.min()), float(px.max())
    bins = np.arange(lo, hi + step, step)
    if len(bins) < 3:
        poc = float(px.iloc[-1])
        return {"poc": poc, "vah": poc, "val": poc}
    # Prefer high-low volume distribution (Streamlit profile). Fallback: close bins.
    use_hl = "high" in df.columns and "low" in df.columns
    if use_hl:
        lo = float(min(df["low"].min(), df["close"].min()))
        hi = float(max(df["high"].max(), df["close"].max()))
        if hi <= lo:
            hi = lo + step
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step
        edges = np.arange(lo, hi + step * 0.5, step)
        if len(edges) < 3:
            last = float(df["close"].iloc[-1])
            return {"poc": last, "vah": last, "val": last}
        n_bins = len(edges) - 1
        vol = np.zeros(n_bins)
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
                    vol[i] += v * (overlap / span)
        if vol.sum() <= 0:
            last = float(df["close"].iloc[-1])
            return {"poc": last, "vah": last, "val": last}
        mid = (edges[:-1] + edges[1:]) / 2.0
        poc_i = int(vol.argmax())
        target = float(vol.sum()) * 0.70
        a = b = poc_i
        taken = float(vol[poc_i])
        while taken < target and (a > 0 or b < n_bins - 1):
            left = float(vol[a - 1]) if a > 0 else -1
            right = float(vol[b + 1]) if b < n_bins - 1 else -1
            if right >= left:
                b = min(b + 1, n_bins - 1)
                taken += float(vol[b])
            else:
                a = max(a - 1, 0)
                taken += float(vol[a])
        return {"poc": float(mid[poc_i]), "val": float(mid[a]), "vah": float(mid[b])}
    idx = np.clip(np.digitize(px, bins) - 1, 0, len(bins) - 2)
    vol = np.zeros(len(bins) - 1)
    for i, v in zip(idx, df["volume"].astype(float)):
        vol[i] += v
    poc_i = int(vol.argmax())
    target = vol.sum() * 0.70
    taken = vol[poc_i]
    a, b = poc_i, poc_i
    while taken < target and (a > 0 or b < len(vol) - 1):
        left = vol[a - 1] if a > 0 else -1
        right = vol[b + 1] if b < len(vol) - 1 else -1
        if right >= left:
            b = min(b + 1, len(vol) - 1)
            taken += max(right, 0)
        else:
            a = max(a - 1, 0)
            taken += max(left, 0)
    mid = (bins[:-1] + bins[1:]) / 2.0
    return {"poc": float(mid[poc_i]), "val": float(bins[a]), "vah": float(bins[b + 1])}


def _profile_bins(df: pd.DataFrame, step: float = 2.0) -> list:
    if df is None or df.empty:
        return []
    lo = float(min(df["low"].min() if "low" in df.columns else df["close"].min(), df["close"].min()))
    hi = float(max(df["high"].max() if "high" in df.columns else df["close"].max(), df["close"].max()))
    if hi <= lo:
        hi = lo + step
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    edges = np.arange(lo, hi + step * 0.5, step)
    n_bins = max(len(edges) - 1, 1)
    vol = np.zeros(n_bins)
    delta = np.zeros(n_bins)
    for _, r in df.iterrows():
        v = float(r.get("volume") or 0)
        dlt = float(r.get("delta") or 0)
        l = float(r["low"] if "low" in r else r["close"])
        h = float(r["high"] if "high" in r else r["close"])
        if h <= l:
            h = l + 1e-6
        span = h - l
        for i in range(n_bins):
            a, b = float(edges[i]), float(edges[i + 1])
            overlap = max(0.0, min(h, b) - max(l, a))
            if overlap > 0:
                frac = overlap / span
                vol[i] += v * frac
                delta[i] += dlt * frac
    mids = (edges[:-1] + edges[1:]) / 2.0
    return [{"px": float(mids[i]), "vol": float(vol[i]), "delta": float(delta[i])} for i in range(n_bins)]


def _eff(df: pd.DataFrame, n=20) -> float:
    c = df["close"].astype(float).tail(n)
    if len(c) < 8:
        return 0.0
    net = abs(float(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    return net / max(path, 1e-9)


def _ols_slope(y):
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
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
    se = float(np.sqrt((resid * resid).sum() / max(n - 2, 1) / den))
    if se <= 1e-18:
        return sl, 0.0
    t = sl / se
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))
    return sl, p


def classify_va(df: pd.DataFrame) -> dict:
    """Same M1/M2 playbook as Options Simulator (no GEX term)."""
    empty = {"action": "NO ENTRY", "micro": "not enough bars", "model": "CHOP"}
    if df is None or len(df) < 12:
        return empty
    px = df["close"].astype(float)
    last = float(px.iloc[-1])
    vp = _value_area(df)
    vah, val, poc = vp.get("vah"), vp.get("val"), vp.get("poc")
    eff = _eff(df)
    dens = 0.0
    if vah and val and float(df["volume"].sum()) > 0:
        m = (df["close"] >= val) & (df["close"] <= vah)
        dens = float(df.loc[m, "volume"].sum() / float(df["volume"].sum()))
    std = float(px.tail(20).std(ddof=1) or 1.0)
    h = df["high"].astype(float) if "high" in df.columns else px
    l = df["low"].astype(float) if "low" in df.columns else px
    prev = px.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr_abs = float(tr.tail(14).mean() or 0)
    sess_rng = float(h.max() - l.min()) if len(h) else 1.0
    realized = sess_rng / max(atr_abs * math.sqrt(max(len(df), 1) / 14.0), 1e-6)
    s_range = 0.0
    s_trend = 0.0
    s_range += 1.2 if dens >= 0.62 else (0.4 if dens >= 0.50 else -0.4)
    s_range += 1.0 if eff < 0.22 else (0.2 if eff < 0.32 else -0.6)
    s_range += 0.4 if realized < 1.15 else -0.3
    s_trend += 1.2 if eff >= 0.38 else (0.4 if eff >= 0.28 else -0.5)
    s_trend += 1.0 if dens <= 0.48 else (0.2 if dens <= 0.58 else -0.6)
    s_trend += 0.5 if realized >= 1.25 else -0.2
    if s_range >= 1.4 and s_range >= s_trend + 0.35:
        regime = "RANGE"
    elif s_trend >= 1.4 and s_trend >= s_range + 0.35:
        regime = "TREND"
    else:
        regime = "CHOP"
    buf = max(0.35 * std, 0.35 * atr_abs, 1.0)
    loc, loc_z = "INSIDE", 0.0
    if vah is not None and val is not None:
        z_h = (last - float(vah)) / max(std, 1e-6)
        z_l = (float(val) - last) / max(std, 1e-6)
        if last >= float(vah) + buf:
            loc, loc_z = "ABOVE_VAH", z_h
        elif last <= float(val) - buf:
            loc, loc_z = "BELOW_VAL", z_l
        elif last > float(vah):
            loc, loc_z = "VAH_EDGE", z_h
        elif last < float(val):
            loc, loc_z = "VAL_EDGE", z_l
    efi = df["efi13"] if "efi13" in df.columns else pd.Series(dtype=float)
    delta_s = df["delta"] if "delta" in df.columns else px.diff().fillna(0.0)
    efi_sl, efi_p = _ols_slope(efi.tail(12)) if len(efi) else (0.0, 1.0)
    px_sl, px_p = _ols_slope(px.tail(12))
    dv_tail = delta_s.tail(5)
    dv_sum = float(dv_tail.sum()) if len(dv_tail) else 0.0
    dv_abs = float(dv_tail.abs().sum()) if len(dv_tail) else 1.0
    absorb_up = (px_sl > 0 and px_p < 0.12) and (dv_sum <= 0.15 * max(dv_abs, 1.0))
    absorb_dn = (px_sl < 0 and px_p < 0.12) and (dv_sum >= -0.15 * max(dv_abs, 1.0))
    efi_div_up = px_sl > 0 and efi_sl <= 0
    efi_div_dn = px_sl < 0 and efi_sl >= 0
    efi_exp_up = efi_sl > 0 and (float(efi.iloc[-1]) if len(efi) else 0) > 0
    efi_exp_dn = efi_sl < 0 and (float(efi.iloc[-1]) if len(efi) else 0) < 0
    persist_h = bool(vah and (px.tail(3) > float(vah)).sum() >= 2)
    persist_l = bool(val and (px.tail(3) < float(val)).sum() >= 2)
    last_dv = float(delta_s.iloc[-1]) if len(delta_s) else 0.0
    tight = False
    if len(df) >= 8:
        sl = df.tail(8)
        hh = float(sl["high"].max()) if "high" in sl.columns else float(sl["close"].max())
        ll = float(sl["low"].min()) if "low" in sl.columns else float(sl["close"].min())
        tight = (hh - ll) <= max(0.90 * atr_abs, 1.0)
    code = "CHOP"
    if regime == "RANGE":
        if loc in ("ABOVE_VAH", "VAH_EDGE") and loc_z >= 0.35:
            code = "M1S_ENTRY" if (absorb_up and efi_div_up) else "M1S_WATCH"
        elif loc in ("BELOW_VAL", "VAL_EDGE") and loc_z >= 0.35:
            code = "M1L_ENTRY" if (absorb_dn and efi_div_dn) else "M1L_WATCH"
        elif loc == "INSIDE" and vah and last < float(vah) and last_dv < 0 and float(df["high"].iloc[-2] if "high" in df.columns else px.iloc[-2]) >= float(vah) * 0.9995:
            code = "M1S_ADD"
        elif loc == "INSIDE" and val and last > float(val) and last_dv > 0 and float(df["low"].iloc[-2] if "low" in df.columns else px.iloc[-2]) <= float(val) * 1.0005:
            code = "M1L_ADD"
        else:
            code = "INSIDE"
    elif regime == "TREND":
        if loc == "ABOVE_VAH" and persist_h and loc_z >= 0.45:
            code = "M2L_ENTRY" if (efi_exp_up and dv_sum > 0 and (tight or persist_h)) else "M2L_WATCH"
        elif loc == "BELOW_VAL" and persist_l and loc_z >= 0.45:
            code = "M2S_ENTRY" if (efi_exp_dn and dv_sum < 0 and (tight or persist_l)) else "M2S_WATCH"
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
    note = (
        f"{regime} {loc} z={loc_z:+.2f} dens={dens:.2f} eff={eff:.2f} "
        f"ΔΣ5={dv_sum:.0f} EFIsl={efi_sl:+.2f} VAH {vah} VAL {val} POC {poc}"
    )
    return {
        "action": action, "micro": f"{micro} · {note}", "model": code, "regime": regime,
        "vah": vah, "val": val, "poc": poc, "loc": loc,
    }


def _va_labels(df: pd.DataFrame) -> list:
    """Every action change, same idea as pdec_session_history in the Streamlit file."""
    out = []
    if df is None or len(df) < 12:
        return out
    prev = None
    for i in range(11, len(df)):
        rec = classify_va(df.iloc[: i + 1])
        act = rec.get("action") or ""
        if act == prev:
            continue
        prev = act
        t0 = df.iloc[i].get("t0")
        try:
            tstr = pd.to_datetime(t0, unit="s", utc=True).tz_convert("Asia/Kolkata").strftime("%H:%M")
        except Exception:
            tstr = ""
        out.append({"i": int(i), "t": tstr, "action": act, "model": rec.get("model")})
    return out[-24:]


def apply_bar_state(spot, fut):
    df = BARS.as_frame()
    vwap = BARS.vwap()
    last = df.iloc[-1] if not df.empty else None
    va = classify_va(df) if not df.empty else {"action": "NO ENTRY", "micro": "warming up"}
    bars_out = []
    pdh = pdl = None
    if not df.empty:
        tail = df.tail(160).copy()
        if "t0" in tail.columns:
            days = pd.to_datetime(tail["t0"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
            last_day = days.dt.date.dropna()
            if len(last_day):
                sess = tail[days.dt.date == last_day.iloc[-1]].copy()
                if len(sess) >= 12:
                    tail_for_va = sess
                else:
                    tail_for_va = tail
            else:
                tail_for_va = tail
        else:
            tail_for_va = tail
        sess_df = tail_for_va.reset_index(drop=True)
        if "vwap" in sess_df.columns and "vwap_idx" not in sess_df.columns:
            sess_df["vwap_idx"] = sess_df["vwap"]
        if "time_str" not in sess_df.columns:
            try:
                sess_df["time_str"] = pd.to_datetime(sess_df["t0"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%H:%M")
            except Exception:
                sess_df["time_str"] = ""
        gex_data = {"total_net_gex_oi": (SNAPSHOT.get("levels") or {}).get("gex", 0)}
        rec = classify_va_setup(sess_df, gex_data)
        va = {
            "action": rec.get("action") or "NO ENTRY",
            "micro": rec.get("micro") or "",
            "model": rec.get("model"),
            "regime": rec.get("regime"),
            "vah": rec.get("vah"),
            "val": rec.get("val"),
            "poc": rec.get("poc"),
            "loc": rec.get("loc"),
        }
        try:
            vp = compute_session_volume_profile(sess_df, bin_step=2.0)
            if vp.get("ok"):
                for k in ("vah1", "val1", "vah15", "val15", "vah80", "val80", "zones"):
                    if k in vp:
                        va[k] = vp[k]
                va["vah"] = vp.get("vah", va.get("vah"))
                va["val"] = vp.get("val", va.get("val"))
                va["poc"] = vp.get("poc", va.get("poc"))
        except Exception:
            pass
        va["labels"] = pdec_session_history(sess_df, gex_data, min_bars=16)
        # previous-day high/low from seeded t0 if present
        if "t0" in tail.columns:
            days = pd.to_datetime(tail["t0"], unit="s", errors="coerce")
            tail["_d"] = days.dt.date
            uniq = [d for d in tail["_d"].dropna().unique()]
            if len(uniq) >= 2:
                prevd = tail[tail["_d"] == uniq[-2]]
                if not prevd.empty:
                    pdh = float(prevd["high"].max())
                    pdl = float(prevd["low"].min())
        for _, r in tail_for_va.iterrows():
            t0 = r.get("t0")
            try:
                tstr = (
                    pd.to_datetime(t0, unit="s", utc=True)
                    .tz_convert("Asia/Kolkata")
                    .strftime("%H:%M")
                )
            except Exception:
                tstr = ""
            bars_out.append({
                "t": tstr,
                "open": float(r["open"]) if "open" in r and pd.notna(r["open"]) else float(r["close"]),
                "high": float(r["high"]) if "high" in r and pd.notna(r["high"]) else float(r["close"]),
                "low": float(r["low"]) if "low" in r and pd.notna(r["low"]) else float(r["close"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]) if pd.notna(r.get("volume")) else 0.0,
                "delta": float(r["delta"]) if pd.notna(r.get("delta")) else 0.0,
                "vwap": float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
                "efi13": float(r["efi13"]) if pd.notna(r.get("efi13")) else 0.0,
                "cvd": float(r["cvd"]) if pd.notna(r.get("cvd")) else 0.0,
            })
        va["labels"] = _va_labels(tail_for_va.reset_index(drop=True))
        va["pdh"] = INDEX_PDHL.get("pdh") or pdh
        va["pdl"] = INDEX_PDHL.get("pdl") or pdl
        prof = _profile_bins(tail_for_va, 2.0)
        va["profile"] = prof
        if prof:
            prices = [p["px"] for p in prof]
            vols = [float(p.get("vol") or 0) for p in prof]
            poc_i = int(max(range(len(vols)), key=lambda i: vols[i]))
            va["poc"] = float(prices[poc_i])
            peak = vols[poc_i] or 1.0
            # HVNs = local volume peaks (the fat pockets you marked)
            hvns = []
            for i in range(1, len(vols) - 1):
                if vols[i] >= vols[i - 1] and vols[i] >= vols[i + 1] and vols[i] >= 0.32 * peak:
                    if any(abs(prices[i] - h) < 8 for h in hvns):
                        continue
                    hvns.append(float(prices[i]))
            hvns.sort()
            va["hvns"] = hvns
            # Tight VA around POC: stop at first valley under 28% of peak
            a = b = poc_i
            while a > 0 and vols[a - 1] >= 0.28 * peak:
                a -= 1
            while b < len(vols) - 1 and vols[b + 1] >= 0.28 * peak:
                b += 1
            va["val"] = float(prices[a])
            va["vah"] = float(prices[b])
        for k in ("vah1", "val1", "vah15", "val15", "vah80", "val80", "zones"):
            va.pop(k, None)
    with LOCK:
        SNAPSHOT.update({
            "spot": spot,
            "fut": fut,
            "vwap": vwap,
            "efi": float(last["efi13"]) if last is not None else None,
            "cvd": float(last["cvd"]) if last is not None else None,
            "bars": bars_out,
            "va": va,
            "levels": dict(SNAPSHOT.get("levels") or {}),
        })
    _publish()


def fetch_structure_levels(api, spot: float) -> dict:
    """Put wall / call wall / flip from nearest NIFTY expiry OI (no full GEX)."""
    out = {}
    if not api or not spot:
        return out
    try:
        with urllib.request.urlopen(SCRIP_MASTER_URL, timeout=30) as resp:
            rows = json.loads(resp.read().decode())
        today = date.today()
        opts = []
        for r in rows:
            if str(r.get("name", "")).upper() != ACTIVE.get("name", "NIFTY"):
                continue
            if str(r.get("exch_seg", "")).upper() != "NFO":
                continue
            if str(r.get("instrumenttype", "")).upper() != "OPTIDX":
                continue
            exp = pd.to_datetime(r.get("expiry"), format="%d%b%Y", errors="coerce")
            if pd.isna(exp) or exp.date() < today:
                continue
            strike = float(r.get("strike") or 0) / 100.0
            if strike > 1e6:
                strike /= 100.0
            if abs(strike - spot) > 600:
                continue
            opts.append((exp.date(), strike, str(r.get("symbol", "")), str(r.get("token", ""))))
        if not opts:
            return out
        near_exp = min(o[0] for o in opts)
        opts = [o for o in opts if o[0] == near_exp]
        tokens = [o[3] for o in opts if o[3]]
        market = {}
        for i in range(0, len(tokens), 40):
            chunk = tokens[i:i + 40]
            res = api.getMarketData("FULL", {"NFO": chunk})
            fetched = ((res or {}).get("data") or {}).get("fetched") or []
            for item in fetched:
                market[str(item.get("symbolToken"))] = item
            time.sleep(0.35)
        rows_oi = []
        for exp, strike, sym, tok in opts:
            rec = market.get(tok) or {}
            oi = float(rec.get("opnInterest") or rec.get("oi") or 0)
            typ = "CE" if str(sym).endswith("CE") else "PE"
            rows_oi.append({"strike": strike, "type": typ, "oi": oi})
        if not rows_oi:
            return out
        dfo = pd.DataFrame(rows_oi)
        ce = dfo[dfo["type"] == "CE"]
        pe = dfo[dfo["type"] == "PE"]
        above = ce[ce["strike"] >= spot]
        below = pe[pe["strike"] <= spot]
        if not above.empty:
            out["call_wall"] = float(above.sort_values("oi", ascending=False).iloc[0]["strike"])
        if not below.empty:
            out["put_wall"] = float(below.sort_values("oi", ascending=False).iloc[0]["strike"])
        # crude flip: strike nearest where CE OI and PE OI swap dominance
        both = []
        for k in sorted(set(dfo["strike"])):
            c = float(ce.loc[ce["strike"] == k, "oi"].sum()) if not ce.empty else 0
            p = float(pe.loc[pe["strike"] == k, "oi"].sum()) if not pe.empty else 0
            both.append((k, p - c))
        flip = None
        for i in range(1, len(both)):
            if both[i - 1][1] == 0:
                continue
            if both[i - 1][1] * both[i][1] <= 0:
                flip = both[i][0] if abs(both[i][1]) < abs(both[i - 1][1]) else both[i - 1][0]
                break
        if flip:
            out["flip"] = float(flip)
        print("levels", out)
    except Exception as e:
        print("structure levels failed", e)
    return out


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def compute_hourly_greeks(api, spot: float) -> dict:
    """GEX / VEX snapshot. Called hourly or on Hard refresh — never from ticks."""
    pack = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "net": None,
        "vex": None,
        "put_wall": None,
        "call_wall": None,
        "flip": None,
        "atm_iv": 0.15,
        "strikes": [],
        "busy": False,
        "gex": 0.0,
    }
    if not api or not spot:
        return pack
    lv = fetch_structure_levels(api, spot)
    pack.update({k: lv[k] for k in lv})
    try:
        with urllib.request.urlopen(SCRIP_MASTER_URL, timeout=30) as resp:
            rows = json.loads(resp.read().decode())
        today = date.today()
        opts = []
        for r in rows:
            if str(r.get("name", "")).upper() != ACTIVE.get("name", "NIFTY"):
                continue
            if str(r.get("exch_seg", "")).upper() != "NFO":
                continue
            if str(r.get("instrumenttype", "")).upper() != "OPTIDX":
                continue
            exp = pd.to_datetime(r.get("expiry"), format="%d%b%Y", errors="coerce")
            if pd.isna(exp) or exp.date() < today:
                continue
            strike = float(r.get("strike") or 0) / 100.0
            if strike > 1e6:
                strike /= 100.0
            if abs(strike - spot) > 500:
                continue
            opts.append((exp.date(), strike, str(r.get("symbol", "")), str(r.get("token", ""))))
        if not opts:
            return pack
        near = min(o[0] for o in opts)
        opts = [o for o in opts if o[0] == near]
        T = max((near - today).days, 1) / 365.0
        tokens = [o[3] for o in opts if o[3]]
        market = {}
        for i in range(0, len(tokens), 40):
            res = api.getMarketData("FULL", {"NFO": tokens[i:i + 40]})
            fetched = ((res or {}).get("data") or {}).get("fetched") or []
            for item in fetched:
                market[str(item.get("symbolToken"))] = item
            time.sleep(0.35)
        # ATM IV from straddle / spot
        atm = min({o[1] for o in opts}, key=lambda k: abs(k - spot))
        ce_atm = pe_atm = None
        by = {}
        for exp, strike, sym, tok in opts:
            rec = market.get(tok) or {}
            ltp = float(rec.get("ltp") or 0)
            oi = float(rec.get("opnInterest") or rec.get("oi") or 0)
            typ = "CE" if str(sym).endswith("CE") else "PE"
            by.setdefault(strike, {})[typ] = {"ltp": ltp, "oi": oi}
            if strike == atm and typ == "CE":
                ce_atm = ltp
            if strike == atm and typ == "PE":
                pe_atm = ltp
        sig = 0.15
        if ce_atm and pe_atm and spot > 0 and T > 0:
            # crude IV: straddle / (spot * sqrt(T) * 0.8)
            sig = max(0.05, min(0.80, (ce_atm + pe_atm) / (spot * math.sqrt(T) * 0.8)))
        pack["atm_iv"] = sig
        net_gex = 0.0
        net_vex = 0.0
        gex_curve = []
        lot = int((WS_HOLD.get("cfg") or {}).get("lot") or LOT)
        gex_scale = lot * (spot ** 2) * 0.01
        for k, sides in sorted(by.items()):
            d1 = (math.log(max(spot, 1e-9) / max(k, 1e-9)) + 0.5 * sig * sig * T) / max(sig * math.sqrt(T), 1e-9)
            gam = _norm_pdf(d1) / max(spot * sig * math.sqrt(T), 1e-9)
            vega = spot * _norm_pdf(d1) * math.sqrt(max(T, 1e-9)) * 0.01
            ce_oi = float((sides.get("CE") or {}).get("oi") or 0)
            pe_oi = float((sides.get("PE") or {}).get("oi") or 0)
            if abs(k - spot) > 250:
                continue
            gex_k = gam * ce_oi * gex_scale - gam * pe_oi * gex_scale
            vex_k = (vega * ce_oi - vega * pe_oi) * lot * 0.01
            net_gex += gex_k
            net_vex += vex_k
            gex_curve.append({"k": k, "gex": gex_k, "vex": vex_k})
        pack["net"] = net_gex
        pack["gex"] = net_gex
        pack["vex"] = net_vex
        pack["strikes"] = gex_curve
        pack["expiry"] = str(near)
        # flip: gex sign change
        prev = None
        for row in gex_curve:
            if prev is not None and prev["gex"] * row["gex"] < 0:
                pack["flip"] = row["k"]
                break
            prev = row
        print("hourly greeks net_gex", round(net_gex, 2), "vex", round(net_vex, 2))
    except Exception as e:
        print("hourly greeks failed", e)
    return pack


# ---------- Angel WS thread ----------
def angel_thread():
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    import pyotp

    api_key = os.getenv("API_KEY", "")
    client = os.getenv("CLIENT_CODE", "")
    pin = os.getenv("PIN", "")
    totp_secret = os.getenv("TOTP_SECRET", "")
    if not all([api_key, client, pin, totp_secret]):
        print("Missing API_KEY / CLIENT_CODE / PIN / TOTP_SECRET in .env")
        return

    api = SmartConnect(api_key=api_key)
    sess = api.generateSession(client, pin, pyotp.TOTP(totp_secret).now())
    if not sess or not sess.get("status"):
        print("SmartAPI login failed", sess)
        return
    jwt = sess["data"]["jwtToken"]
    feed = api.getfeedToken()
    sws = SmartWebSocketV2(jwt, api_key, client, feed)

    name0 = ACTIVE["name"] if ACTIVE["name"] in INDEX_MAP else "NIFTY"
    cfg0 = INDEX_MAP[name0]
    fut_token = resolve_fut_token(name0, cfg0)
    spot_token = cfg0.get("spot") or ""
    last_spot = {"px": None}
    last_fut = {"px": None, "vol": 0.0}
    WS_HOLD.update({"sws": sws, "api": api, "spot": spot_token, "fut": fut_token, "cfg": cfg0})
    SNAPSHOT["index"] = name0
    SNAPSHOT["indexes"] = list(INDEX_MAP.keys())

    def seed_history():
        """Load last session 3-min futures candles so charts/VA are not empty."""
        fut_token = WS_HOLD.get("fut") or ""
        if not fut_token:
            print("no fut token — skip candle seed")
            return
        from datetime import datetime, timedelta
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=5)
        interval = "THREE_MINUTE" if BAR_SECONDS <= 180 else "FIVE_MINUTE"
        param = {
            "exchange": WS_HOLD["cfg"]["fut_ex"],
            "symboltoken": str(fut_token),
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        }
        try:
            res = api.getCandleData(param)
        except Exception as e:
            print("seed candles failed", e)
            return
        rows = (res or {}).get("data") or []
        print(f"seeded {len(rows)} {interval} candles")
        for row in rows:
            # [timestamp, open, high, low, close, volume]
            ts = pd.to_datetime(row[0])
            t0 = ts.timestamp()
            o, h, l, c = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            vol = float(row[5] or 0)
            BARS.seed_bar(t0, o, h, l, c, vol, spot=c)
        if BARS.rows:
            last_fut["px"] = float(BARS.rows[-1]["close"])
            apply_bar_state(last_spot["px"], last_fut["px"])

    seed_history()
    API_HOLD["api"] = api
    try:
        from datetime import datetime, timedelta
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=7)
        res = api.getCandleData({
            "exchange": WS_HOLD["cfg"]["spot_ex"] or WS_HOLD["cfg"]["fut_ex"],
            "symboltoken": str(WS_HOLD["spot"] or fut_token),
            "interval": "ONE_DAY",
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": to_dt.strftime("%Y-%m-%d 15:30"),
        })
        days = (res or {}).get("data") or []
        if len(days) >= 2:
            prev = days[-2]
            INDEX_PDHL["pdh"] = float(prev[2])
            INDEX_PDHL["pdl"] = float(prev[3])
            print("Nifty PDH/PDL", INDEX_PDHL)
            apply_bar_state(last_spot["px"], last_fut["px"] or last_spot["px"])
    except Exception as e:
        print("nifty pdh/pdl failed", e)

    def gex_loop():
        while True:
            try:
                with LOCK:
                    SNAPSHOT.setdefault("gex", {})["busy"] = True
                px = last_spot["px"] or last_fut["px"]
                pack = compute_hourly_greeks(api, float(px or 0))
                pack["busy"] = False
                with LOCK:
                    SNAPSHOT["gex"] = pack
                    lv = SNAPSHOT.get("levels") or {}
                    lv.update({
                        "put_wall": pack.get("put_wall"),
                        "call_wall": pack.get("call_wall"),
                        "flip": pack.get("flip"),
                        "gex": pack.get("net") or 0,
                    })
                    SNAPSHOT["levels"] = lv
                apply_bar_state(last_spot["px"], last_fut["px"] or last_spot["px"])
            except Exception as e:
                print("gex loop", e)
                with LOCK:
                    SNAPSHOT.setdefault("gex", {})["busy"] = False
            GEX_WAKE.clear()
            GEX_WAKE.wait(timeout=max(GEX_SECONDS, 60))

    threading.Thread(target=gex_loop, daemon=True).start()

    def on_data(_ws, msg):
        try:
            token = str(msg.get("token") or "")
            ltp = msg.get("last_traded_price")
            if ltp is None:
                return
            px = float(ltp)
            # Angel often sends paise * 100 for indices; Nifty LTP ~2500000 → divide
            if px > 100000:
                px = px / 100.0
            vol = float(msg.get("last_traded_quantity") or msg.get("volume_trade_for_the_day") or 0)
            ts = float(msg.get("exchange_timestamp") or time.time() * 1000) / 1000.0
            if WS_HOLD["spot"] and token == str(WS_HOLD["spot"]):
                last_spot["px"] = px
                if not WS_HOLD["fut"]:
                    BARS.on_tick(ts, px, 0.0, spot=px)
            elif WS_HOLD["fut"] and token == str(WS_HOLD["fut"]):
                last_fut["px"] = px
                day_v = float(msg.get("volume_trade_for_the_day") or 0)
                inc = 0.0
                if day_v and last_fut["vol"] and day_v >= last_fut["vol"]:
                    inc = day_v - last_fut["vol"]
                last_fut["vol"] = day_v or last_fut["vol"]
                BARS.on_tick(ts, px, inc or max(vol, 0.0), spot=last_spot["px"] or px)
            apply_bar_state(last_spot["px"], last_fut["px"] or last_spot["px"])
        except Exception as e:
            print("tick parse", e)

    def on_open(_ws):
        tokens = []
        cfg = WS_HOLD["cfg"]
        if WS_HOLD["spot"]:
            tokens.append({"exchangeType": cfg["spot_ws"], "tokens": [str(WS_HOLD["spot"])]})
        if WS_HOLD["fut"]:
            tokens.append({"exchangeType": cfg["fut_ws"], "tokens": [str(WS_HOLD["fut"])]})
        sws.subscribe("tape01", 3, tokens)
        print("subscribed", tokens)

    sws.on_data = on_data
    sws.on_open = on_open
    print("connecting Angel SmartWebSocketV2…")
    sws.connect()


# ---------- tiny HTTP + WebSocket (stdlib handshake) ----------
import base64
import hashlib
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def switch_index(name: str) -> bool:
    name = (name or "").upper().strip()
    if name not in INDEX_MAP:
        return False
    ACTIVE["name"] = name
    cfg = INDEX_MAP[name]
    fut = resolve_fut_token(name, cfg)
    WS_HOLD["cfg"] = cfg
    WS_HOLD["spot"] = cfg.get("spot") or ""
    WS_HOLD["fut"] = fut
    try:
        BARS.rows.clear()
    except Exception:
        pass
    INDEX_PDHL["pdh"] = INDEX_PDHL["pdl"] = None
    with LOCK:
        SNAPSHOT["index"] = name
        SNAPSHOT["bars"] = []
        SNAPSHOT["va"] = {"action": "NO ENTRY", "micro": f"switching {name}"}
    sws = WS_HOLD.get("sws")
    if sws:
        try:
            tokens = []
            if WS_HOLD["spot"]:
                tokens.append({"exchangeType": cfg["spot_ws"], "tokens": [str(WS_HOLD["spot"])]})
            if fut:
                tokens.append({"exchangeType": cfg["fut_ws"], "tokens": [str(fut)]})
            sws.subscribe("tape01", 3, tokens)
            print("resubscribed", tokens)
        except Exception as e:
            print("resub failed", e)
    GEX_WAKE.set()
    return True


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/live_tape.html"):
            data = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/ws":
            self._ws()
            return
        if self.path.startswith("/api/gex"):
            with LOCK:
                body = json.dumps(SNAPSHOT.get("gex") or {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/index"):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode() or "{}")
            except Exception:
                body = {}
            name = str(body.get("index") or "")
            ok = switch_index(name)
            msg = json.dumps({"ok": ok, "index": ACTIVE["name"], "indexes": list(INDEX_MAP)}).encode()
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        if self.path.startswith("/api/refresh-gex"):
            GEX_WAKE.set()
            msg = json.dumps({"ok": True, "msg": "GEX refresh queued"}).encode()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass

    def _ws(self):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400)
            return
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        q: asyncio.Queue = asyncio.Queue()
        CLIENTS.add(q)
        with LOCK:
            first = json.dumps(SNAPSHOT, default=str)
        try:
            self._send_text(first)
            while True:
                # block without asyncio: poll snapshot every 200ms if queue empty
                try:
                    # thread is not the event loop; just sleep-push latest
                    time.sleep(0.2)
                    with LOCK:
                        payload = json.dumps(SNAPSHOT, default=str)
                    self._send_text(payload)
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            CLIENTS.discard(q)

    def _send_text(self, text: str):
        raw = text.encode("utf-8")
        header = bytearray([0x81])
        n = len(raw)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", n))
        self.wfile.write(header + raw)


def main():
    t = threading.Thread(target=angel_thread, daemon=True)
    t.start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"tape UI  http://{HOST}:{PORT}")
    print("GEX/VEX/IV are not loaded on this page.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
