"""Gemini Scalper — live tapes, volume profile, manual setups, continuous position watch."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

from multi_index_scalper import (
    ORDER, LOT_SIZES, STEP, TF_API,
    rows_from_df, attach_efi, seed_book, apply_live_tape,
    session_open, _ist_now,
)

MODELS = (
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash",
    "gemini-2.5-flash", "gemini-2.0-flash",
)
TFS = ["1 min", "3 min", "5 min", "15 min"]


def _key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def attach_obv(rows):
    obv = 0.0
    prev = None
    for r in rows or []:
        try:
            c = float(r.get("price"))
            v = float(r.get("volume") or 0)
        except Exception:
            c, v = None, 0.0
        if prev is not None and c is not None:
            if c > prev:
                obv += v
            elif c < prev:
                obv -= v
        r["OBV"] = round(obv, 2)
        if c is not None:
            prev = c
    return rows


def _today_rows(rows):
    day = _ist_now().strftime("%Y-%m-%d")
    return [r for r in (rows or []) if str(r.get("time") or "").startswith(day)]


def session_hl(rows):
    rs = _today_rows(rows)
    if not rs:
        rs = rows or []
    if not rs:
        return None, None
    return max(float(r["high"]) for r in rs), min(float(r["low"]) for r in rs)


def pdh_pdl(rows):
    today = _ist_now().strftime("%Y-%m-%d")
    days = sorted({str(r.get("time") or "")[:10] for r in (rows or []) if r.get("time")})
    prev = [d for d in days if d < today]
    if not prev:
        return None, None
    d = prev[-1]
    chunk = [r for r in rows if str(r.get("time") or "").startswith(d)]
    if not chunk:
        return None, None
    return max(float(r["high"]) for r in chunk), min(float(r["low"]) for r in chunk)


def volume_profile(rows, n_bins=24, va_frac=0.70):
    rs = _today_rows(rows) or list(rows or [])
    if len(rs) < 3:
        return {"POC": None, "VAH": None, "VAL": None, "HVN": [], "LVN": [], "fat": []}
    lo = min(float(r.get("low") if r.get("low") is not None else r.get("price") or 0) for r in rs)
    hi = max(float(r.get("high") if r.get("high") is not None else r.get("price") or 0) for r in rs)
    if hi <= lo:
        return {"POC": round(hi, 2), "VAH": round(hi, 2), "VAL": round(lo, 2), "HVN": [], "LVN": [], "fat": []}
    width = (hi - lo) / n_bins
    bins = [0.0] * n_bins
    centers = [lo + (i + 0.5) * width for i in range(n_bins)]
    for r in rs:
        px = float(r.get("price") or r.get("close") or 0)
        v = float(r.get("volume") or 0)
        i = int((px - lo) / width)
        i = max(0, min(n_bins - 1, i))
        bins[i] += v
    poc_i = max(range(n_bins), key=lambda i: bins[i])
    total = sum(bins) or 1.0
    acc = bins[poc_i]
    L = R = poc_i
    while acc / total < va_frac and (L > 0 or R < n_bins - 1):
        left = bins[L - 1] if L > 0 else -1
        right = bins[R + 1] if R < n_bins - 1 else -1
        if right >= left:
            R = min(n_bins - 1, R + 1)
            acc += max(bins[R], 0)
        else:
            L = max(0, L - 1)
            acc += max(bins[L], 0)
    med = sorted(bins)[n_bins // 2] or 1.0
    fat, hvn, lvn = [], [], []
    for i, v in enumerate(bins):
        if v >= 1.6 * med:
            fat.append(round(centers[i], 2))
        if i > 0 and i < n_bins - 1 and v >= bins[i - 1] and v >= bins[i + 1] and v >= 1.2 * med:
            hvn.append(round(centers[i], 2))
        if i > 0 and i < n_bins - 1 and v <= bins[i - 1] and v <= bins[i + 1] and v <= 0.6 * med:
            lvn.append(round(centers[i], 2))
    return {
        "POC": round(centers[poc_i], 2),
        "VAH": round(lo + (R + 1) * width, 2),
        "VAL": round(lo + L * width, 2),
        "HVN": hvn[:6],
        "LVN": lvn[:6],
        "fat": fat[:8],
    }


def volume_profile_mapped(fut_rows, spot_rows, n_bins=24, va_frac=0.70):
    """Futures volume, binned at the spot price of the same time bar."""
    by = {str(r.get("time"))[:16]: r for r in (spot_rows or [])}
    mapped = []
    for fr in fut_rows or []:
        sp = by.get(str(fr.get("time"))[:16]) or {}
        px = sp.get("price")
        if px is None:
            px = fr.get("price")
        mapped.append({
            "price": px,
            "low": sp.get("low", px),
            "high": sp.get("high", px),
            "volume": fr.get("volume") or 0,
            "time": fr.get("time"),
        })
    return volume_profile(mapped, n_bins=n_bins, va_frac=va_frac)


def _prior_swings(rows, sessions=3):
    """Swing highs/lows from the previous N completed sessions."""
    today = _ist_now().strftime("%Y-%m-%d")
    days = sorted({str(r.get("time") or "")[:10] for r in (rows or []) if r.get("time") and str(r.get("time"))[:10] < today})
    days = days[-sessions:]
    out = []
    for d in days:
        chunk = [r for r in rows if str(r.get("time") or "").startswith(d)]
        if len(chunk) < 5:
            continue
        for i in range(2, len(chunk) - 2):
            w = chunk[i - 2:i + 3]
            hi = float(chunk[i].get("high") or chunk[i].get("price") or 0)
            lo = float(chunk[i].get("low") or chunk[i].get("price") or 0)
            if hi and hi == max(float(x.get("high") or x.get("price") or 0) for x in w):
                out.append({"day": d, "side": "SH", "price": round(hi, 2), "time": chunk[i].get("time")})
            if lo and lo == min(float(x.get("low") or x.get("price") or 0) for x in w):
                out.append({"day": d, "side": "SL", "price": round(lo, 2), "time": chunk[i].get("time")})
    return days, out


def _nearest_sl(swings, ltp):
    below = [s for s in swings if s.get("side") == "SL" and s.get("price") is not None and float(s["price"]) < float(ltp)]
    if not below:
        return None
    return max(below, key=lambda s: float(s["price"]))


def _ema_setups(table, ce_swings, pe_swings):
    """table is newest-first. CE buy on CE X LONG. PE buy on PE 9 cross above 21, shown as spot SHORT."""
    ce = pe = None
    for r in table or []:
        if ce is None and r.get("CE X") == "LONG" and r.get("CE LTP") and (r.get("Spot RSI") or 0) > 65:
            sl = _nearest_sl(ce_swings, r["CE LTP"])
            ce = {
                "side": "CE Buy",
                "trigger": "CE 9 EMA cross above 21 and Spot RSI>65",
                "bar": r.get("time"),
                "ltp": r.get("CE LTP"),
                "rsi": r.get("Spot RSI"),
                "sl": None if not sl else sl["price"],
                "sl_at": None if not sl else sl.get("time"),
                "target": "none — trailing SL",
                "exit": "CE 9 EMA cross below 21 (SHORT)",
            }
        if pe is None and r.get("PE X") == "SHORT" and r.get("PE LTP") and (r.get("Spot RSI") or 100) < 35:
            sl = _nearest_sl(pe_swings, r["PE LTP"])
            pe = {
                "side": "PE Buy",
                "trigger": "PE 9 EMA cross above 21 and Spot RSI<35",
                "bar": r.get("time"),
                "ltp": r.get("PE LTP"),
                "rsi": r.get("Spot RSI"),
                "sl": None if not sl else sl["price"],
                "sl_at": None if not sl else sl.get("time"),
                "target": "none — trailing SL",
                "exit": "PE 9 EMA cross below 21 (spot LONG)",
            }
        if ce and pe:
            break
    return ce, pe


def _walk(rows, i, side, sl, entry):
    """After trigger index i. LONG: SL if low<=sl, exit on SHORT cross. Returns result, time, points."""
    for j in range(i + 1, len(rows)):
        r = rows[j]
        try:
            lo = float(r.get("low") if r.get("low") is not None else r.get("price"))
            hi = float(r.get("high") if r.get("high") is not None else r.get("price"))
            px = float(r.get("price"))
            slv = float(sl) if sl is not None else None
            ent = float(entry)
        except Exception:
            continue
        if slv is not None and side == "LONG" and lo <= slv:
            return "SL hit", r.get("time"), round(slv - ent, 2)
        if slv is not None and side == "SHORT" and hi >= slv:
            return "SL hit", r.get("time"), round(ent - slv, 2)
        x = r.get("x")
        if side == "LONG" and x == "SHORT":
            return "Exit cross", r.get("time"), round(px - ent, 2)
        if side == "SHORT" and x == "LONG":
            return "Exit cross", r.get("time"), round(ent - px, 2)
    return "Open", "", None


def _spot_backtest(rows, sessions):
    """Nifty spot 9/21 cross. LONG = CE buy, SHORT = PE buy. No fixed target."""
    days = sorted({str(r.get("time") or "")[:10] for r in rows if r.get("time")})
    keep = set(days[-sessions:]) if sessions else set(days)
    use = [r for r in rows if str(r.get("time") or "")[:10] in keep]
    px = []
    for r in use:
        try:
            px.append(float(r.get("price")))
        except Exception:
            px.append(None)
    e9, e21 = _ema(px, 9), _ema(px, 21)
    rsi = _rsi(px, 14)
    marks = _cross(e9, e21)
    swings = []
    for i in range(2, len(use) - 2):
        w = use[i - 2:i + 3]
        hi = float(use[i].get("high") or use[i].get("price") or 0)
        lo = float(use[i].get("low") or use[i].get("price") or 0)
        if hi and hi == max(float(x.get("high") or x.get("price") or 0) for x in w):
            swings.append({"i": i, "side": "SH", "price": hi, "time": use[i].get("time")})
        if lo and lo == min(float(x.get("low") or x.get("price") or 0) for x in w):
            swings.append({"i": i, "side": "SL", "price": lo, "time": use[i].get("time")})
    path = []
    for i, r in enumerate(use):
        path.append({"price": r.get("price"), "low": r.get("low"), "high": r.get("high"), "time": r.get("time"), "x": marks[i]})
    out = []
    for i, m in enumerate(marks):
        if m not in ("LONG", "SHORT"):
            continue
        rv = rsi[i]
        if m == "LONG" and not (rv is not None and rv > 65):
            continue
        if m == "SHORT" and not (rv is not None and rv < 35):
            continue
        ltp = px[i]
        if ltp is None:
            continue
        if m == "LONG":
            cands = [s for s in swings if s["i"] < i and s["side"] == "SL" and s["price"] < ltp]
            sl = max(cands, key=lambda s: s["price"]) if cands else None
            side = "CE Buy"
        else:
            cands = [s for s in swings if s["i"] < i and s["side"] == "SH" and s["price"] > ltp]
            sl = min(cands, key=lambda s: s["price"]) if cands else None
            side = "PE Buy"
        res, when, pts = _walk(path, i, m, None if not sl else sl["price"], ltp)
        out.append({
            "side": side,
            "trigger": f"Spot 9/21 {m} RSI {rv}",
            "bar": use[i].get("time"),
            "ltp": round(ltp, 2),
            "sl": None if not sl else round(sl["price"], 2),
            "result": res,
            "hit": when,
            "points": pts,
            "target": "none — trailing SL",
            "exit": "opposite 9/21 cross",
        })
    return out


def _fmt_vp(vp):
    if not vp or not vp.get("POC"):
        return "—"
    return (
        f"POC {vp['POC']}  VAL {vp['VAL']}  VAH {vp['VAH']}  "
        f"fat {vp['fat'] or '—'}  HVN {vp['HVN'] or '—'}  LVN {vp['LVN'] or '—'}"
    )


def _live_models(key):
    cached = st.session_state.get("_gs_model_ids")
    if cached:
        return list(cached)
    ids = []
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": key, "pageSize": 100},
            headers={"x-goog-api-key": key},
            timeout=20,
        )
        for m in (r.json() or {}).get("models") or []:
            name = str(m.get("name") or "").split("/")[-1]
            methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
            acts = m.get("supportedActions") or m.get("supported_actions") or []
            ok = (not methods and not acts) or ("generateContent" in methods) or ("generateContent" in acts)
            if name.startswith("gemini") and "tts" not in name and "embed" not in name and ok:
                ids.append(name)
    except Exception:
        ids = []
    prefer = [
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
        "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3-flash",
        "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest",
    ]
    ordered = [p for p in prefer if p in ids] + [i for i in ids if i not in prefer]
    if not ordered:
        ordered = prefer
    st.session_state["_gs_model_ids"] = ordered
    return ordered


def _gemini(prompt, max_tokens=2500):
    key = _key()
    if not key:
        return None, "No GEMINI_API_KEY"
    cool = float(st.session_state.get("_gs_cool_until") or 0)
    now = time.time()
    if now < cool:
        return None, f"cooldown {int(cool - now)}s — last call was 429/503"
    model = st.session_state.get("_gs_ok_model") or "gemini-3.5-flash-lite"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
    }
    try:
        st.session_state["_gs_call_ts"] = now
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key},
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=80,
        )
    except Exception as e:
        return None, str(e)[:80]
    detail = ""
    try:
        errj = r.json()
        detail = str(((errj.get("error") or {}).get("message") or errj))[:240]
    except Exception:
        detail = (r.text or "")[:180]
    st.session_state["_gs_http"] = f"{model} HTTP {r.status_code} {detail}"
    if r.status_code in (429, 503):
        st.session_state["_gs_cool_until"] = time.time() + 90
        return None, f"{model} {r.status_code} {detail}"
    if r.status_code >= 400:
        return None, f"{model} {r.status_code} {detail}"
    parts = (((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    txt = "".join(p.get("text", "") for p in parts).strip()
    if txt:
        st.session_state["_gs_ok_model"] = model
        return txt, model
    return None, f"{model} empty"


def _parse_json(txt):
    raw = txt or ""
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        if raw.lstrip().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        try:
            return json.loads(raw[a:b + 1]) if a >= 0 and b > a else {}
        except Exception:
            return {}


def _last(rows):
    return rows[-1] if rows else {}


def seed_gs(name, fetch_fn, df_master, token_map, fut_fn, tf_lab):
    old = st.session_state.get("mis_tf")
    hold = st.session_state.get("_mis_books")
    st.session_state["mis_tf"] = tf_lab
    st.session_state["_mis_books"] = {}
    try:
        book = dict(seed_book(name, fetch_fn, df_master, token_map, fut_fn=fut_fn) or {})
    finally:
        if old:
            st.session_state["mis_tf"] = old
        if hold is not None:
            st.session_state["_mis_books"] = hold
    for k in ("idx_rows", "ce_rows", "pe_rows"):
        book[k] = attach_obv(attach_efi(list(book.get(k) or [])))
    if fetch_fn and book.get("cash_tok"):
        try:
            interval = TF_API.get(tf_lab, "FIVE_MINUTE")
            dspot, _ = fetch_fn(str(book["cash_tok"]), book.get("cash_exch") or "NSE", interval, name, "gs_spot")
            book["spot_rows"] = rows_from_df(dspot)
            if book["spot_rows"]:
                book["spot"] = book["spot_rows"][-1]["price"]
        except Exception:
            book["spot_rows"] = []
    st.session_state["_gs_book"] = book
    st.session_state["_gs_seed_ts"] = _ist_now().strftime("%H:%M:%S")
    return book


def live_tick(book, quote_fn):
    if not book or not quote_fn:
        return book
    pairs = []
    for tok, exch in (
        (book.get("idx_tok"), book.get("idx_exch")),
        (book.get("cash_tok"), book.get("cash_exch") or "NSE"),
        (book.get("ce_tok"), book.get("idx_exch")),
        (book.get("pe_tok"), book.get("idx_exch")),
    ):
        if tok:
            pairs.append((exch or "NFO", str(tok)))
    try:
        q = quote_fn(pairs) or {}
    except Exception:
        q = {}
    def px(tok):
        it = q.get(str(tok) or "") or {}
        return it.get("ltp"), it.get("volume"), it.get("ltq")
    if book.get("cash_tok"):
        p, v, qy = px(book["cash_tok"])
        if p is not None:
            book["spot"] = p
    if book.get("idx_tok"):
        p, v, qy = px(book["idx_tok"])
        if p is not None:
            book["fut"] = p
            apply_live_tape(book, "idx", p, v, qy)
    if book.get("ce_tok"):
        p, v, qy = px(book["ce_tok"])
        if p is not None:
            apply_live_tape(book, "ce", p, v, qy)
    if book.get("pe_tok"):
        p, v, qy = px(book["pe_tok"])
        if p is not None:
            apply_live_tape(book, "pe", p, v, qy)
    for k in ("idx_rows", "ce_rows", "pe_rows"):
        book[k] = attach_obv(attach_efi(list(book.get(k) or [])))
    return book


def _rsi(vals, n=14):
    out = [None] * len(vals)
    gain = loss = 0.0
    for i in range(1, len(vals)):
        if vals[i] is None or vals[i - 1] is None:
            continue
        ch = vals[i] - vals[i - 1]
        gain += max(ch, 0)
        loss += max(-ch, 0)
        if i < n:
            continue
        if i == n:
            ag, al = gain / n, loss / n
        else:
            ag = (ag * (n - 1) + max(ch, 0)) / n
            al = (al * (n - 1) + max(-ch, 0)) / n
        out[i] = 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)
    return out


def _ema(vals, n):
    out = [None] * len(vals)
    k = 2.0 / (n + 1)
    ema = None
    for i, v in enumerate(vals):
        if v is None:
            out[i] = ema
            continue
        ema = v if ema is None else (v * k + ema * (1 - k))
        out[i] = round(ema, 2)
    return out


def _cross(e9, e21):
    marks = [""] * len(e9)
    for i in range(1, len(e9)):
        a, b, pa, pb = e9[i], e21[i], e9[i - 1], e21[i - 1]
        if None in (a, b, pa, pb):
            continue
        if pa <= pb and a > b:
            marks[i] = "LONG"
        elif pa >= pb and a < b:
            marks[i] = "SHORT"
    return marks


def _session_table(book, spot, limit=None):
    fut_rows = _today_rows(book.get("idx_rows") or []) or list(book.get("idx_rows") or [])
    ce_rows = _today_rows(book.get("ce_rows") or []) or list(book.get("ce_rows") or [])
    pe_rows = _today_rows(book.get("pe_rows") or []) or list(book.get("pe_rows") or [])
    sp_rows = _today_rows(book.get("spot_rows") or []) or list(book.get("spot_rows") or [])
    by_ce = {str(r.get("time"))[:16]: r for r in ce_rows}
    by_pe = {str(r.get("time"))[:16]: r for r in pe_rows}
    by_sp = {str(r.get("time"))[:16]: r for r in sp_rows}
    chrono = list(fut_rows)
    if limit:
        chrono = chrono[-limit:]
    spots, ce_px, pe_px = [], [], []
    pv = vv = 0.0
    vwaps = []
    for fr in chrono:
        key = str(fr.get("time"))[:16]
        sp = by_sp.get(key) or {}
        cr = by_ce.get(key) or {}
        pr = by_pe.get(key) or {}
        try:
            s = float(sp.get("price")) if sp.get("price") is not None else float(fr.get("price"))
        except Exception:
            s = None
        v = float(fr.get("volume") or 0)
        if s is not None and v:
            pv += s * v
            vv += v
        vwaps.append(round(pv / vv, 2) if vv else None)
        spots.append(s)
        try:
            ce_px.append(float(cr.get("price")) if cr.get("price") is not None else None)
        except Exception:
            ce_px.append(None)
        try:
            pe_px.append(float(pr.get("price")) if pr.get("price") is not None else None)
        except Exception:
            pe_px.append(None)
    s9, s21 = _ema(spots, 9), _ema(spots, 21)
    rsi = _rsi(spots, 14)
    c9, c21 = _ema(ce_px, 9), _ema(ce_px, 21)
    p9, p21 = _ema(pe_px, 9), _ema(pe_px, 21)
    sx, cx, px = _cross(s9, s21), _cross(c9, c21), _cross(p9, p21)
    # PE premium up = spot down. Flip so the tag is the spot bias.
    px = [{"LONG": "SHORT", "SHORT": "LONG"}.get(x, x) for x in px]
    out = []
    for i, fr in enumerate(chrono):
        key = str(fr.get("time"))[:16]
        cr = by_ce.get(key) or {}
        pr = by_pe.get(key) or {}
        out.append({
            "time": fr.get("time"),
            "Spot": spots[i],
            "Spot VWAP": vwaps[i],
            "Spot 21 EMA": s21[i],
            "Spot 9 EMA": s9[i],
            "Spot RSI": rsi[i],
            "Spot X": sx[i],
            "CE LTP": ce_px[i] if ce_px[i] is not None else cr.get("price"),
            "CE 21 EMA": c21[i],
            "CE 9 EMA": c9[i],
            "CE X": cx[i],
            "PE LTP": pe_px[i] if pe_px[i] is not None else pr.get("price"),
            "PE 21 EMA": p21[i],
            "PE 9 EMA": p9[i],
            "PE X": px[i],
            "Fut": fr.get("price"),
        })
    out.reverse()
    if spot is not None and out:
        out[0]["Spot"] = spot
    return out


def _color_first(df):
    if df is None or len(df) < 2:
        return df.style if df is not None else None
    live, prev = df.iloc[0], df.iloc[1]
    cols = [c for c in df.columns if c != "time"]

    def style_row(row):
        if row.name != 0:
            return [""] * len(row)
        out = []
        for c in row.index:
            if c == "time":
                out.append("background-color:#1b5e20;color:#fff")
                continue
            try:
                a, b = float(live[c]), float(prev[c])
            except Exception:
                out.append("")
                continue
            if a > b:
                out.append("background-color:#2e7d32;color:#fff")
            elif a < b:
                out.append("background-color:#c62828;color:#fff")
            else:
                out.append("background-color:#37474f;color:#fff")
        return out

    return df.style.apply(style_row, axis=1)


def _digest(book, vp_f, vp_c, vp_p, table_tail, max_loss):
    fut_r, ce_r, pe_r = _last(book.get("idx_rows")), _last(book.get("ce_rows")), _last(book.get("pe_rows"))
    sh, sl = session_hl(book.get("idx_rows"))
    pdh, pdl = pdh_pdl(book.get("idx_rows"))
    lines = [
        f"INDEX {book.get('name')} ATM {book.get('atm')} EXP {book.get('exp')} LOT {LOT_SIZES.get(book.get('name'),1)} MAXLOSS {max_loss}",
        f"SPOT {book.get('spot')} FUT {book.get('fut')} CE {book.get('ce_sym')} LTP {ce_r.get('price')} PE {book.get('pe_sym')} LTP {pe_r.get('price')}",
        f"SESSION HIGH {sh} LOW {sl} PDH {pdh} PDL {pdl}",
        f"FUT VP {_fmt_vp(vp_f)}",
        f"CE VP {_fmt_vp(vp_c)}",
        f"PE VP {_fmt_vp(vp_p)}",
        f"LAST FUT CVD {fut_r.get('CVD')} EFI {fut_r.get('EFI')} OBV {fut_r.get('OBV')} VWAP {fut_r.get('VWAP')}",
        f"LAST CE CVD {ce_r.get('CVD')} EFI {ce_r.get('EFI')} OBV {ce_r.get('OBV')} VWAP {ce_r.get('VWAP')}",
        f"LAST PE EFI {pe_r.get('EFI')} OBV {pe_r.get('OBV')} VWAP {pe_r.get('VWAP')}",
        f"BARS sent {min(len(table_tail), 24)} most-recent of {len(table_tail)} session (newest first)",
        "time S F Fcvd Fefi Fobv Fvw CE CEcvd CEefi CEobv PE PEefi PEobv PEvw",
    ]
    # session extremes only + last 24 — keeps TPM down so 429 is less likely
    extra = []
    if table_tail:
        extra.append(table_tail[-1])  # oldest of the sent? table is newest-first so last is oldest recent
        # session high / low futures from full list
        try:
            hi = max(table_tail, key=lambda r: float(r.get("Fut") or 0))
            lo = min((r for r in table_tail if r.get("Fut") is not None), key=lambda r: float(r.get("Fut") or 0))
            extra.extend([hi, lo])
        except Exception:
            pass
    seen = set()
    slim = []
    for r in extra + list(table_tail[:24]):
        k = r.get("time")
        if k in seen:
            continue
        seen.add(k)
        slim.append(r)
    for r in slim:
        lines.append(
            f"{r.get('time')} Spot={r.get('Spot')} Svw={r.get('Spot VWAP')} S9={r.get('Spot 9 EMA')} S21={r.get('Spot 21 EMA')} Sx={r.get('Spot X')} "
            f"CE={r.get('CE LTP')} CE9={r.get('CE 9 EMA')} CE21={r.get('CE 21 EMA')} CEx={r.get('CE X')} "
            f"PE={r.get('PE LTP')} PE9={r.get('PE 9 EMA')} PE21={r.get('PE 21 EMA')} PEx={r.get('PE X')}"
        )
    return "\n".join(lines)


def run_setups(book, vp_f, vp_c, vp_p, table, max_loss):
    prompt = (
        "You are an Indian/MCX options scalper. BUY only — never sell/write premium.\n"
        "DATA already has live LTPs, VP nodes, session H/L, PDH/PDL, and the last 24 tape rows.\n"
        "You MUST output one CE buy watch AND one PE buy watch. Do not answer NO TRADE when LTPs exist.\n"
        "Build trigger from Fut VP / session high or low / PDH-PDL plus option LTP vs its VP/VWAP.\n"
        "Type examples: SETUP WATCH - LONG BREAKOUT | SETUP WATCH - SHORT BREAKDOWN | SETUP WATCH - MEAN REVERSION.\n"
        "Return ONLY JSON:\n"
        '{"CE":{"watch":true,"type":"SETUP WATCH - LONG BREAKOUT","trigger":"Fut > session high & CE > POC",'
        '"entry":"222-224","target":240,"tgt_pct":8.1,"sl":210,"sl_pct":-5.4,'
        '"lots":2,"max_profit":2600,"max_loss":1400,"rr":"1/1.3",'
        '"rationale":"cite VP/PDH/EFI/CVD from DATA"},'
        '"PE":{"watch":true,"type":"SETUP WATCH - SHORT BREAKDOWN","trigger":"...","entry":"...","target":0,'
        '"tgt_pct":0,"sl":0,"sl_pct":0,"lots":1,"max_profit":0,"max_loss":0,"rr":"1/1.3","rationale":"..."}}\n'
        "lots = floor(MAXLOSS / ((entry_mid-sl)*LOT)). Target above entry. SL below entry.\n"
        "DATA:\n"
    )
    try:
        prompt = prompt + _digest(book, vp_f, vp_c, vp_p, table, max_loss)
    except NameError:
        prompt = prompt + str({"spot": book.get("spot"), "fut": book.get("fut"), "rows": len(table or [])})
    txt, model = _gemini(prompt)
    st.session_state["_gs_setup_raw"] = (txt or str(model) or "")[:2500]
    if not txt:
        return {"CE": {"watch": False, "type": "NO TRADE", "rationale": str(model)},
                "PE": {"watch": False, "type": "NO TRADE", "rationale": str(model)},
                "_model": model}, model or "err"
    data = _parse_json(txt)
    if not isinstance(data, dict) or not data:
        low = (txt or "").replace(" ", "")
        data = {}
    # accept lowercase / nested
    if "CE" not in data and "ce" in data:
        data["CE"] = data.get("ce")
    if "PE" not in data and "pe" in data:
        data["PE"] = data.get("pe")
    for side in ("CE", "PE"):
        b = data.get(side)
        if isinstance(b, dict):
            if b.get("entry") or b.get("trigger") or (b.get("type") and "NO TRADE" not in str(b.get("type")).upper()):
                b["watch"] = True
    if not data.get("CE") and not data.get("PE"):
        data["CE"] = {"watch": False, "type": "NO TRADE", "rationale": (txt or "")[:400]}
        data["PE"] = {"watch": False, "type": "NO TRADE", "rationale": "JSON parse failed"}
    data["_model"] = model
    return data, model


def run_watch(book, pos, vp_f, vp_c, vp_p, table):
    prompt = (
        "You manage an OPEN option long. Do not invent a new trade.\n"
        "Position: " + json.dumps(pos) + "\n"
        "Return ONLY JSON:\n"
        '{"action":"Hold - Bullish trend is still active",'
        '"trail_sl":"LTP>230 => trail SL to 225",'
        '"validation":"Futures holds above 24575",'
        '"invalidation":"Futures fails below 24550, keep SL at 220"}\n'
        "DATA:\n" + _digest(book, vp_f, vp_c, vp_p, table, pos.get("max_loss") or 2000)
    )
    txt, model = _gemini(prompt, max_tokens=800)
    if not txt:
        return {}, model or "err"
    data = _parse_json(txt)
    data["_model"] = model
    data["_ts"] = _ist_now().strftime("%H:%M:%S")
    return data, model


def _mid_px(entry):
    s = str(entry or "").replace(" ", "")
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _size_side(b, max_loss, lot):
    b = dict(b or {})
    mid = _mid_px(b.get("entry"))
    try:
        sl = float(b.get("sl"))
    except Exception:
        sl = None
    try:
        tgt = float(b.get("target"))
    except Exception:
        tgt = None
    risk = abs(mid - sl) if mid is not None and sl is not None else 0.0
    per_lot = risk * float(lot or 0)
    lots = int(float(max_loss) // per_lot) if per_lot else 0
    qty = lots * int(lot or 0)
    mx = round(risk * qty, 0) if qty else max_loss
    mp = round(abs(tgt - mid) * qty, 0) if qty and tgt is not None and mid is not None else None
    rr = f"1 to {round(abs(tgt - mid) / risk, 2)}" if risk and tgt is not None and mid is not None else b.get("rr")
    b["qty"] = qty
    b["lots"] = lots
    b["max_loss"] = mx
    if mp is not None:
        b["max_profit"] = mp
    if rr:
        b["rr"] = rr
    return b


def _setup_table(ce, pe, max_loss, lot=10):
    ce = _size_side(ce if isinstance(ce, dict) else {}, max_loss, lot)
    pe = _size_side(pe if isinstance(pe, dict) else {}, max_loss, lot)

    def g(b, k, default=""):
        v = (b or {}).get(k)
        return default if v is None or v == "" else v

    def tgt(b):
        t, p = (b or {}).get("target"), (b or {}).get("tgt_pct")
        if t is None:
            return "—"
        return f"{t} ({p} %)" if p is not None else str(t)

    def sl(b):
        s, p = (b or {}).get("sl"), (b or {}).get("sl_pct")
        if s is None:
            return "—"
        return f"{s} ({p} %)" if p is not None else str(s)

    rows = [
        ("SETUP", g(ce, "type", "NO TRADE CE"), g(pe, "type", "NO TRADE PE")),
        ("Trigger", g(ce, "trigger"), g(pe, "trigger")),
        ("Entry", g(ce, "entry"), g(pe, "entry")),
        ("Target", tgt(ce), tgt(pe)),
        ("SL", sl(ce), sl(pe)),
        ("Entry Qty", g(ce, "qty"), g(pe, "qty")),
        ("max Profit", g(ce, "max_profit"), g(pe, "max_profit")),
        ("max Loss", g(ce, "max_loss", max_loss), g(pe, "max_loss", max_loss)),
        ("Risk to Reward", g(ce, "rr"), g(pe, "rr")),
        ("Rationale", g(ce, "rationale"), g(pe, "rationale")),
    ]
    body = "".join(
        f"<tr><td style='color:#FFD54F;white-space:nowrap;padding:3px 6px;border:1px solid #2a2d33;font-size:12px;'>{a}</td>"
        f"<td style='padding:3px 6px;border:1px solid #2a2d33;font-size:12px;color:#ECEFF1;'>{b}</td>"
        f"<td style='padding:3px 6px;border:1px solid #2a2d33;font-size:12px;color:#ECEFF1;'>{c}</td></tr>"
        for a, b, c in rows
    )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;background:#0E1117;'>"
        "<tr><th style='padding:4px 6px;border:1px solid #2a2d33;color:#90A4AE;font-size:11px;'></th>"
        "<th style='padding:4px 6px;border:1px solid #2a2d33;color:#69F0AE;font-size:11px;'>CE buy</th>"
        "<th style='padding:4px 6px;border:1px solid #2a2d33;color:#FF8A80;font-size:11px;'>PE buy</th></tr>"
        + body + "</table>",
        unsafe_allow_html=True,
    )


def render_gemini_scalper(fetch_fn=None, df_master=None, token_map=None, get_client=None, fut_fn=None, quote_fn=None):
    st.session_state["gemini_enabled"] = False
    if st.session_state.get("_gs_ok_model") == "gemini-3.8-flash":
        st.session_state["_gs_ok_model"] = "gemini-3.5-flash-lite"
    name = st.session_state.get("gs_index") or "NIFTY"
    tf = st.session_state.get("gs_tf") or "5 min"
    top = st.columns([1.6, 0.9, 0.7, 0.7, 1.1])
    with top[0]:
        st.markdown("<span style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Gemini Scalper</span>", unsafe_allow_html=True)
        st.caption("build 0930-digest")
    with top[1]:
        name = st.selectbox("Index", ORDER, index=ORDER.index(name) if name in ORDER else 0, key="gs_index")
    with top[2]:
        tf = st.selectbox("TF", TFS, index=TFS.index(tf) if tf in TFS else 2, key="gs_tf")
        if tf != st.session_state.get("gs_tf_applied"):
            st.session_state["gs_tf_applied"] = tf
            st.session_state["gs_need_seed"] = True
    with top[3]:
        auto = st.checkbox("Auto 5s", value=bool(st.session_state.get("enable_main_refresh")), key="gs_auto")
        if auto != bool(st.session_state.get("enable_main_refresh")):
            st.session_state["enable_main_refresh"] = auto
            st.rerun()
    with top[4]:
        if st.button("Seed / refresh tapes", key="gs_seed"):
            st.session_state["gs_need_seed"] = True

    if st.session_state.pop("gs_need_seed", False) and fetch_fn and token_map is not None:
        with st.spinner(f"Seeding {name} {tf}…"):
            seed_gs(name, fetch_fn, df_master, token_map, fut_fn, tf)

    book = st.session_state.get("_gs_book") or {}
    if book.get("name") != name and fetch_fn:
        st.session_state["gs_need_seed"] = True
        st.info("Index changed — click Seed / refresh tapes.")
    if quote_fn and book:
        book = live_tick(book, quote_fn)
        st.session_state["_gs_book"] = book

    if not book.get("idx_rows"):
        st.info("Seed tapes for the selected index and TF.")
        return

    fut_rows = book.get("idx_rows") or []
    ce_rows = book.get("ce_rows") or []
    pe_rows = book.get("pe_rows") or []
    vp_spot = volume_profile_mapped(_today_rows(fut_rows) or fut_rows, book.get("spot_rows") or [])
    sh, sl = session_hl(fut_rows)
    pdh, pdl = pdh_pdl(fut_rows)
    sw_days, sw = _prior_swings(book.get("spot_rows") or fut_rows, 3)
    ce_l, pe_l = _last(ce_rows), _last(pe_rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot", book.get("spot") or "—")
    m2.metric("Futures", book.get("fut") or "—")
    m3.metric("ATM CE LTP", ce_l.get("price") or "—")
    m4.metric("ATM PE LTP", pe_l.get("price") or "—")
    st.caption(
        f"{book.get('ce_sym') or ''} / {book.get('pe_sym') or ''} · ATM {book.get('atm')} · exp {book.get('exp')} · "
        f"seed {st.session_state.get('_gs_seed_ts') or '—'}"
    )
    st.markdown(
        f"<div style='font-size:13px;color:#ECEFF1;line-height:1.55;'>"
        f"<b>Spot VP</b> (futures volume at that bar's spot) {_fmt_vp(vp_spot)}<br>"
        f"<b>Prior {len(sw_days)} sessions</b> {', '.join(sw_days) or '—'}<br>"
        f"<b>Swing highs</b> {', '.join(str(s['price'])+' @ '+str(s['time'])[5:16] for s in sw if s['side']=='SH') or '—'}<br>"
        f"<b>Swing lows</b> {', '.join(str(s['price'])+' @ '+str(s['time'])[5:16] for s in sw if s['side']=='SL') or '—'}<br>"
        f"<b>Session</b> high {sh} low {sl} &nbsp; <b>PDH</b> {pdh} &nbsp; <b>PDL</b> {pdl}"
        f"</div>",
        unsafe_allow_html=True,
    )

    spot = book.get("spot")
    full_table = _session_table(book, spot)
    table = full_table[:80]
    df = pd.DataFrame(table)
    st.caption(f"Tape {tf} · Spot VWAP uses futures volume · 9/21 EMA cross = LONG/SHORT")
    try:
        st.dataframe(_color_first(df), use_container_width=True, hide_index=True, height=320)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, height=320)

    _, ce_sw = _prior_swings(ce_rows, 3)
    _, pe_sw = _prior_swings(pe_rows, 3)
    ce_set, pe_set = _ema_setups(full_table, ce_sw, pe_sw)
    chrono = list(reversed(full_table))
    for setup, xkey, side in ((ce_set, "CE X", "LONG"), (pe_set, "PE X", "SHORT")):
        if not setup or not setup.get("bar"):
            continue
        path = []
        start = None
        for i, r in enumerate(chrono):
            path.append({"price": r.get("CE LTP") if side == "LONG" else r.get("PE LTP"),
                         "low": r.get("CE LTP") if side == "LONG" else r.get("PE LTP"),
                         "high": r.get("CE LTP") if side == "LONG" else r.get("PE LTP"),
                         "time": r.get("time"), "x": r.get(xkey)})
            if str(r.get("time")) == str(setup.get("bar")):
                start = i
        if start is not None:
            res, when, pts = _walk(path, start, "LONG", setup.get("sl"), setup.get("ltp"))
            setup["result"] = res
            setup["hit"] = when
            setup["points"] = pts
    st.markdown("**EMA setups**")
    st.dataframe(pd.DataFrame([
        ce_set or {"side": "CE Buy", "trigger": "waiting CE 9>21", "bar": "", "ltp": "", "sl": "", "result": "", "hit": "", "target": "none — trailing SL", "exit": "CE 9<21"},
        pe_set or {"side": "PE Buy", "trigger": "waiting PE 9>21 (spot SHORT)", "bar": "", "ltp": "", "sl": "", "result": "", "hit": "", "target": "none — trailing SL", "exit": "PE 9<21 (spot LONG)"},
    ]), use_container_width=True, hide_index=True)

    bt_on = st.checkbox("Backtest spot 9/21 crosses", key="gs_bt_on")
    if bt_on:
        nses = st.selectbox("Sessions", [30, 50, 100], key="gs_bt_n")
        if st.button("Run backtest", key="gs_bt_go"):
            st.session_state["gs_fetch_days"] = int(nses * 1.6) + 5
            with st.spinner(f"Fetching {nses} sessions…"):
                interval = TF_API.get(tf, "FIVE_MINUTE")
                tok = book.get("cash_tok") or book.get("idx_tok")
                exch = book.get("cash_exch") or book.get("idx_exch")
                dfb, _ = fetch_fn(str(tok), exch, interval, name, "gs_bt")
                st.session_state["gs_fetch_days"] = 5
                rows = rows_from_df(dfb)
            st.session_state["_gs_bt"] = _spot_backtest(rows, int(nses))
            st.session_state["_gs_bt_n"] = len({str(r.get("time") or "")[:10] for r in rows})
        bt = st.session_state.get("_gs_bt") or []
        if bt:
            closed = [r for r in bt if r.get("result") in ("SL hit", "Exit cross")]
            wins = [r for r in closed if (r.get("points") or 0) > 0]
            pts = round(sum(float(r.get("points") or 0) for r in closed), 2)
            wr = round(100.0 * len(wins) / len(closed), 1) if closed else 0
            st.markdown(f"**Win rate {wr}% · total points {pts} · {len(bt)} crosses · days loaded {st.session_state.get('_gs_bt_n')}**")
            st.dataframe(pd.DataFrame(bt), use_container_width=True, hide_index=True, height=360)

    cool_left = max(0, int((st.session_state.get("_gs_cool_until") or 0) - time.time()))
    h1, h2, h3 = st.columns([1.4, 0.7, 0.8])
    with h1:
        st.markdown("**SETUP WATCH**")
        if cool_left:
            st.caption(f"Gemini cooldown {cool_left}s — do not click Run")
    with h2:
        max_loss = st.number_input("Max loss ₹", min_value=500, value=int(st.session_state.get("gs_max_loss") or 2000), step=500, key="gs_max_loss")
    with h3:
        if st.button("Run Gemini setups", key="gs_run_setup", disabled=bool(cool_left)):
            st.session_state["_gs_setup_go"] = True

    if st.session_state.pop("_gs_setup_go", False):
        if cool_left:
            st.warning(f"Blocked — cooldown {cool_left}s")
        else:
            st.info("Gemini running — one call to gemini-3.8-flash…")
            with st.status("Gemini setups running…", expanded=True) as status:
                status.write("One request. No retries.")
                data, model = run_setups(book, vp_spot, vp_spot, vp_spot, full_table, max_loss)
                st.session_state["_gs_setups"] = data
                st.session_state["_gs_setup_model"] = model
                if (data.get("CE") or {}).get("watch") or (data.get("PE") or {}).get("watch"):
                    status.update(label=f"Setups ready · {model}", state="complete")
                else:
                    status.update(label=f"No setups · {model}", state="error")

    setups = st.session_state.get("_gs_setups") or {}
    lot = LOT_SIZES.get(book.get("name") or name, 10)
    c_set, c_pos, c_an = st.columns([0.45, 0.20, 0.35])
    with c_set:
        _setup_table(setups.get("CE"), setups.get("PE"), max_loss, lot)
        if setups.get("_model"):
            st.caption(str(setups.get("_model")) + f" · lot {lot}")
        http = st.session_state.get("_gs_http") or ""
        if http:
            st.caption(http)
        raw = st.session_state.get("_gs_setup_raw") or ""
        if raw and not (setups.get("CE") or {}).get("watch"):
            with st.expander("Gemini raw"):
                st.code(raw[:800], language=None)
    with c_pos:
        st.markdown("**Open position**")
        lots = st.number_input("Entry lots", min_value=0, value=int(st.session_state.get("gs_pos_lots") or 0), step=1, key="gs_pos_lots")
        entry = st.number_input("Entry price", min_value=0.0, value=float(st.session_state.get("gs_pos_entry") or 0.0), step=0.05, key="gs_pos_entry")
        sl = st.number_input("SL", min_value=0.0, value=float(st.session_state.get("gs_pos_sl") or 0.0), step=0.05, key="gs_pos_sl")
        tgt = st.number_input("Target", min_value=0.0, value=float(st.session_state.get("gs_pos_tgt") or 0.0), step=0.05, key="gs_pos_tgt")
        side = st.selectbox("Side", ["CE", "PE"], key="gs_pos_side")
        watch = st.checkbox("Gemini Watch (continuous)", key="gs_watch")
        if lots:
            st.caption(f"Qty {int(lots) * int(lot)}  (lot {lot})")
    with c_an:
        st.markdown("**Position analysis**")
        pos = {"lots": lots, "qty": int(lots) * int(lot), "entry": entry, "sl": sl, "target": tgt, "side": side,
               "max_loss": max_loss, "ltp": (ce_l if side == "CE" else pe_l).get("price")}
        if watch and lots and entry and cool_left == 0:
            last = float(st.session_state.get("_gs_watch_ts") or 0)
            busy = bool(st.session_state.get("_gs_watch_busy"))
            if (not busy) and (time.time() - last >= 60):
                st.session_state["_gs_watch_busy"] = True
                try:
                    w, model = run_watch(book, pos, vp_f, vp_c, vp_p, full_table)
                    st.session_state["_gs_watch_out"] = w
                    st.session_state["_gs_watch_ts"] = time.time()
                finally:
                    st.session_state["_gs_watch_busy"] = False
        w = st.session_state.get("_gs_watch_out") or {}
        if w:
            st.markdown(
                f"<div style='font-size:13px;line-height:1.5;color:#ECEFF1;'>"
                f"<b>Action</b> : {w.get('action') or '—'}<br>"
                f"<b>Trail SL</b> : {w.get('trail_sl') or '—'}<br>"
                f"<b>Validation</b> : {w.get('validation') or '—'}<br>"
                f"<b>Invalidation</b> : {w.get('invalidation') or '—'}<br>"
                f"<span style='color:#90A4AE;'>{w.get('_ts') or ''} · {w.get('_model') or ''}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif watch:
            st.caption("Watch on — first pass after 10s.")
        else:
            st.caption("Tick Gemini Watch for live position analysis.")
