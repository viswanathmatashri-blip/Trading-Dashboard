"""Gemini Scalper — spot VWAP, 9/21 EMA, RSI, mapped VP, EMA setups, spot backtest."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

from multi_index_scalper import (
    ORDER, LOT_SIZES, TF_API,
    rows_from_df, attach_efi, seed_book, apply_live_tape,
    _ist_now,
)

TFS = ["1 min", "3 min", "5 min", "15 min"]


def _key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _today_rows(rows):
    day = _ist_now().strftime("%Y-%m-%d")
    hit = [r for r in (rows or []) if str(r.get("time") or "").startswith(day)]
    return hit or list(rows or [])


def session_hl(rows):
    rs = _today_rows(rows)
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
    rs = list(rows or [])
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
        i = max(0, min(n_bins - 1, int((px - lo) / width)))
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
        if 0 < i < n_bins - 1 and v >= bins[i - 1] and v >= bins[i + 1] and v >= 1.2 * med:
            hvn.append(round(centers[i], 2))
        if 0 < i < n_bins - 1 and v <= bins[i - 1] and v <= bins[i + 1] and v <= 0.6 * med:
            lvn.append(round(centers[i], 2))
    return {"POC": round(centers[poc_i], 2), "VAH": round(lo + (R + 1) * width, 2), "VAL": round(lo + L * width, 2),
            "HVN": hvn[:6], "LVN": lvn[:6], "fat": fat[:8]}


def volume_profile_mapped(fut_rows, spot_rows, n_bins=24, va_frac=0.70):
    by = {str(r.get("time"))[:16]: r for r in (spot_rows or [])}
    mapped = []
    for fr in fut_rows or []:
        sp = by.get(str(fr.get("time"))[:16]) or {}
        px = sp.get("price", fr.get("price"))
        mapped.append({"price": px, "low": sp.get("low", px), "high": sp.get("high", px),
                       "volume": fr.get("volume") or 0, "time": fr.get("time")})
    return volume_profile(mapped, n_bins=n_bins, va_frac=va_frac)


def _fmt_vp(vp):
    if not vp or not vp.get("POC"):
        return "—"
    return (f"POC {vp['POC']}  VAL {vp['VAL']}  VAH {vp['VAH']}  "
            f"fat {vp['fat'] or '—'}  HVN {vp['HVN'] or '—'}  LVN {vp['LVN'] or '—'}")


def _ema(vals, n):
    out, ema, k = [None] * len(vals), None, 2.0 / (n + 1)
    for i, v in enumerate(vals):
        if v is None:
            out[i] = ema
            continue
        ema = v if ema is None else (v * k + ema * (1 - k))
        out[i] = round(ema, 2)
    return out


def _rsi(vals, n=14):
    out = [None] * len(vals)
    gain = loss = 0.0
    ag = al = None
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


def _prior_swings(rows, sessions=3):
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


def _nearest_below(swings, level):
    below = [s for s in swings if s.get("price") is not None and float(s["price"]) < float(level)]
    return max(below, key=lambda s: float(s["price"])) if below else None


def _session_table(book, spot, limit=None):
    fut_rows = _today_rows(book.get("idx_rows") or [])
    ce_rows = _today_rows(book.get("ce_rows") or [])
    pe_rows = _today_rows(book.get("pe_rows") or [])
    sp_rows = _today_rows(book.get("spot_rows") or [])
    by_ce = {str(r.get("time"))[:16]: r for r in ce_rows}
    by_pe = {str(r.get("time"))[:16]: r for r in pe_rows}
    by_sp = {str(r.get("time"))[:16]: r for r in sp_rows}
    chrono = list(fut_rows)
    if limit:
        chrono = chrono[-limit:]
    spots, ce_px, pe_px, vwaps = [], [], [], []
    pv = vv = 0.0
    for fr in chrono:
        key = str(fr.get("time"))[:16]
        sp, cr, pr = by_sp.get(key) or {}, by_ce.get(key) or {}, by_pe.get(key) or {}
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
        ce_px.append(float(cr["price"]) if cr.get("price") is not None else None)
        pe_px.append(float(pr["price"]) if pr.get("price") is not None else None)
    s9, s21, rsi = _ema(spots, 9), _ema(spots, 21), _rsi(spots, 14)
    c9, c21 = _ema(ce_px, 9), _ema(ce_px, 21)
    p9, p21 = _ema(pe_px, 9), _ema(pe_px, 21)
    sx, cx = _cross(s9, s21), _cross(c9, c21)
    px = [{"LONG": "SHORT", "SHORT": "LONG"}.get(x, x) for x in _cross(p9, p21)]
    out = []
    for i, fr in enumerate(chrono):
        key = str(fr.get("time"))[:16]
        cr, pr = by_ce.get(key) or {}, by_pe.get(key) or {}
        out.append({
            "time": fr.get("time"), "Spot": spots[i], "Spot VWAP": vwaps[i],
            "Spot 21 EMA": s21[i], "Spot 9 EMA": s9[i], "Spot RSI": rsi[i], "Spot X": sx[i],
            "CE LTP": ce_px[i], "CE low": cr.get("low"), "CE 21 EMA": c21[i], "CE 9 EMA": c9[i], "CE X": cx[i],
            "PE LTP": pe_px[i], "PE low": pr.get("low"), "PE 21 EMA": p21[i], "PE 9 EMA": p9[i], "PE X": px[i],
            "Fut": fr.get("price"),
        })
    out.reverse()
    if spot is not None and out:
        out[0]["Spot"] = spot
    return out


def _ema_setups(table, ce_swings, pe_swings):
    ce = pe = None
    for r in table or []:
        if ce is None and r.get("CE X") == "LONG" and r.get("CE LTP") and (r.get("Spot RSI") or 0) > 65:
            level = r.get("CE low") or r.get("CE LTP")
            sl = _nearest_below(ce_swings, level)
            ce = {"side": "CE Buy", "trigger": "CE 9>21 and Spot RSI>65", "bar": r.get("time"),
                  "ltp": r.get("CE LTP"), "bar_low": level, "rsi": r.get("Spot RSI"),
                  "sl": None if not sl else sl["price"], "sl_at": None if not sl else sl.get("time"),
                  "target": "none — trailing SL", "exit": "CE 9<21"}
        if pe is None and r.get("PE X") == "SHORT" and r.get("PE LTP") and (r.get("Spot RSI") or 100) < 35:
            level = r.get("PE low") or r.get("PE LTP")
            sl = _nearest_below(pe_swings, level)
            pe = {"side": "PE Buy", "trigger": "PE 9>21 and Spot RSI<35", "bar": r.get("time"),
                  "ltp": r.get("PE LTP"), "bar_low": level, "rsi": r.get("Spot RSI"),
                  "sl": None if not sl else sl["price"], "sl_at": None if not sl else sl.get("time"),
                  "target": "none — trailing SL", "exit": "PE 9<21 (spot LONG)"}
        if ce and pe:
            break
    return ce, pe


def _walk(rows, i, sl, entry):
    for j in range(i + 1, len(rows)):
        r = rows[j]
        try:
            lo = float(r.get("low") if r.get("low") is not None else r.get("price"))
            px = float(r.get("price"))
            ent = float(entry)
            slv = float(sl) if sl is not None else None
        except Exception:
            continue
        if slv is not None and lo <= slv:
            return "SL hit", r.get("time"), round(slv - ent, 2)
        if r.get("x") == "SHORT":
            return "Exit cross", r.get("time"), round(px - ent, 2)
    return "Open", "", None


def _spot_backtest(rows, sessions):
    days = sorted({str(r.get("time") or "")[:10] for r in rows if r.get("time")})
    keep = set(days[-sessions:]) if sessions else set(days)
    use = [r for r in rows if str(r.get("time") or "")[:10] in keep]
    px = [float(r["price"]) if r.get("price") is not None else None for r in use]
    e9, e21, rsi = _ema(px, 9), _ema(px, 21), _rsi(px, 14)
    marks = _cross(e9, e21)
    swings = []
    for i in range(2, max(2, len(use) - 2)):
        w = use[i - 2:i + 3]
        hi = float(use[i].get("high") or use[i].get("price") or 0)
        lo = float(use[i].get("low") or use[i].get("price") or 0)
        if hi and hi == max(float(x.get("high") or x.get("price") or 0) for x in w):
            swings.append({"i": i, "side": "SH", "price": hi})
        if lo and lo == min(float(x.get("low") or x.get("price") or 0) for x in w):
            swings.append({"i": i, "side": "SL", "price": lo})
    path = [{"price": r.get("price"), "low": r.get("low"), "high": r.get("high"), "time": r.get("time"), "x": marks[i]} for i, r in enumerate(use)]
    out = []
    for i, m in enumerate(marks):
        rv = rsi[i]
        if m == "LONG" and not (rv is not None and rv > 65):
            continue
        if m == "SHORT" and not (rv is not None and rv < 35):
            continue
        if m not in ("LONG", "SHORT") or px[i] is None:
            continue
        level = float(use[i].get("low") or px[i])
        cands = [s for s in swings if s["i"] < i and s["price"] < level]
        sl = max(cands, key=lambda s: s["price"]) if cands else None
        res, when, pts = _walk(path, i, None if not sl else sl["price"], px[i])
        out.append({"side": "CE Buy" if m == "LONG" else "PE Buy", "trigger": f"Spot 9/21 {m} RSI {rv}",
                    "bar": use[i].get("time"), "ltp": round(px[i], 2), "bar_low": round(level, 2),
                    "sl": None if not sl else round(sl["price"], 2), "result": res, "hit": when, "points": pts,
                    "target": "none — trailing SL", "exit": "opposite 9/21 cross"})
    return out


def seed_gs(name, fetch_fn, df_master, token_map, fut_fn, tf_lab):
    old, hold = st.session_state.get("mis_tf"), st.session_state.get("_mis_books")
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
        book[k] = attach_efi(list(book.get(k) or []))
    if fetch_fn and book.get("cash_tok"):
        try:
            dspot, _ = fetch_fn(str(book["cash_tok"]), book.get("cash_exch") or "NSE", TF_API.get(tf_lab, "FIVE_MINUTE"), name, "gs_spot")
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
    pairs = [(ex or "NFO", str(tok)) for tok, ex in (
        (book.get("idx_tok"), book.get("idx_exch")), (book.get("cash_tok"), book.get("cash_exch") or "NSE"),
        (book.get("ce_tok"), book.get("idx_exch")), (book.get("pe_tok"), book.get("idx_exch"))) if tok]
    try:
        q = quote_fn(pairs) or {}
    except Exception:
        q = {}
    def px(tok):
        it = q.get(str(tok) or "") or {}
        return it.get("ltp"), it.get("volume"), it.get("ltq")
    if book.get("cash_tok"):
        p, _, _ = px(book["cash_tok"])
        if p is not None:
            book["spot"] = p
    if book.get("idx_tok"):
        p, v, qy = px(book["idx_tok"])
        if p is not None:
            book["fut"] = p
            apply_live_tape(book, "idx", p, v, qy)
    for side, tok in (("ce", book.get("ce_tok")), ("pe", book.get("pe_tok"))):
        if tok:
            p, v, qy = px(tok)
            if p is not None:
                apply_live_tape(book, side, p, v, qy)
    return book


def _color_first(df):
    if df is None or len(df) < 2:
        return df
    live, prev = df.iloc[0], df.iloc[1]

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
            out.append("background-color:#2e7d32;color:#fff" if a > b else "background-color:#c62828;color:#fff" if a < b else "")
        return out
    return df.style.apply(style_row, axis=1)


def render_gemini_scalper(fetch_fn=None, df_master=None, token_map=None, get_client=None, fut_fn=None, quote_fn=None):
    st.session_state["gemini_enabled"] = False
    name = st.session_state.get("gs_index") or "NIFTY"
    tf = st.session_state.get("gs_tf") or "5 min"
    top = st.columns([1.6, 0.9, 0.7, 0.7, 1.1])
    with top[0]:
        st.markdown("<span style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Gemini Scalper</span>", unsafe_allow_html=True)
        st.caption("build 1003-rsi")
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
        st.info("Index changed — click Seed / refresh tapes.")
        return
    if quote_fn and book:
        book = live_tick(book, quote_fn)
        st.session_state["_gs_book"] = book
    if not book.get("idx_rows"):
        st.info("Seed tapes for the selected index and TF.")
        return
    fut_rows = book.get("idx_rows") or []
    vp_spot = volume_profile_mapped(_today_rows(fut_rows), book.get("spot_rows") or [])
    sh, sl = session_hl(fut_rows)
    pdh, pdl = pdh_pdl(fut_rows)
    sw_days, sw = _prior_swings(book.get("spot_rows") or fut_rows, 3)
    ce_l, pe_l = (book.get("ce_rows") or [{}])[-1], (book.get("pe_rows") or [{}])[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", book.get("spot") or "—")
    c2.metric("Futures", book.get("fut") or "—")
    c3.metric("ATM CE LTP", ce_l.get("price") or "—")
    c4.metric("ATM PE LTP", pe_l.get("price") or "—")
    st.caption(f"{book.get('ce_sym') or ''} / {book.get('pe_sym') or ''} · ATM {book.get('atm')} · exp {book.get('exp')}")
    st.markdown(
        f"<div style='font-size:12px;color:#ECEFF1;line-height:1.45;'>"
        f"<b>Spot VP</b> {_fmt_vp(vp_spot)}<br>"
        f"<b>Prior {len(sw_days)} sessions</b> {', '.join(sw_days) or '—'}<br>"
        f"<b>Swing highs</b> {', '.join(str(s['price']) for s in sw if s['side']=='SH') or '—'}<br>"
        f"<b>Swing lows</b> {', '.join(str(s['price']) for s in sw if s['side']=='SL') or '—'}<br>"
        f"<b>Session</b> high {sh} low {sl} · PDH {pdh} PDL {pdl}</div>",
        unsafe_allow_html=True,
    )
    full_table = _session_table(book, book.get("spot"))
    show = pd.DataFrame(full_table[:80])
    keep = [c for c in show.columns if c not in ("CE low", "PE low")]
    st.caption(f"Tape {tf} · Spot VWAP = spot × futures volume · SL uses trigger-bar low")
    try:
        st.dataframe(_color_first(show[keep]), use_container_width=True, hide_index=True, height=320)
    except Exception:
        st.dataframe(show[keep], use_container_width=True, hide_index=True, height=320)
    _, ce_sw = _prior_swings(book.get("ce_rows") or [], 3)
    _, pe_sw = _prior_swings(book.get("pe_rows") or [], 3)
    ce_set, pe_set = _ema_setups(full_table, ce_sw, pe_sw)
    chrono = list(reversed(full_table))
    for setup, xkey in ((ce_set, "CE X"), (pe_set, "PE X")):
        if not setup or not setup.get("bar"):
            continue
        path, start = [], None
        col = "CE LTP" if xkey == "CE X" else "PE LTP"
        lowk = "CE low" if xkey == "CE X" else "PE low"
        for i, r in enumerate(chrono):
            path.append({"price": r.get(col), "low": r.get(lowk) or r.get(col), "time": r.get("time"), "x": r.get(xkey)})
            if str(r.get("time")) == str(setup.get("bar")):
                start = i
        if start is not None:
            res, when, pts = _walk(path, start, setup.get("sl"), setup.get("ltp"))
            setup["result"], setup["hit"], setup["points"] = res, when, pts
    st.markdown("**EMA setups**")
    st.dataframe(pd.DataFrame([
        ce_set or {"side": "CE Buy", "trigger": "waiting CE 9>21 and RSI>65"},
        pe_set or {"side": "PE Buy", "trigger": "waiting PE 9>21 and RSI<35"},
    ]), use_container_width=True, hide_index=True)
    if st.checkbox("Backtest spot 9/21 crosses", key="gs_bt_on"):
        nses = st.selectbox("Sessions", [30, 50, 100], key="gs_bt_n")
        if st.button("Run backtest", key="gs_bt_go") and fetch_fn:
            st.session_state["gs_fetch_days"] = int(nses * 1.6) + 5
            with st.spinner(f"Fetching {nses} sessions…"):
                tok = book.get("cash_tok") or book.get("idx_tok")
                exch = book.get("cash_exch") or book.get("idx_exch")
                dfb, _ = fetch_fn(str(tok), exch, TF_API.get(tf, "FIVE_MINUTE"), name, "gs_bt")
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
