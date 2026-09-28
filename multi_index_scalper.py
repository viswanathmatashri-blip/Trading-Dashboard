"""Multi Index Scalper — Colab Index CVD Desk logic (WATCH / CONFIRMED / setups)."""
from __future__ import annotations

import datetime as dt
import math
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

ORDER = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "GOLDM", "CRUDEOIL"]
LOT_SIZES = {
    "NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120,
    "SENSEX": 20, "GOLDM": 100, "CRUDEOIL": 100,
}
STEP = {
    "NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25,
    "SENSEX": 100, "GOLDM": 100, "CRUDEOIL": 50,
}
MIN_SL_PCT = -3.0
RR_BLUE = 1.0
PIVOT_L, PIVOT_R = 5, 5
TF_MIN = {
    "1 min": 1, "2 min": 2, "3 min": 3, "5 min": 5, "15 min": 15,
}
TF_API = {
    "1 min": "ONE_MINUTE",
    "2 min": "TWO_MINUTE",
    "3 min": "THREE_MINUTE",
    "5 min": "FIVE_MINUTE",
    "15 min": "FIFTEEN_MINUTE",
}
FILL = {
    "CONFIRMED SHORT": "background-color:#c62828;color:#fff",
    "CONFIRMED LONG": "background-color:#2e7d32;color:#fff",
    "WATCH SHORT": "background-color:#ffcdd2;color:#b71c1c",
    "WATCH LONG": "background-color:#c8e6c9;color:#1b5e20",
    "TREND LONG DENIED": "background-color:#ffcdd2;color:#b71c1c",
    "TREND SHORT DENIED": "background-color:#c8e6c9;color:#1b5e20",
}
PRIO = list(FILL.keys())


def loc_flow(o, h, l, c, v):
    rng = float(h) - float(l)
    if rng <= 0 or not v:
        return 0.0
    return float(v) * max(-1.0, min(1.0, 2.0 * ((float(c) - float(l)) / rng) - 1.0))


def signed_delta(o, c, v):
    if not v:
        return 0.0
    return float(v) if c > o else (-float(v) if c < o else 0.0)


def rows_from_df(df: pd.DataFrame) -> list:
    if df is None or getattr(df, "empty", True):
        return []
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time")
    out, cvd, pv, vv, p2 = [], 0.0, 0.0, 0.0, 0.0
    for _, r in d.iterrows():
        o = float(r.get("open", r["close"]))
        h = float(r.get("high", r["close"]))
        l = float(r.get("low", r["close"]))
        c = float(r["close"])
        v = float(r.get("volume", 0) or 0)
        sf = loc_flow(o, h, l, c, v)
        if sf == 0 and v:
            sf = signed_delta(o, c, v)
        cvd += sf
        pv += c * v
        vv += v
        p2 += c * c * v
        vwap = pv / vv if vv else c
        sig = math.sqrt(max(p2 / vv - vwap * vwap, 0)) if vv else 0.0
        ts = pd.Timestamp(r["time"]).strftime("%Y-%m-%d %H:%M:%S")
        out.append({
            "time": ts, "price": round(c, 2), "open": round(o, 2),
            "high": round(h, 2), "low": round(l, 2), "volume": int(v),
            "CVD": round(cvd, 2), "VWAP": round(vwap, 2),
            "VWAP_1.5σ_up": round(vwap + 1.5 * sig, 2),
            "VWAP_1.5σ_dn": round(vwap - 1.5 * sig, 2),
            "source": "HIST_1m",
        })
    return out


def _tf_minutes():
    lab = str(st.session_state.get("mis_tf") or "1 min")
    for n in (15, 5, 3, 2, 1):
        if str(n) in lab:
            return n
    return 1


def _ist_minute():
    step = _tf_minutes()
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("Asia/Kolkata")).replace(second=0, microsecond=0)
    except Exception:
        now = dt.datetime.now().replace(second=0, microsecond=0)
    minute = (now.minute // step) * step
    now = now.replace(minute=minute)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def _rebuild_last_from_prev(rows, t, o, h, l, c, v):
    """Rewrite or append only the forming minute. Prior bars stay as cached."""
    prev_cvd = float(rows[-1]["CVD"]) if rows and str(rows[-1].get("time")) != str(t) else (
        float(rows[-2]["CVD"]) if len(rows) >= 2 else 0.0
    )
    if rows and str(rows[-1].get("time")) == str(t):
        # CVD of previous closed bar
        prev_cvd = float(rows[-2]["CVD"]) if len(rows) >= 2 else 0.0
    sf = loc_flow(o, h, l, c, v)
    if sf == 0 and v:
        sf = signed_delta(o, c, v)
    cvd = prev_cvd + sf
    row = {
        "time": t, "price": round(c, 2), "open": round(o, 2),
        "high": round(h, 2), "low": round(l, 2), "volume": int(max(v, 0)),
        "CVD": round(cvd, 2), "VWAP": rows[-1].get("VWAP", c) if rows else round(c, 2),
        "source": "LIVE_1m",
    }
    if rows and str(rows[-1].get("time")) == str(t):
        # keep session VWAP from last closed if present
        if len(rows) >= 2:
            row["VWAP"] = rows[-2].get("VWAP", row["VWAP"])
        rows[-1] = row
    else:
        if rows:
            row["VWAP"] = rows[-1].get("VWAP", row["VWAP"])
        rows.append(row)
    if len(rows) > 1600:
        del rows[:-1500]
    return rows


def apply_live_tape(book, side, price, day_vol=None, ltq=None):
    """Only the current IST minute of THIS product. Closed sessions are not stamped."""
    if price is None:
        return book
    nm = book.get("name") or ""
    if nm and not session_open(nm):
        return book
    try:
        px = float(price)
    except Exception:
        return book
    key_rows = {"idx": "idx_rows", "ce": "ce_rows", "pe": "pe_rows"}[side]
    key_sw = {"idx": "idx_sw", "ce": "ce_sw", "pe": "pe_sw"}[side]
    key_form = f"form_{side}"
    key_dv = f"dayvol_{side}"
    rows = list(book.get(key_rows) or [])
    t = _ist_minute()
    dv = 0.0
    last_dv = book.get(key_dv)
    try:
        if day_vol is not None:
            d = float(day_vol)
            if last_dv is not None and d >= last_dv:
                dv = d - last_dv
            elif ltq:
                dv = float(ltq)
            book[key_dv] = d
        elif ltq:
            dv = float(ltq)
    except Exception:
        dv = float(ltq or 0) if ltq else 0.0
    form = book.get(key_form)
    if form is None or form.get("t") != t:
        if form is not None:
            rows = _rebuild_last_from_prev(
                rows, form["t"], form["o"], form["h"], form["l"], form["c"], form["v"]
            )
        form = {"t": t, "o": px, "h": px, "l": px, "c": px, "v": max(dv, 0.0)}
    else:
        form["h"] = max(form["h"], px)
        form["l"] = min(form["l"], px)
        form["c"] = px
        form["v"] += max(dv, 0.0)
    rows = _rebuild_last_from_prev(rows, form["t"], form["o"], form["h"], form["l"], form["c"], form["v"])
    book[key_form] = form
    book[key_rows] = rows
    book[key_sw] = rebuild_swings(rows)
    if side == "idx":
        book["fut"] = px
        book["spot"] = book.get("spot") or px
    return book


def live_apply_quotes(books, enabled, quotes: dict):
    """quotes: token -> {ltp, volume, ltq}"""
    if not quotes:
        return books
    for name, book in list(books.items()):
        if not enabled.get(name):
            continue
        if not session_open(name):
            continue
        try:
            cash = str(book.get("cash_tok") or "")
            if cash and cash in quotes:
                try:
                    book["spot"] = float(quotes[cash].get("ltp"))
                except Exception:
                    pass
            tok = str(book.get("idx_tok") or "")
            if tok and tok in quotes:
                q = quotes[tok]
                apply_live_tape(book, "idx", q.get("ltp"), q.get("volume"), q.get("ltq"))
                if not cash:
                    book["spot"] = book.get("fut") or book.get("spot")
            ctok = str(book.get("ce_tok") or "")
            if ctok and ctok in quotes:
                q = quotes[ctok]
                apply_live_tape(book, "ce", q.get("ltp"), q.get("volume"), q.get("ltq"))
            ptok = str(book.get("pe_tok") or "")
            if ptok and ptok in quotes:
                q = quotes[ptok]
                apply_live_tape(book, "pe", q.get("ltp"), q.get("volume"), q.get("ltq"))
        except Exception:
            continue
    return books


def rebuild_swings(rows: list) -> list:
    swings, seen = [], set()
    n = len(rows)
    if n < 9:
        return swings
    L = PIVOT_L if n >= 40 else 3
    R = PIVOT_R if n >= 40 else 3
    px = [r["price"] for r in rows]

    def last(side):
        for s in reversed(swings):
            if s["side"] == side:
                return s
        return None

    def add(i, side):
        if (i, side) in seen:
            return
        seen.add((i, side))
        prev = last(side)
        rec = {
            "i": i, "timestamp": rows[i]["time"], "session": rows[i]["time"][:10],
            "side": side, "price": rows[i]["price"], "CVD": rows[i]["CVD"],
            "label": "", "cvd_label": "", "div": "",
        }
        if prev:
            if side == "SH":
                rec["label"] = "HH" if rec["price"] > prev["price"] else ("LH" if rec["price"] < prev["price"] else "EH")
                rec["cvd_label"] = "HH" if rec["CVD"] > prev["CVD"] else ("LH" if rec["CVD"] < prev["CVD"] else "EH")
                if rec["label"] == "HH" and rec["cvd_label"] == "LH":
                    rec["div"] = "Bearish"
            else:
                rec["label"] = "LL" if rec["price"] < prev["price"] else ("HL" if rec["price"] > prev["price"] else "EL")
                rec["cvd_label"] = "LL" if rec["CVD"] < prev["CVD"] else ("HL" if rec["CVD"] > prev["CVD"] else "EL")
                if rec["label"] == "LL" and rec["cvd_label"] == "HL":
                    rec["div"] = "Bullish"
        swings.append(rec)

    for i in range(L, n - R):
        w = px[i - L:i + R + 1]
        mid = px[i]
        if mid == max(w) and w.count(mid) == 1:
            add(i, "SH")
        if mid == min(w) and w.count(mid) == 1:
            add(i, "SL")
    return swings


def rdi_of(r):
    h, l, c = r.get("high"), r.get("low"), r.get("price")
    if None in (h, l, c):
        return None
    rng = h - l
    return 0.0 if rng <= 0 else max(-1.0, min(1.0, 2.0 * ((c - l) / rng) - 1.0))


def disp_of(r):
    o, h, l, c = r.get("open"), r.get("high"), r.get("low"), r.get("price")
    if None in (o, h, l, c):
        return None
    rng = h - l
    return 0.0 if rng <= 0 else abs(c - o) / rng


def flow_tag(rdi, disp, status):
    if rdi is None or disp is None:
        return "—"
    ar = abs(rdi)
    at = any(x in (status or "") for x in ("WATCH", "SWING", "CONFIRMED"))
    if ar >= 0.40 and disp <= 0.20:
        return "ABSORPTION"
    if ar <= 0.10 and at:
        return "EXHAUSTION"
    if ar >= 0.35 and disp >= 0.85:
        return "ACCEL"
    return "—"


def annotate_bars(rows, swings):
    by_ts = {}
    for s in swings:
        by_ts.setdefault(str(s.get("timestamp")), []).append(s)
    last_sh = last_sl = None
    out = []
    for r in rows:
        ts = str(r.get("time"))
        px, cvd = r.get("price"), r.get("CVD")
        tags, reason = [], ""
        here = by_ts.get(ts) or []
        for s in here:
            if s.get("side") == "SH" and s.get("div") == "Bearish":
                tags.append("CONFIRMED SHORT")
            elif s.get("side") == "SL" and s.get("div") == "Bullish":
                tags.append("CONFIRMED LONG")
            elif s.get("side") == "SH":
                tags.append("SWING HIGH")
            elif s.get("side") == "SL":
                tags.append("SWING LOW")
        if last_sh and px is not None and cvd is not None:
            if px >= last_sh["price"] and cvd < last_sh["CVD"]:
                tags.append("WATCH SHORT")
                reason = f"px {px} >= SH {last_sh['price']} @ {last_sh['timestamp']} CVD {cvd} < {last_sh['CVD']}"
            elif px >= last_sh["price"] and cvd >= last_sh["CVD"]:
                tags.append("TREND SHORT DENIED")
        if last_sl and px is not None and cvd is not None:
            if px <= last_sl["price"] and cvd > last_sl["CVD"]:
                tags.append("WATCH LONG")
                reason = (reason + " | " if reason else "") + (
                    f"px {px} <= SL {last_sl['price']} @ {last_sl['timestamp']} CVD {cvd} > {last_sl['CVD']}"
                )
            elif px <= last_sl["price"] and cvd <= last_sl["CVD"]:
                tags.append("TREND LONG DENIED")
        status = " | ".join(tags) if tags else "—"
        rdi, disp = rdi_of(r), disp_of(r)
        row = dict(r)
        row["status"] = status
        row["watch_reason"] = reason
        row["RDI"] = None if rdi is None else round(rdi, 3)
        row["Disp"] = None if disp is None else round(disp, 3)
        row["flow"] = flow_tag(rdi, disp, status)
        out.append(row)
        for s in here:
            if s.get("side") == "SH":
                last_sh = s
            elif s.get("side") == "SL":
                last_sl = s
    return out


MCX_NAMES = {"GOLDM", "CRUDEOIL"}
LONG_BITS = ("WATCH LONG", "TREND SHORT DENIED", "CONFIRMED LONG")
SHORT_BITS = ("WATCH SHORT", "TREND LONG DENIED", "CONFIRMED SHORT")


def _ist_now():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return dt.datetime.now()


def session_open(name: str) -> bool:
    now = _ist_now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    if name in MCX_NAMES:
        return dt.time(9, 0) <= t <= dt.time(23, 30)
    return dt.time(9, 15) <= t <= dt.time(15, 30)


def _parse_bar(ts):
    if not ts:
        return None
    s = str(ts).replace("T", " ")[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def session_close_hm(name):
    return (23, 30) if name in MCX_NAMES else (15, 30)


def in_session_ts(name, ts) -> bool:
    t = _parse_bar(ts)
    if t is None:
        return False
    start = dt.time(9, 0) if name in MCX_NAMES else dt.time(9, 15)
    ch, cm = session_close_hm(name)
    return start <= t.time() <= dt.time(ch, cm)


def clip_session_rows(name, rows):
    return [r for r in (rows or []) if in_session_ts(name, r.get("time"))]


def clip_book_session(book):
    name = book.get("name") or ""
    if not name:
        return book
    for k in ("idx_rows", "ce_rows", "pe_rows", "ce_1m_rows", "pe_1m_rows"):
        book[k] = clip_session_rows(name, book.get(k))
    for side in ("idx", "ce", "pe"):
        form = book.get(f"form_{side}")
        if form and not in_session_ts(name, form.get("t")):
            book[f"form_{side}"] = None
    return book


def last_session_bar(name, rows, fallback):
    """Outside hours: last print at/before 15:30 (15:15 ok) / MCX 23:30 (23:00 ok)."""
    close_h, close_m = (23, 30) if name in MCX_NAMES else (15, 30)
    alt_h, alt_m = (23, 0) if name in MCX_NAMES else (15, 15)
    best = None
    for r in reversed(rows or []):
        ts = _parse_bar(r.get("time"))
        if ts is None:
            continue
        hm = (ts.hour, ts.minute)
        if hm <= (close_h, close_m):
            best = r.get("time")
            if hm in ((close_h, close_m), (alt_h, alt_m)):
                return str(best)[:19]
            break
    if best:
        return str(best)[:19]
    fb = _parse_bar(fallback)
    if fb and (fb.hour, fb.minute) <= (close_h, close_m):
        return str(fallback)[:19]
    return str(fallback or "")[:19]


def _vote_status(text, invert=False):
    t = str(text or "").upper()
    lg = any(b in t for b in LONG_BITS)
    sh = any(b in t for b in SHORT_BITS)
    if invert:
        lg, sh = sh, lg
    if lg and not sh:
        return "LONG"
    if sh and not lg:
        return "SHORT"
    return None


def _vote_flow(pack, invert=False):
    fl = str(pack.get("Flow") or "").upper()
    if fl in ("", "—", "NONE"):
        return None
    rdi = pack.get("RDI")
    try:
        rdi = float(rdi)
    except Exception:
        rdi = 0.0
    side = "LONG" if rdi > 0 else ("SHORT" if rdi < 0 else None)
    if fl == "ABSORPTION" and side:
        side = "SHORT" if side == "LONG" else "LONG"
    if invert and side:
        side = "SHORT" if side == "LONG" else "LONG"
    return side


def _row_at(ann, ts):
    if not ann:
        return {}
    ts = str(ts or "")
    for r in reversed(ann):
        if str(r.get("time") or "") == ts:
            return r
    before = [r for r in ann if str(r.get("time") or "") <= ts]
    return before[-1] if before else {}


def attach_bar_confluence(idx_ann, ce_ann, pe_ann):
    out = []
    for r in idx_ann or []:
        nr = dict(r)
        ce = _row_at(ce_ann, r.get("time"))
        pe = _row_at(pe_ann, r.get("time"))
        nr["Trend"] = confluence_label([
            _vote_status(r.get("status")),
            _vote_status(ce.get("status")),
            _vote_status(pe.get("status"), invert=True),
        ], "TREND")
        nr["Flow x/3"] = confluence_label([
            _vote_flow({"Flow": r.get("flow"), "RDI": r.get("RDI")}),
            _vote_flow({"Flow": ce.get("flow"), "RDI": ce.get("RDI")}),
            _vote_flow({"Flow": pe.get("flow"), "RDI": pe.get("RDI")}, invert=True),
        ], "FLOW")
        out.append(nr)
    return out


def confluence_label(votes, kind="TREND"):
    longs = sum(1 for v in votes if v == "LONG")
    shorts = sum(1 for v in votes if v == "SHORT")
    prefix = "LONG CONFLUENCE" if kind == "TREND" else "LONG FLOW"
    prefix_s = "SHORT CONFLUENCE" if kind == "TREND" else "SHORT FLOW"
    if longs and shorts:
        return f"CONFLICTING {longs}L/{shorts}S"
    if longs:
        return f"{prefix} {longs}/3"
    if shorts:
        return f"{prefix_s} {shorts}/3"
    return "—"


def last_pack(rows, swings, ltp=None, symbol=""):
    ann = annotate_bars(rows or [], swings or [])
    last = ann[-1] if ann else {}
    return {
        "WATCH": last.get("status") or "—",
        "Flow": last.get("flow") or "—",
        "RDI": last.get("RDI") if last.get("RDI") is not None else "—",
        "Reason": last.get("watch_reason") or "",
        "Bar": last.get("time") or "",
        "LTP": ltp if ltp is not None else (last.get("price") if last else "—"),
        "ann": ann,
        "swings": swings or [],
        "symbol": symbol or "",
    }


def _tf_minutes():
    return int(TF_MIN.get(st.session_state.get("mis_tf") or "1 min", 1))


def confirm_ready_ts(pivot_ts):
    t = _parse_bar(pivot_ts)
    if not t:
        return None
    return t + dt.timedelta(minutes=PIVOT_R * _tf_minutes())


def is_live_confirm(pivot_ts, last_tape_ts):
    """True only if the 5-bar confirmation just completed (last 2 closed bars)."""
    due = confirm_ready_ts(pivot_ts)
    last = _parse_bar(last_tape_ts) or _ist_now().replace(tzinfo=None)
    if due is None or last is None:
        return False
    try:
        last = last.replace(tzinfo=None)
        due = due.replace(tzinfo=None)
    except Exception:
        pass
    window = dt.timedelta(minutes=_tf_minutes() * 2)
    return due <= last + dt.timedelta(seconds=30) and due >= last - window


def rdi_agrees(kind, bar) -> bool:
    """Reject WATCH/CONFIRMED if bar RDI fights the side."""
    try:
        rdi = bar.get("RDI")
        if rdi is None:
            return True
        rdi = float(rdi)
    except Exception:
        return True
    if "LONG" in str(kind):
        return rdi >= -0.10
    if "SHORT" in str(kind):
        return rdi <= 0.10
    return True


def all_watch(ann):
    out, seen = [], set()
    for r in ann or []:
        stt = str(r.get("status") or "")
        kind = None
        if "WATCH LONG" in stt:
            kind = "WATCH LONG"
        elif "WATCH SHORT" in stt:
            kind = "WATCH SHORT"
        if not kind:
            continue
        key = (kind, str(r.get("time") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, r))
    return out


def causal_watch_events(rows):
    """WATCH printable at each close using only bars up to that close (no future)."""
    rows = list(rows or [])
    out = []
    if len(rows) < 8:
        return out
    for i in range(7, len(rows)):
        prefix = rows[: i + 1]
        ann = annotate_bars(prefix, rebuild_swings(prefix))
        if not ann:
            continue
        last = ann[-1]
        stt = str(last.get("status") or "")
        kind = "WATCH LONG" if "WATCH LONG" in stt else ("WATCH SHORT" if "WATCH SHORT" in stt else None)
        if kind:
            out.append((kind, last))
    return out


def last_watch(ann):
    """Live signal: WATCH on the latest bar only (no 5-bar wait)."""
    if not ann:
        return None, None
    r = ann[-1]
    stt = str(r.get("status") or "")
    if "WATCH LONG" in stt:
        return "WATCH LONG", r
    if "WATCH SHORT" in stt:
        return "WATCH SHORT", r
    if len(ann) >= 2:
        r2 = ann[-2]
        stt2 = str(r2.get("status") or "")
        if "WATCH LONG" in stt2:
            return "WATCH LONG", r2
        if "WATCH SHORT" in stt2:
            return "WATCH SHORT", r2
    return None, None


def last_confirmed(ann):
    for r in reversed(ann or []):
        stt = r.get("status") or ""
        if "CONFIRMED LONG" in stt:
            return "CONFIRMED LONG", r
        if "CONFIRMED SHORT" in stt:
            return "CONFIRMED SHORT", r
    return None, None


def all_confirmed(ann):
    out = []
    seen = set()
    for r in reversed(ann or []):
        stt = str(r.get("status") or "")
        kind = None
        if "CONFIRMED LONG" in stt:
            kind = "CONFIRMED LONG"
        elif "CONFIRMED SHORT" in stt:
            kind = "CONFIRMED SHORT"
        if not kind:
            continue
        key = (kind, str(r.get("time") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, r))
    return out


def price_at(pack, ts):
    rows = pack.get("ann") or []
    for r in rows:
        if str(r.get("time")) == str(ts):
            return r.get("price")
    before = [r for r in rows if str(r.get("time")) <= str(ts)]
    return before[-1].get("price") if before else None


def nearest_sh_above(swings, ltp):
    above = [s for s in (swings or []) if s.get("side") == "SH" and s.get("price") is not None and s["price"] > ltp]
    return min(above, key=lambda s: s["price"]) if above else None


def target_for_long(pack, ts, ltp):
    sh = nearest_sh_above(pack.get("swings"), ltp)
    if sh and float(sh["price"]) > float(ltp):
        return float(sh["price"])
    highs = []
    for r in pack.get("ann") or []:
        if str(r.get("time") or "") < str(ts or ""):
            continue
        try:
            h = float(r.get("high") if r.get("high") is not None else r.get("price"))
        except Exception:
            continue
        if h > float(ltp):
            highs.append(h)
    return min(highs) if highs else None


def sl_below_ltp(pack, ts, ltp):
    rows = pack.get("ann") or []
    if not rows:
        return None
    i = None
    for k, r in enumerate(rows):
        if str(r.get("time")) == str(ts):
            i = k
            break
    if i is None:
        before = [k for k, r in enumerate(rows) if str(r.get("time")) <= str(ts)]
        i = before[-1] if before else len(rows) - 1
    for r in reversed(rows[: i + 1]):
        lo = r.get("low")
        if lo is None:
            continue
        try:
            lo = float(lo)
        except Exception:
            continue
        if lo < ltp:
            return lo
    return None


def apply_min_sl(ltp, slpx):
    px = float(ltp or 0)
    if px < 70:
        pct = -10.0
    elif px < 100:
        pct = -7.0
    else:
        pct = -5.0
    floor = ltp * (1.0 + pct / 100.0)
    if slpx is None:
        return round(floor, 2)
    return round(min(float(slpx), floor), 2)


def setup_accuracy(buy_pack, trigger_ts, slpx, target, now_ltp=None):
    """After trigger, which level prints first on the option tape."""
    if slpx is None and target is None:
        return "—"
    try:
        slpx = None if slpx is None else float(slpx)
        target = None if target is None else float(target)
    except Exception:
        return "—"
    ann = list(buy_pack.get("ann") or [])
    started = False
    for r in ann:
        ts = str(r.get("time") or "")
        if not started:
            if ts >= str(trigger_ts or ""):
                started = True
            else:
                continue
        if ts == str(trigger_ts or ""):
            continue
        try:
            hi = float(r.get("high") if r.get("high") is not None else r.get("price"))
            lo = float(r.get("low") if r.get("low") is not None else r.get("price"))
        except Exception:
            continue
        hit_sl = slpx is not None and lo <= slpx
        hit_tgt = target is not None and hi >= target
        if not hit_sl and not hit_tgt:
            continue
        t0 = _parse_bar(trigger_ts)
        t1 = _parse_bar(ts)
        elapsed = ""
        if t0 and t1:
            sec = max(int((t1 - t0).total_seconds()), 0)
            elapsed = f" +{sec // 60:02d}:{sec % 60:02d}"
        if hit_sl:
            return f"SL Hit{elapsed}"
        return f"Target Hit{elapsed}"
    try:
        live = float(now_ltp) if now_ltp is not None else float(buy_pack.get("LTP"))
    except Exception:
        live = None
    t0 = _parse_bar(trigger_ts)
    now = _ist_now()
    try:
        now_n = now.replace(tzinfo=None)
    except Exception:
        now_n = now
    if live is not None and t0 is not None and t0.date() == now_n.date():
        sec = max(int((now_n - t0).total_seconds()), 0)
        elapsed = f" +{sec // 60:02d}:{sec % 60:02d}"
        if slpx is not None and live <= slpx:
            return f"SL Hit{elapsed}"
        if target is not None and live >= target:
            return f"Target Hit{elapsed}"
    return "OPEN"


def setup_row(index, source, confirmed, buy_side, buy_pack, strike, oexp, lot, trigger_ts, max_loss):
    freeze = st.session_state.setdefault("_mis_setup_freeze", {})
    fk = f"{index}|{source}|{confirmed}|{trigger_ts}|{buy_side}|{strike}|pxv2"
    if fk in freeze:
        prev = dict(freeze[fk])
        if not in_session_ts(index, prev.get("Bar")):
            prev["Bar"] = trigger_ts or prev.get("Pivot") or prev.get("Bar")
        prev["Accuracy"] = setup_accuracy(
            dict(buy_pack, ann=buy_pack.get("_acc_ann") or buy_pack.get("ann")),
            trigger_ts, prev.get("SL") if prev.get("SL") != "—" else None,
            prev.get("Target") if prev.get("Target") != "—" else None,
            now_ltp=buy_pack.get("LTP"),
        )
        return prev

    ltp = buy_pack.get("_entry_px")
    if ltp is None:
        ltp = price_at(buy_pack, trigger_ts)
    if ltp is None:
        ltp = buy_pack.get("LTP")
    try:
        ltp = float(ltp)
    except Exception:
        return None
    slpx = apply_min_sl(ltp, sl_below_ltp(buy_pack, trigger_ts, ltp))
    target = target_for_long(buy_pack, trigger_ts, ltp)
    if target is None or float(target) <= ltp:
        return None
    tgt_pct = round((target - ltp) / ltp * 100, 2)
    if tgt_pct < 3.0:
        return None
    sl_pct = round((slpx - ltp) / ltp * 100, 2) if slpx and ltp else None
    risk = (ltp - slpx) if slpx is not None else None
    reward = (target - ltp) if target is not None else None
    if risk is None or risk <= 0 or lot <= 0:
        lots, rr, risk_lot = 0, None, None
    else:
        risk_lot = risk * lot
        lots = int(max_loss // risk_lot) if risk_lot > 0 else 0
        rr = round(reward / risk, 2) if reward and reward > 0 else None
    return {
        "Index": index, "Trigger": source, "Confirmed": confirmed,
        "Buy": f"{strike} {buy_side}" if strike else buy_side,
        "Opt exp": oexp or "—", "LTP": round(ltp, 2),
        "Target": target if target is not None else "—",
        "Target %": tgt_pct if tgt_pct is not None else "—",
        "SL": slpx if slpx is not None else "—",
        "SL %": sl_pct if sl_pct is not None else "—",
        "Lot Qty": lot, "Entry lots": lots, "Qty": lots * lot,
        "Risk/lot": round(risk_lot, 2) if risk_lot else "—",
        "R:R": rr if rr is not None else "—",
        "Accuracy": setup_accuracy(
            dict(buy_pack, ann=buy_pack.get("_acc_ann") or buy_pack.get("ann")),
            trigger_ts, slpx, target, now_ltp=buy_pack.get("LTP"),
        ),
        "Bar": trigger_ts or "",
        "Pivot": trigger_ts or "",
    }
    freeze[fk] = {k: v for k, v in row.items() if k != "Accuracy"}
    return row


def _send_mis_telegram(setups):
    def _sec(k, alt=""):
        try:
            if k in st.secrets:
                return str(st.secrets[k]).strip()
        except Exception:
            pass
        return (os.getenv(k) or os.getenv(alt) or "").strip()

    tok = _sec("TELE_BOTTOKEN", "TELEGRAM_BOT_TOKEN")
    chat = _sec("TELE_CHATID", "TELEGRAM_CHAT_ID")
    if not tok or not chat or not setups:
        return
    sent = st.session_state.setdefault("_mis_tg_sent", set())
    if not isinstance(sent, set):
        sent = set(sent)
        st.session_state["_mis_tg_sent"] = sent
    for row in setups:
        conf = str(row.get("Confirmed") or "")
        if not any(x in conf for x in ("CONFIRMED LONG", "CONFIRMED SHORT", "WATCH LONG", "WATCH SHORT")):
            continue
        key = f"{row.get('Index')}|{row.get('Trigger')}|{row.get('Bar')}|{row.get('Buy')}"
        if key in sent:
            continue
        tgt = row.get("Target")
        tgt_pct = row.get("Target %")
        sl = row.get("SL")
        sl_pct = row.get("SL %")
        msg = (
            f"{row.get('Index')}\n"
            f"{row.get('Trigger')}\n"
            f"{row.get('Buy')}\n"
            f"Buy : {row.get('LTP')}\n"
            f"Target : {tgt} ({tgt_pct}%)\n"
            f"SL -{sl} ({sl_pct}%)\n"
            f"Entry Lot Qty = {row.get('Qty')}\n"
            f"R:R = {row.get('R:R')}"
        )
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": chat, "text": msg, "disable_web_page_preview": True},
                timeout=8,
            )
            if r.ok:
                sent.add(key)
        except Exception:
            continue
    st.session_state["_mis_tg_sent"] = sent


def watch_css(val):
    text = "" if val is None else str(val)
    for k in PRIO:
        if k in text:
            return FILL[k]
    return ""


def confluence_css(val):
    text = str(val or "").upper()
    if text.startswith("LONG"):
        return "background-color:#2e7d32;color:#fff"
    if text.startswith("SHORT"):
        return "background-color:#c62828;color:#fff"
    if text.startswith("CONFLICT"):
        return "background-color:#00838f;color:#fff"
    return ""


def style_watch(df):
    sty = df.style
    use = [c for c in df.columns if "WATCH" in str(c).upper()]
    if use:
        sty = sty.map(watch_css, subset=use)
    extra = [c for c in ("Trend", "Flow x/3") if c in df.columns]
    if extra:
        sty = sty.map(confluence_css, subset=extra)
    gold = [c for c in ("Idx WATCH", "CE WATCH", "PE WATCH") if c in df.columns]
    head_css = []
    for i, c in enumerate(df.columns):
        if c in gold:
            head_css.append({
                "selector": f"th.col_heading.level0.col{i}",
                "props": [
                    ("background-color", "#C9A227"),
                    ("color", "#111111"),
                    ("font-weight", "800"),
                ],
            })
    if head_css:
        sty = sty.set_table_styles(head_css, overwrite=False)
    return sty.hide(axis="index")


def _cell_css(col, val):
    if col in ("Trend", "Flow x/3"):
        return confluence_css(val)
    if "WATCH" in str(col).upper():
        return watch_css(val)
    return ""


def watch_log_html(df: pd.DataFrame) -> str:
    gold = {"Idx WATCH", "CE WATCH", "PE WATCH"}
    rows = []
    rows.append("<thead><tr>")
    for c in df.columns:
        if c in gold:
            rows.append(
                f'<th style="background:#C9A227;color:#111;font-weight:800;padding:6px 8px;'
                f'white-space:nowrap;position:sticky;top:0;z-index:2;border-bottom:1px solid #333;">{c}</th>'
            )
        else:
            rows.append(
                f'<th style="background:#16181d;color:#c8ccd4;font-weight:700;padding:6px 8px;'
                f'white-space:nowrap;position:sticky;top:0;z-index:2;border-bottom:1px solid #333;">{c}</th>'
            )
    rows.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        rows.append("<tr>")
        for c in df.columns:
            v = r[c]
            extra = _cell_css(c, v)
            if c == "Index":
                rows.append(
                    f'<td style="padding:5px 8px;white-space:nowrap;">{v}</td>'
                )
            else:
                rows.append(
                    f'<td style="padding:5px 8px;white-space:nowrap;{extra}">{"" if v is None else v}</td>'
                )
        rows.append("</tr>")
    rows.append("</tbody>")
    return (
        '<div style="overflow:auto;width:100%;border:1px solid #2a2d33;border-radius:6px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:12px;font-family:ui-sans-serif,system-ui;'
        'color:#e0e0e0;background:#0E1117;">'
        + "".join(rows)
        + "</table></div>"
    )


def style_setups(df):
    def row_style(row):
        css = [""] * len(row)
        acc = str(row.get("Accuracy") or "")
        if "Accuracy" in row.index:
            if acc.startswith("Target Hit"):
                css[row.index.get_loc("Accuracy")] = "background-color:#2e7d32;color:#fff"
            elif acc.startswith("SL Hit"):
                css[row.index.get_loc("Accuracy")] = "background-color:#c62828;color:#fff"
        for col in ("Trend", "Flow"):
            if col in row.index:
                fill = confluence_css(row.get(col))
                if fill:
                    css[row.index.get_loc(col)] = fill
        try:
            rr = float(row["R:R"])
        except Exception:
            rr = 0
        if rr <= RR_BLUE:
            return css
        buy = str(row.get("Buy") or "")
        if "Buy" in row.index:
            if buy.endswith("PE") or " PE" in buy:
                css[row.index.get_loc("Buy")] = "background-color:#c62828;color:#fff"
            elif buy.endswith("CE") or " CE" in buy:
                css[row.index.get_loc("Buy")] = "background-color:#2e7d32;color:#fff"
        for col in ("Target %", "SL %", "Entry lots"):
            if col in row.index:
                css[row.index.get_loc(col)] = "background-color:#1565c0;color:#fff"
        return css
    return df.style.apply(row_style, axis=1).hide(axis="index")


def _atm_from_master(df_master, name, spot, fo):
    if df_master is None or getattr(df_master, "empty", True) or not spot:
        return None
    today = dt.datetime.now().date()
    step = STEP.get(name, 50)
    atm = int(round(float(spot) / step) * step)
    names = ["GOLDM", "GOLD"] if name == "GOLDM" else [name]
    d = df_master.copy()
    d["name_u"] = d["name"].astype(str).str.upper()
    d = d[d["name_u"].isin([x.upper() for x in names])]
    if "exch_seg" in d.columns:
        d = d[d["exch_seg"].astype(str) == str(fo)]
    if "instrumenttype" in d.columns:
        d = d[d["instrumenttype"].astype(str).isin(["OPTIDX", "OPTSTK", "OPTFUT", "OPTCOM"])]
    ce = pe = exp = None
    rows = []
    for _, r in d.iterrows():
        try:
            raw = str(r.get("expiry") or "")
            e = None
            for fmt in ("%d%b%Y", "%d%b%y", "%Y-%m-%d"):
                try:
                    e = dt.datetime.strptime(raw, fmt).date()
                    break
                except Exception:
                    pass
            if not e or e < today:
                continue
            k = float(r.get("strike") or 0)
            if k > 10000:
                k = k / 100.0
            if abs(k - atm) > 0.01:
                continue
            sym = str(r.get("symbol") or "")
            side = "CE" if sym.endswith("CE") else ("PE" if sym.endswith("PE") else None)
            if not side:
                continue
            rows.append((e, side, str(r.get("token")), sym, fo))
        except Exception:
            continue
    by = {}
    for e, side, tok, sym, ex in rows:
        by.setdefault(e, {})[side] = (tok, sym, ex)
    for e in sorted(by):
        if "CE" in by[e] and "PE" in by[e]:
            return atm, str(e), by[e]["CE"], by[e]["PE"]
    return None


def seed_book(name, fetch_fn, df_master, token_map, fut_fn=None):
    idx_tok, cash, fo = token_map.get(name, ("", "NSE", "NFO"))
    book = st.session_state.setdefault("_mis_books", {}).get(name) or {
        "name": name, "idx_rows": [], "idx_sw": [], "ce_rows": [], "ce_sw": [],
        "pe_rows": [], "pe_sw": [], "spot": None, "fut": None, "atm": None, "exp": None,
        "ce_tok": None, "pe_tok": None, "ce_sym": "", "pe_sym": "",
        "idx_tok": None, "idx_exch": fo, "err": "",
    }
    tok = None
    if fut_fn:
        try:
            tok, _ = fut_fn(df_master, name, fo)
        except Exception:
            tok = None
    token = tok or idx_tok
    exch = fo if tok else cash
    book["idx_tok"] = token
    book["idx_exch"] = exch
    interval = TF_API.get(st.session_state.get("mis_tf") or "1 min", "ONE_MINUTE")
    df, _ = fetch_fn(token, exch, interval, name, "mis_idx")
    rows = rows_from_df(df)
    book["idx_rows"] = rows
    book["idx_sw"] = rebuild_swings(rows)
    if rows:
        book["fut"] = rows[-1]["price"]
        book["spot"] = rows[-1]["price"]
    book["cash_tok"] = str(idx_tok) if idx_tok else ""
    book["cash_exch"] = cash
    if idx_tok:
        try:
            dspot, _ = fetch_fn(str(idx_tok), cash, interval, name, "mis_spot")
            srows = rows_from_df(dspot)
            if srows:
                book["spot"] = srows[-1]["price"]
        except Exception:
            pass
    atm = _atm_from_master(df_master, name, book.get("spot"), fo)
    if atm:
        strike, exp, ce, pe = atm
        book["atm"], book["exp"] = strike, exp
        book["ce_tok"], book["ce_sym"] = ce[0], ce[1]
        book["pe_tok"], book["pe_sym"] = pe[0], pe[1]
        dce, _ = fetch_fn(ce[0], ce[2], interval, name, "mis_ce")
        dpe, _ = fetch_fn(pe[0], pe[2], interval, name, "mis_pe")
        book["ce_rows"] = rows_from_df(dce)
        book["pe_rows"] = rows_from_df(dpe)
        book["ce_sw"] = rebuild_swings(book["ce_rows"])
        book["pe_sw"] = rebuild_swings(book["pe_rows"])
        if interval == "ONE_MINUTE":
            book["ce_1m_rows"] = book["ce_rows"]
            book["pe_1m_rows"] = book["pe_rows"]
        else:
            try:
                dce1, _ = fetch_fn(ce[0], ce[2], "ONE_MINUTE", name, "mis_ce1m")
                dpe1, _ = fetch_fn(pe[0], pe[2], "ONE_MINUTE", name, "mis_pe1m")
                book["ce_1m_rows"] = rows_from_df(dce1)
                book["pe_1m_rows"] = rows_from_df(dpe1)
            except Exception:
                book["ce_1m_rows"] = book["ce_rows"]
                book["pe_1m_rows"] = book["pe_rows"]
    for side, rk in (("idx", "idx_rows"), ("ce", "ce_rows"), ("pe", "pe_rows")):
        rs = book.get(rk) or []
        if rs:
            last = rs[-1]
            book[f"form_{side}"] = {
                "t": last["time"][:19] if len(str(last.get("time") or "")) >= 16 else last.get("time"),
                "o": last["open"], "h": last["high"], "l": last["low"],
                "c": last["price"], "v": last.get("volume") or 0,
            }
    books = st.session_state.get("_mis_books") or {}
    books[name] = book
    st.session_state["_mis_books"] = books
    return book


def render_multi_index_scalper(fetch_fn, df_master, token_map, get_client, fut_fn=None, quote_fn=None):
    a_lab = "ON" if st.session_state.get("_smart_api_obj") and not st.session_state.get("_smart_api_err") else (
        "RATE LIMIT" if st.session_state.get("_angel_rl_ts") else ("DOWN" if st.session_state.get("_smart_api_err") else "IDLE")
    )
    a_col = {"ON": "#00E676", "RATE LIMIT": "#FFB300", "DOWN": "#FF5252"}.get(a_lab, "#90A4AE")
    if st.session_state.get("_ab_session"):
        b_lab, b_col = "ON", "#00E676"
    elif st.session_state.get("_ab_err"):
        b_lab, b_col = "DOWN", "#FF5252"
    else:
        b_lab, b_col = "STANDBY", "#FFB300"
    last = st.session_state.get("_last_broker") or "—"
    h1, hL, h2, hTf, h3 = st.columns([1.7, 2.0, 0.85, 0.22, 0.55])
    with h1:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
            f"<span style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Multi Index Scalper</span>"
            f"<span style='color:{a_col};font-size:11px;font-weight:700;'>● SmartAPI {a_lab}</span>"
            f"<span style='color:{b_col};font-size:11px;font-weight:700;'>● AliceBlue {b_lab}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with hL:
        st.markdown(
            "<div style='font-size:11px;color:#90A4AE;line-height:1.25;'>"
            "<b style='color:#B0BEC5;'>Status</b> WATCH · CONFIRMED · TREND DENIED &nbsp;|&nbsp; "
            "<b style='color:#B0BEC5;'>Flow</b> ABSORB |RDI|≥0.40 Disp≤0.20 · EXH |RDI|≤0.10 · ACCEL |RDI|≥0.35 Disp≥0.85"
            "</div>",
            unsafe_allow_html=True,
        )
    with h2:
        _mis_auto = st.checkbox(
            "Auto-Refresh 5s",
            value=bool(st.session_state.get("enable_main_refresh")),
            key="cb_mis_refresh_main",
        )
    with hTf:
        st.markdown("<div style='padding-top:8px;font-size:12px;color:#B0BEC5;font-weight:700;'>TF</div>", unsafe_allow_html=True)
    with h3:
        _tfs = ["1 min", "2 min", "3 min", "5 min", "15 min"]
        _cur = st.session_state.get("mis_tf") or "1 min"
        if _cur not in _tfs:
            _cur = "1 min"
        _tf = st.selectbox("TF", _tfs, index=_tfs.index(_cur), key="mis_tf_sel", label_visibility="collapsed")
        if _tf != st.session_state.get("mis_tf"):
            st.session_state["mis_tf"] = _tf
            st.session_state["mis_need_seed"] = True
            st.session_state["_mis_books"] = {}
            st.rerun()
        st.session_state["mis_tf"] = _tf
    if _mis_auto != bool(st.session_state.get("enable_main_refresh")):
        st.session_state["enable_main_refresh"] = _mis_auto
        st.rerun()
    _live = st.session_state.get("_mis_live_ts")
    if _live:
        st.caption(f"Live tape {_live} · cached history kept · only current 1-min bar updates")
    if st.session_state.get("_mis_live_err"):
        st.caption(f"Quote tick: {st.session_state.get('_mis_live_err')}")
    enabled = dict(st.session_state.get("mis_enabled") or {n: True for n in ORDER})
    max_loss = float(st.session_state.get("mis_max_loss") or 2000)
    books = st.session_state.get("_mis_books") or {}
    if st.session_state.get("mis_need_seed"):
        names = [n for n in ORDER if enabled.get(n)]
        _tf_lab = st.session_state.get("mis_tf") or "1 min"
        bar = st.progress(0.0, text=f"Seeding {_tf_lab} tapes…")
        for i, n in enumerate(names):
            try:
                seed_book(n, fetch_fn, df_master, token_map, fut_fn=fut_fn)
            except Exception as e:
                books = st.session_state.get("_mis_books") or {}
                books[n] = {"name": n, "err": str(e), "idx_rows": []}
                st.session_state["_mis_books"] = books
            bar.progress((i + 1) / max(len(names), 1), text=f"Seeded {n}")
        bar.empty()
        st.session_state["mis_need_seed"] = False
        books = st.session_state.get("_mis_books") or {}

    live_ok = any(session_open(n) for n in ORDER if enabled.get(n))
    if st.session_state.get("enable_main_refresh") and books and quote_fn and live_ok:
        try:
            toks = []
            for n, book in books.items():
                if not enabled.get(n) or not session_open(n):
                    continue
                for k, exch_k in (("idx_tok", "idx_exch"), ("cash_tok", "cash_exch"), ("ce_tok", "idx_exch"), ("pe_tok", "idx_exch")):
                    tok = book.get(k)
                    if tok:
                        toks.append((str(book.get(exch_k) or "NFO"), str(tok)))
            quotes = quote_fn(toks) or {}
            books = live_apply_quotes(books, enabled, quotes)
            st.session_state["_mis_books"] = books
            st.session_state["_mis_live_ts"] = dt.datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            st.session_state["_mis_live_err"] = str(e)[:180]

    for _n, _b in list(books.items()):
        if isinstance(_b, dict):
            books[_n] = clip_book_session(_b)
    st.session_state["_mis_books"] = books

    summary, live_alerts = [], []
    journal = st.session_state.setdefault("_mis_live_setups", [])
    packs = {}
    for name in ORDER:
        if not enabled.get(name):
            continue
        book = books.get(name)
        if not book:
            continue
        idx = last_pack(book.get("idx_rows"), book.get("idx_sw"), book.get("fut"))
        ce = last_pack(book.get("ce_rows"), book.get("ce_sw"), None, book.get("ce_sym"))
        pe = last_pack(book.get("pe_rows"), book.get("pe_sw"), None, book.get("pe_sym"))
        idx["ann"] = attach_bar_confluence(idx.get("ann"), ce.get("ann"), pe.get("ann"))
        t_lab = confluence_label([
            _vote_status(idx["WATCH"], invert=False),
            _vote_status(ce["WATCH"], invert=False),
            _vote_status(pe["WATCH"], invert=True),
        ], "TREND")
        f_lab = confluence_label([
            _vote_flow(idx, invert=False),
            _vote_flow(ce, invert=False),
            _vote_flow(pe, invert=True),
        ], "FLOW")
        packs[name] = (book, idx, ce, pe, t_lab, f_lab)
        strike = book.get("atm") or "—"
        oexp = book.get("exp") or "—"
        t_votes = [
            _vote_status(idx["WATCH"], invert=False),
            _vote_status(ce["WATCH"], invert=False),
            _vote_status(pe["WATCH"], invert=True),
        ]
        f_votes = [
            _vote_flow(idx, invert=False),
            _vote_flow(ce, invert=False),
            _vote_flow(pe, invert=True),
        ]
        bar_ts = idx["Bar"]
        if not session_open(name):
            bar_ts = last_session_bar(name, book.get("idx_rows") or [], bar_ts)
        summary.append({
            "Index": name,
            "Spot": book.get("spot") if book.get("spot") is not None else "—",
            "Fut LTP": book.get("fut") if book.get("fut") is not None else "—",
            "ATM": strike, "Opt exp": oexp,
            "Idx WATCH": idx["WATCH"], "Idx Flow": idx["Flow"], "Idx Reason": idx["Reason"],
            "CE WATCH": ce["WATCH"], "CE Flow": ce["Flow"], "CE Reason": ce["Reason"], "CE LTP": ce["LTP"],
            "PE WATCH": pe["WATCH"], "PE Flow": pe["Flow"], "PE Reason": pe["Reason"], "PE LTP": pe["LTP"],
            "Trend": confluence_label(t_votes, "TREND"),
            "Flow x/3": confluence_label(f_votes, "FLOW"),
            "Bar": bar_ts,
        })
        lot = LOT_SIZES.get(name, 1)
        for source, pack, rule in (
            ("FUT", idx, {"CONFIRMED LONG": "CE", "CONFIRMED SHORT": "PE"}),
            ("CE", ce, {"CONFIRMED LONG": "CE", "CONFIRMED SHORT": "PE"}),
            ("PE", pe, {"CONFIRMED LONG": "PE", "CONFIRMED SHORT": "CE"}),
        ):
            rule_w = {
                "WATCH LONG": rule["CONFIRMED LONG"],
                "WATCH SHORT": rule["CONFIRMED SHORT"],
                "CONFIRMED LONG": rule["CONFIRMED LONG"],
                "CONFIRMED SHORT": rule["CONFIRMED SHORT"],
            }
            raw = book.get("idx_rows") if source == "FUT" else (book.get("ce_rows") if source == "CE" else book.get("pe_rows"))
            events = [(k, b) for k, b in causal_watch_events(raw) if in_session_ts(name, (b or {}).get("time"))]
            live_kind, live_bar = last_watch(pack.get("ann"))
            for kind, bar in events:
                buy = rule_w[kind]
                buy_pack = ce if buy == "CE" else pe
                ts = bar.get("time") if bar else ""
                if not in_session_ts(name, ts):
                    continue
                if source == buy and bar.get("price") is not None:
                    buy_pack = dict(buy_pack)
                    buy_pack["_entry_px"] = bar.get("price")
                fine = book.get("ce_1m_rows") if buy == "CE" else book.get("pe_1m_rows")
                if fine:
                    buy_pack = dict(buy_pack)
                    buy_pack["_acc_ann"] = [
                        {"time": r.get("time"), "high": r.get("high"), "low": r.get("low"), "price": r.get("price")}
                        for r in fine
                    ]
                hit = _row_at(idx.get("ann") or [], ts)
                t_at = str((hit or {}).get("Trend") or t_lab or "—")
                f_at = str((hit or {}).get("Flow x/3") or f_lab or "—")
                if t_at in ("", "—"):
                    t_at = "LONG CONFLUENCE 1/3" if "LONG" in kind else "SHORT CONFLUENCE 1/3"
                tu2, fu2 = t_at.upper(), f_at.upper()
                row = setup_row(name, f"{source} {kind}", kind, buy, buy_pack, strike, oexp, lot, ts, max_loss)
                if row:
                    chk = row.get("Pivot") or row.get("Bar") or ts
                    if not in_session_ts(name, chk):
                        continue
                    row["Trend"] = t_at
                    row["Flow"] = f_at
                    row["_jk"] = f"{name}|{source}|{kind}|{ts}|{buy}"
                    have = {str(r.get("_jk")) for r in journal}
                    if row["_jk"] not in have:
                        journal.append(row)
                        if session_open(name):
                            live_alerts.append(row)
                    else:
                        for r in journal:
                            if r.get("_jk") == row["_jk"]:
                                r["Accuracy"] = row.get("Accuracy")
                                break

    st.session_state["_mis_live_setups"] = journal
    setups = [dict(r) for r in journal]
    for r in setups:
        r.pop("_jk", None)
    _send_mis_telegram(live_alerts)

    n_watch = max(len(summary), 1)
    watch_h = min(38 * (n_watch + 1) + 20, 320)
    setup_h = 38 * 8 + 20
    st.markdown(
        f"""
<style>
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > .main,
  .main .block-container {{
    overflow: visible !important;
  }}
  .main .block-container {{ padding-top: 0.4rem; max-width: 100%; }}
  .mis-pin {{
    position: sticky;
    top: 0;
    z-index: 200;
    background: #0E1117;
    border-bottom: 1px solid #2a2d33;
    padding: 2px 0 8px 0;
    box-shadow: 0 10px 18px rgba(0,0,0,0.5);
  }}
</style>
<div class="mis-pin">
""",
        unsafe_allow_html=True,
    )
    def _take_index_click(ev, df):
        try:
            rows = list((ev.selection or {}).get("rows") or ev.selection.rows)
        except Exception:
            try:
                rows = list(ev.selection.rows)
            except Exception:
                rows = []
        if rows and df is not None and "Index" in df.columns:
            try:
                st.session_state["mis_jump"] = str(df.iloc[int(rows[0])]["Index"])
            except Exception:
                pass

    _upd = _ist_now().strftime("%d-%m-%Y %H:%M:%S")
    st.caption(f"WATCH LOG    last updated {_upd}")
    if summary:
        wdf = pd.DataFrame(summary)
        st.markdown(watch_log_html(wdf), unsafe_allow_html=True)
    else:
        st.info("Seed 1-min tapes in the sidebar.")
    st.caption(f"SETUPS · causal WATCH at each bar close (no future bars)    last updated {_upd}")
    if setups:
        sdf = pd.DataFrame(setups)
        if "Bar" in sdf.columns:
            sdf = sdf.sort_values("Bar", ascending=False).reset_index(drop=True)
        front = [c for c in ("Index", "Trigger", "Confirmed", "Trend", "Flow") if c in sdf.columns]
        rest = [c for c in sdf.columns if c not in front]
        sdf = sdf[front + rest]
        try:
            evs = st.dataframe(
                style_setups(sdf),
                use_container_width=True,
                height=setup_h,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="mis_setup_pick",
            )
            _take_index_click(evs, sdf)
        except TypeError:
            st.dataframe(style_setups(sdf), use_container_width=True, height=setup_h, hide_index=True)
    else:
        st.write("No live WATCH journal yet — Auto-Refresh during session, or last-bar WATCH at the close.")
    st.markdown("</div>", unsafe_allow_html=True)

    def draw_tape(title, pack, extra_cols=None):
        st.markdown(f"**{title}**  ·  last updated {_upd}")
        if pack["ann"]:
            df = pd.DataFrame(pack["ann"][::-1])
            cols = [c for c in ["time", "status", "flow", "Trend", "Flow x/3", "watch_reason", "price", "CVD", "RDI", "Disp", "volume", "VWAP"] if c in df.columns]
            if extra_cols:
                cols = extra_cols + [c for c in cols if c not in extra_cols]
            sty = df[cols].style
            extra = [c for c in ("Trend", "Flow x/3") if c in df.columns]
            if extra:
                sty = sty.map(confluence_css, subset=extra)
            st.dataframe(sty.hide(axis="index"), use_container_width=True, height=260)
        else:
            st.write("No bars.")

    tape_box = st.container(height=640)
    with tape_box:
      for name in ORDER:
        if name not in packs:
            continue
        book, idx, ce, pe, t_lab, f_lab = packs[name]
        strike = book.get("atm") or "—"
        oexp = book.get("exp") or "—"
        if book.get("err"):
            st.caption(book["err"])

        def _pill(txt):
            u = str(txt or "").upper()
            bg = "#2e7d32" if u.startswith("LONG") else ("#c62828" if u.startswith("SHORT") else ("#00838f" if u.startswith("CONFLICT") else "#37474f"))
            return (
                f"<span style='background:{bg};color:#fff;font-size:11px;font-weight:700;"
                f"padding:2px 8px;border-radius:10px;letter-spacing:0.02em;'>{txt}</span>"
            )

        st.markdown(
            f"<div id='mis-sec-{name}' style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:8px 0 4px 0;'>"
            f"<span style='font-size:1.05rem;font-weight:700;color:#69F0AE;'>{name} · {st.session_state.get('mis_tf') or '1 min'}</span>"
            f"<span style='color:#78909C;font-size:11px;'>last updated {_upd}</span>"
            f"<span style='color:#B0BEC5;font-size:13px;'>Spot <b style='color:#EEE;'>{book.get('spot') or '—'}</b></span>"
            f"<span style='color:#B0BEC5;font-size:13px;'>Fut <b style='color:#EEE;'>{book.get('fut') or '—'}</b></span>"
            f"<span style='color:#B0BEC5;font-size:13px;'>ATM <b style='color:#EEE;'>{strike}</b> {oexp}</span>"
            f"<span style='color:#B0BEC5;font-size:13px;'>CE/PE <b style='color:#69F0AE;'>{ce['LTP']}</b> / <b style='color:#FF8A80;'>{pe['LTP']}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        draw_tape("Index / Fut", idx)
        c1, c2 = st.columns(2)
        with c1:
            draw_tape(book.get("ce_sym") or f"{strike} CE", ce)
        with c2:
            draw_tape(book.get("pe_sym") or f"{strike} PE", pe)
        st.divider()

    jump = str(st.session_state.get("mis_jump") or "")
    if jump:
        components.html(
            f"""
<script>
const id = "mis-sec-{jump}";
const doc = window.parent.document;
const el = doc.getElementById(id);
if (el) {{
  el.scrollIntoView({{behavior: "smooth", block: "start"}});
  let p = el.parentElement;
  while (p) {{
    const oy = window.parent.getComputedStyle(p).overflowY;
    if (oy === "auto" || oy === "scroll") {{
      const top = el.getBoundingClientRect().top - p.getBoundingClientRect().top + p.scrollTop - 8;
      p.scrollTo({{top: top, behavior: "smooth"}});
      break;
    }}
    p = p.parentElement;
  }}
}}
</script>
""",
            height=0,
        )
