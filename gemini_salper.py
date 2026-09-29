"""Gemini Scalper — live metric tapes for 7 indexes (structure, OI, IV, execution)."""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import streamlit as st

from multi_index_scalper import (
    ORDER, STEP, LOT_SIZES, TF_API, MCX_NAMES,
    rows_from_df, attach_efi, session_open, last_session_bar,
    _atm_from_master, _ist_now,
)

GS_TF = ("FIVE_MINUTE", "FIFTEEN_MINUTE")


def _ema(vals, span):
    if not vals:
        return None
    k = 2.0 / (span + 1.0)
    e = float(vals[0])
    for x in vals[1:]:
        e = float(x) * k + e * (1.0 - k)
    return e


def _sma(vals, n):
    if not vals or len(vals) < n:
        return None
    chunk = vals[-n:]
    return sum(chunk) / float(n)


def _bs_d1(S, K, T, r, sig):
    if S <= 0 or K <= 0 or T <= 0 or sig <= 0:
        return None
    return (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _iv_newton(price, S, K, T, r, call=True):
    if not price or S <= 0 or K <= 0 or T <= 0:
        return None
    sig = 0.25
    for _ in range(20):
        d1 = _bs_d1(S, K, T, r, sig)
        if d1 is None:
            return None
        d2 = d1 - sig * math.sqrt(T)
        theo = (S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)) if call else (
            K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        )
        vega = S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        if vega < 1e-8:
            break
        sig = max(0.01, sig - (theo - price) / vega)
    return sig


def _delta(S, K, T, r, sig, call=True):
    d1 = _bs_d1(S, K, T, r, sig)
    if d1 is None:
        return None
    return _norm_cdf(d1) if call else _norm_cdf(d1) - 1.0


def _last(rows):
    return rows[-1] if rows else {}


def _day_hl(rows, day):
    hi = lo = None
    for r in rows or []:
        t = str(r.get("time") or "")
        if not t.startswith(day):
            continue
        try:
            h, l = float(r["high"]), float(r["low"])
        except Exception:
            continue
        hi = h if hi is None else max(hi, h)
        lo = l if lo is None else min(lo, l)
    return hi, lo


def _vol_vs_sma(rows, n=10):
    vs = [float(r.get("volume") or 0) for r in (rows or [])]
    last = vs[-1] if vs else None
    sma = _sma(vs, n)
    if last is None or sma in (None, 0):
        return last, sma, None
    return last, sma, last / sma


def _strikes_band(df_master, name, spot, fo, n=3):
    if df_master is None or not spot:
        return []
    step = STEP.get(name, 50)
    atm = int(round(float(spot) / step) * step)
    want = [atm + i * step for i in range(-n, n + 1)]
    today = dt.datetime.now().date()
    names = ["GOLDM", "GOLD"] if name == "GOLDM" else [name]
    d = df_master.copy()
    d["name_u"] = d["name"].astype(str).str.upper()
    d = d[d["name_u"].isin([x.upper() for x in names])]
    if "exch_seg" in d.columns:
        d = d[d["exch_seg"].astype(str) == str(fo)]
    out = []
    nearest_exp = None
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
            if min(abs(k - w) for w in want) > 0.01:
                continue
            sym = str(r.get("symbol") or "")
            side = "CE" if sym.endswith("CE") else ("PE" if sym.endswith("PE") else None)
            if not side:
                continue
            rows.append((e, int(round(k)), side, str(r.get("token")), sym))
        except Exception:
            continue
    if not rows:
        return []
    nearest_exp = min(e for e, *_ in rows)
    for e, k, side, tok, sym in rows:
        if e != nearest_exp:
            continue
        out.append({"k": k, "side": side, "tok": tok, "sym": sym, "exp": str(e)})
    return out


def _seed_tf(name, fetch_fn, df_master, token_map, fut_fn, interval, tag):
    from multi_index_scalper import seed_book
    old = st.session_state.get("mis_tf")
    hold = st.session_state.get("_mis_books")
    lab = "5 min" if interval == "FIVE_MINUTE" else "15 min"
    st.session_state["mis_tf"] = lab
    st.session_state["_mis_books"] = {}
    try:
        book = seed_book(name, fetch_fn, df_master, token_map, fut_fn=fut_fn)
        return dict(book or {})
    finally:
        if old:
            st.session_state["mis_tf"] = old
        if hold is not None:
            st.session_state["_mis_books"] = hold


def seed_gemini_scalper(fetch_fn, df_master, token_map, fut_fn, names):
    store = st.session_state.setdefault("_gs_books", {})
    bar = st.progress(0.0, text="Seeding Gemini Scalper 5m/15m…")
    n = max(len(names), 1)
    for i, name in enumerate(names):
        try:
            b5 = _seed_tf(name, fetch_fn, df_master, token_map, fut_fn, "FIVE_MINUTE", "5")
            b15 = _seed_tf(name, fetch_fn, df_master, token_map, fut_fn, "FIFTEEN_MINUTE", "15")
            store[name] = {"5": b5, "15": b15}
        except Exception as e:
            store[name] = {"err": str(e)[:200], "5": {}, "15": {}}
        bar.progress((i + 1) / n, text=f"Seeded {name}")
    bar.empty()
    st.session_state["_gs_books"] = store
    st.session_state["_gs_seed_ts"] = _ist_now().strftime("%H:%M:%S")


def _metrics(name, pack, quotes, df_master, token_map):
    b5, b15 = pack.get("5") or {}, pack.get("15") or {}
    r5, r15 = attach_efi(list(b5.get("idx_rows") or [])), attach_efi(list(b15.get("idx_rows") or []))
    c5, p5 = attach_efi(list(b5.get("ce_rows") or [])), attach_efi(list(b5.get("pe_rows") or []))
    L5, L15 = _last(r5), _last(r15)
    closes5 = [float(x["price"]) for x in r5 if x.get("price") is not None]
    today = _ist_now().strftime("%Y-%m-%d")
    yday = (_ist_now().date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    idh, idl = _day_hl(r5, today)
    pdh, pdl = _day_hl(r5, yday)
    if pdh is None:
        # last other-day high in tape
        days = sorted({str(x.get("time") or "")[:10] for x in r5 if x.get("time")})
        prev = [d for d in days if d < today]
        if prev:
            pdh, pdl = _day_hl(r5, prev[-1])
    v5, vs5, vr5 = _vol_vs_sma(r5, 10)
    v15, vs15, vr15 = _vol_vs_sma(r15, 10)
    spot = b5.get("spot") or L5.get("price")
    fut = b5.get("fut") or L5.get("price")
    atm = b5.get("atm")
    ce_l = _last(c5)
    pe_l = _last(p5)
    T = 5 / 365.0
    try:
        exp = str(b5.get("exp") or "")
        if exp:
            ed = dt.datetime.strptime(exp[:10], "%Y-%m-%d").date()
            T = max((ed - _ist_now().date()).days, 1) / 365.0
    except Exception:
        pass
    S = float(spot or 0) or 0.0
    K = float(atm or 0) or 0.0
    ce_px = float(ce_l.get("price") or 0) or None
    pe_px = float(pe_l.get("price") or 0) or None
    iv_ce = _iv_newton(ce_px, S, K, T, 0.06, True) if ce_px and S and K else None
    iv_pe = _iv_newton(pe_px, S, K, T, 0.06, False) if pe_px and S and K else None
    d_ce = _delta(S, K, T, 0.06, iv_ce or 0.15, True) if S and K else None
    d_pe = _delta(S, K, T, 0.06, iv_pe or 0.15, False) if S and K else None
    iv_avg = None
    if iv_ce and iv_pe:
        iv_avg = 0.5 * (iv_ce + iv_pe)

    band = _strikes_band(df_master, name, spot, token_map.get(name, ("", "NSE", "NFO"))[2], 3)
    oi_lines = []
    ce_oi = pe_oi = ce_doi = pe_doi = 0.0
    for rec in band:
        q = quotes.get(str(rec["tok"])) or {}
        oi = q.get("oi") or q.get("opi") or q.get("openInterest")
        doi = q.get("doi") or q.get("oichange") or q.get("oiChange")
        ltp = q.get("ltp")
        try:
            oi_f = float(oi or 0)
        except Exception:
            oi_f = 0.0
        try:
            doi_f = float(doi or 0)
        except Exception:
            doi_f = 0.0
        if rec["side"] == "CE":
            ce_oi += oi_f
            ce_doi += doi_f
        else:
            pe_oi += oi_f
            pe_doi += doi_f
        mark = "ATM" if atm and abs(rec["k"] - float(atm)) < 0.1 else ("ITM" if (rec["side"] == "CE" and rec["k"] < float(spot or 0)) or (rec["side"] == "PE" and rec["k"] > float(spot or 0)) else "OTM")
        oi_lines.append(f"{int(rec['k'])} {rec['side']} {mark} OI={oi_f:.0f} ΔOI={doi_f:.0f} LTP={ltp}")

    pcr = (pe_oi / ce_oi) if ce_oi else None
    unwind = "—"
    if ce_doi < 0 and pe_doi >= 0:
        unwind = "CE OI falling (call covering / unwind)"
    elif pe_doi < 0 and ce_doi >= 0:
        unwind = "PE OI falling (put covering / unwind)"
    elif ce_doi < 0 and pe_doi < 0:
        unwind = "Both sides unwinding"
    elif ce_doi > 0 and pe_doi > 0:
        unwind = "Both sides adding OI"

    # RR from spot: 0.4% SL / 0.8% tgt as placeholder structure vs option premium
    sl_pct, tgt_pct = -0.4, 0.8
    try:
        if ce_px:
            ce_sl = ce_px * (1 + sl_pct / 0.4 * 0.08)
            # keep simple 8% opt sl / 16% tgt if RR 1:2 on premium
        ce_rr = 2.0
    except Exception:
        ce_rr = None

    vix = None
    vq = quotes.get("VIX") or quotes.get("26017")
    if vq:
        vix = vq.get("ltp")

    return {
        "Index": name,
        "Spot": spot, "Fut": fut, "ATM": atm, "Exp": b5.get("exp"),
        "5m O": L5.get("open"), "5m H": L5.get("high"), "5m L": L5.get("low"), "5m C": L5.get("price"),
        "15m O": L15.get("open"), "15m H": L15.get("high"), "15m L": L15.get("low"), "15m C": L15.get("price"),
        "5m Vol": v5, "5m VolSMA10": vs5, "5m Vol/SMA": None if vr5 is None else round(vr5, 2),
        "15m Vol/SMA": None if vr15 is None else round(vr15, 2),
        "IDH": idh, "IDL": idl, "PDH": pdh, "PDL": pdl,
        "EMA20": None if not closes5 else round(_ema(closes5, 20) or 0, 2),
        "EMA200": None if len(closes5) < 30 else round(_ema(closes5, 200) or 0, 2),
        "CE LTP": ce_px, "CE VWAP": ce_l.get("VWAP"),
        "PE LTP": pe_px, "PE VWAP": pe_l.get("VWAP"),
        "CE vs VWAP": None if not (ce_px and ce_l.get("VWAP")) else round(ce_px - float(ce_l["VWAP"]), 2),
        "PE vs VWAP": None if not (pe_px and pe_l.get("VWAP")) else round(pe_px - float(pe_l["VWAP"]), 2),
        "Δ CE": None if d_ce is None else round(d_ce, 3),
        "Δ PE": None if d_pe is None else round(d_pe, 3),
        "IV CE": None if iv_ce is None else round(iv_ce * 100, 1),
        "IV PE": None if iv_pe is None else round(iv_pe * 100, 1),
        "IV avg": None if iv_avg is None else round(iv_avg * 100, 1),
        "PCR": None if pcr is None else round(pcr, 3),
        "CE OI": ce_oi, "PE OI": pe_oi, "CE ΔOI": ce_doi, "PE ΔOI": pe_doi,
        "Unwind": unwind,
        "VIX": vix,
        "oi_lines": oi_lines,
        "Spot SL": None if not spot else round(float(spot) * 0.996, 2),
        "Spot Tgt": None if not spot else round(float(spot) * 1.008, 2),
        "RR": 2.0,
        "Bar": L5.get("time"),
    }


def render_gemini_scalper(fetch_fn, df_master, token_map, get_client, fut_fn=None, quote_fn=None):
    st.markdown(
        "<div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap;'>"
        "<span style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Gemini Scalper</span>"
        "<span style='font-size:11px;color:#90A4AE;'>5m + 15m structure · ATM±3 OI · IV/Δ · premium vs VWAP</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    enabled = dict(st.session_state.get("gs_enabled") or {n: True for n in ORDER})
    names = [n for n in ORDER if enabled.get(n)]
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    with c1:
        auto = st.checkbox("Auto-Refresh 5s", value=bool(st.session_state.get("enable_main_refresh")), key="cb_gs_refresh")
        if auto != bool(st.session_state.get("enable_main_refresh")):
            st.session_state["enable_main_refresh"] = auto
            st.rerun()
    with c2:
        if st.button("Seed / refresh tapes", key="btn_gs_seed"):
            seed_gemini_scalper(fetch_fn, df_master, token_map, fut_fn, names)
            st.rerun()
    with c3:
        ts = st.session_state.get("_gs_seed_ts")
        st.caption(f"Last seed {ts or '—'} · quotes tick when Auto-Refresh is on")

    books = st.session_state.get("_gs_books") or {}
    if not books:
        st.info("Click Seed / refresh tapes.")
        return

    quotes = {}
    if quote_fn:
        pairs = []
        for name in names:
            pack = books.get(name) or {}
            b5 = pack.get("5") or {}
            for k in ("idx_tok", "cash_tok", "ce_tok", "pe_tok"):
                tok = b5.get(k)
                if tok:
                    exch = b5.get("idx_exch") if k != "cash_tok" else b5.get("cash_exch")
                    pairs.append((exch or "NFO", str(tok)))
            band = _strikes_band(df_master, name, b5.get("spot"), token_map.get(name, ("", "NSE", "NFO"))[2], 3)
            for rec in band:
                pairs.append((token_map.get(name, ("", "NSE", "NFO"))[2], rec["tok"]))
        # India VIX
        pairs.append(("NSE", "26017"))
        try:
            quotes = quote_fn(pairs) or {}
            quotes["VIX"] = quotes.get("26017") or {}
        except Exception as e:
            st.caption(f"Quote tick: {e}")

    rows = []
    oi_blocks = []
    for name in names:
        pack = books.get(name)
        if not pack:
            continue
        try:
            m = _metrics(name, pack, quotes, df_master, token_map)
            oi_blocks.append((name, m.pop("oi_lines")))
            rows.append(m)
        except Exception as e:
            rows.append({"Index": name, "Unwind": str(e)[:80]})

    if not rows:
        st.write("No metrics.")
        return

    df = pd.DataFrame(rows)
    struct = [c for c in (
        "Index", "Spot", "Fut", "5m O", "5m H", "5m L", "5m C", "15m O", "15m H", "15m L", "15m C",
        "5m Vol", "5m VolSMA10", "5m Vol/SMA", "15m Vol/SMA",
        "IDH", "IDL", "PDH", "PDL", "EMA20", "EMA200", "Bar",
    ) if c in df.columns]
    deriv = [c for c in (
        "Index", "ATM", "Exp", "CE OI", "PE OI", "CE ΔOI", "PE ΔOI", "PCR", "Unwind",
    ) if c in df.columns]
    greek = [c for c in (
        "Index", "Δ CE", "Δ PE", "IV CE", "IV PE", "IV avg", "VIX",
    ) if c in df.columns]
    exe = [c for c in (
        "Index", "CE LTP", "CE VWAP", "CE vs VWAP", "PE LTP", "PE VWAP", "PE vs VWAP",
        "Spot SL", "Spot Tgt", "RR",
    ) if c in df.columns]

    st.caption("1. Spot structure & trend")
    st.dataframe(df[struct], use_container_width=True, hide_index=True, height=280)
    st.caption("2. Derivative & OI (ATM ± 3 from master + FULL quote)")
    st.dataframe(df[deriv], use_container_width=True, hide_index=True, height=280)
    cols = st.columns(min(4, max(len(oi_blocks), 1)))
    for i, (nm, lines) in enumerate(oi_blocks):
        with cols[i % len(cols)]:
            st.markdown(f"**{nm} strikes**")
            st.text("\n".join(lines[:14]) or "—")
    st.caption("3. Option pricing & Greeks (BS from LTP; Δ ATM target 0.45–0.55)")
    st.dataframe(df[greek], use_container_width=True, hide_index=True, height=220)
    st.caption("4. Premium LTP vs VWAP · spot SL/target (0.4% / 0.8%) · RR")
    st.dataframe(df[exe], use_container_width=True, hide_index=True, height=260)
