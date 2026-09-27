"""Multi Index Crypto Scalper — same desk UI, public Binance + Deribit."""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from multi_index_scalper import (
    LOT_SIZES,
    TF_API,
    _ist_now,
    _send_mis_telegram,
    apply_live_tape,
    attach_bar_confluence,
    confluence_css,
    confluence_label,
    last_confirmed,
    last_pack,
    rebuild_swings,
    rows_from_df,
    setup_row,
    style_setups,
    watch_log_html,
    _vote_flow,
    _vote_status,
)

ORDER = ["BTC", "ETH", "SOL", "BNB", "XRP"]
BINANCE_FUT = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}
DERIBIT_CCY = {"BTC": "BTC", "ETH": "ETH"}
STEP = {"BTC": 500, "ETH": 50, "SOL": 5, "BNB": 10, "XRP": 0.05}
CRYPTO_LOTS = {"BTC": 1, "ETH": 1, "SOL": 1, "BNB": 1, "XRP": 1}
TF_BINANCE = {
    "1 min": "1m",
    "2 min": "1m",
    "3 min": "3m",
    "5 min": "5m",
    "15 min": "15m",
}
TF_DERIBIT = {
    "1 min": "1",
    "2 min": "1",
    "3 min": "3",
    "5 min": "5",
    "15 min": "15",
}


_HDR = {"User-Agent": "Mozilla/5.0 GammaDesk/1.0", "Accept": "application/json"}


def _get_json(url, params=None, timeout=12):
    r = requests.get(url, params=params or {}, timeout=timeout, headers=_HDR)
    r.raise_for_status()
    return r.json()


def _klines_from_binance_list(js):
    rows = []
    for k in js or []:
        rows.append({
            "time": dt.datetime.utcfromtimestamp(int(k[0]) / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
        })
    return pd.DataFrame(rows)


def fetch_binance_klines(symbol, interval, limit=500):
    """fapi is 451 in many regions (IN). Fall back to Vision spot, Bybit, Deribit perp."""
    errors = []
    for url, params in (
        ("https://data-api.binance.vision/api/v3/klines",
         {"symbol": symbol, "interval": interval, "limit": int(limit)}),
        ("https://api.binance.com/api/v3/klines",
         {"symbol": symbol, "interval": interval, "limit": int(limit)}),
        ("https://fapi.binance.com/fapi/v1/klines",
         {"symbol": symbol, "interval": interval, "limit": int(limit)}),
    ):
        try:
            js = _get_json(url, params)
            df = _klines_from_binance_list(js)
            if not df.empty:
                return df
        except Exception as e:
            errors.append(f"{url.split('/')[2]}:{e}")
    by_int = {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(interval, "1")
    try:
        js = _get_json(
            "https://api.bybit.com/v5/market/kline",
            {"category": "linear", "symbol": symbol, "interval": by_int, "limit": min(int(limit), 200)},
        )
        lst = (((js or {}).get("result") or {}).get("list") or [])
        rows = []
        for k in reversed(lst):
            rows.append({
                "time": dt.datetime.utcfromtimestamp(int(k[0]) / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
                "close": float(k[4]), "volume": float(k[5]),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            return df
    except Exception as e:
        errors.append(f"bybit:{e}")
    ccy = symbol.replace("USDT", "")
    if ccy in ("BTC", "ETH"):
        try:
            res = {"1m": "1", "3m": "3", "5m": "5", "15m": "15"}.get(interval, "1")
            df = deribit_chart(f"{ccy}-PERPETUAL", res)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            errors.append(f"deribit:{e}")
    raise RuntimeError(" / ".join(errors[:3]) or "no kline source")


def fetch_binance_ticker(symbol):
    for url, params, path in (
        ("https://data-api.binance.vision/api/v3/ticker/price", {"symbol": symbol}, "price"),
        ("https://api.binance.com/api/v3/ticker/price", {"symbol": symbol}, "price"),
        ("https://fapi.binance.com/fapi/v1/ticker/price", {"symbol": symbol}, "price"),
        ("https://api.bybit.com/v5/market/tickers", {"category": "linear", "symbol": symbol}, "bybit"),
    ):
        try:
            js = _get_json(url, params)
            if path == "bybit":
                lst = (((js or {}).get("result") or {}).get("list") or [])
                if lst:
                    return float(lst[0].get("lastPrice"))
            else:
                return float(js.get("price") or 0)
        except Exception:
            continue
    ccy = symbol.replace("USDT", "")
    if ccy in ("BTC", "ETH"):
        return deribit_index(ccy)
    raise RuntimeError(f"no ticker for {symbol}")


def deribit_index(ccy):
    js = _get_json(
        "https://www.deribit.com/api/v2/public/get_index_price",
        {"index_name": f"{ccy.lower()}_usd"},
    )
    return float(((js or {}).get("result") or {}).get("index_price") or 0)


def deribit_instruments(ccy):
    js = _get_json(
        "https://www.deribit.com/api/v2/public/get_instruments",
        {"currency": ccy, "kind": "option", "expired": "false"},
    )
    return (js or {}).get("result") or []


def deribit_chart(name, resolution="1", hours=36):
    now = int(time.time() * 1000)
    start = now - hours * 3600 * 1000
    js = _get_json(
        "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
        {
            "instrument_name": name,
            "start_timestamp": start,
            "end_timestamp": now,
            "resolution": str(resolution),
        },
    )
    res = (js or {}).get("result") or {}
    ticks = res.get("ticks") or []
    rows = []
    for i, t in enumerate(ticks):
        try:
            rows.append({
                "time": dt.datetime.utcfromtimestamp(int(t) / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(res["open"][i]),
                "high": float(res["high"][i]),
                "low": float(res["low"][i]),
                "close": float(res["close"][i]),
                "volume": float((res.get("volume") or [0] * len(ticks))[i] or 0),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def pick_atm_deribit(ccy, spot):
    if not spot:
        return None
    ins = deribit_instruments(ccy)
    today = dt.datetime.utcnow().date()
    step = STEP.get(ccy, 50)
    atm = int(round(float(spot) / step) * step)
    by_exp = {}
    for it in ins:
        try:
            exp_ms = int(it.get("expiration_timestamp") or 0)
            exp = dt.datetime.utcfromtimestamp(exp_ms / 1000).date()
            if exp < today:
                continue
            k = float(it.get("strike") or 0)
            if abs(k - atm) > step * 0.51:
                continue
            side = str(it.get("option_type") or "").lower()
            if side not in ("call", "put"):
                continue
            by_exp.setdefault(exp, {})[side] = it.get("instrument_name")
        except Exception:
            continue
    for exp in sorted(by_exp):
        if "call" in by_exp[exp] and "put" in by_exp[exp]:
            return atm, str(exp), by_exp[exp]["call"], by_exp[exp]["put"]
    return None


def seed_crypto(name):
    tf = st.session_state.get("cry_tf") or "1 min"
    b_int = TF_BINANCE.get(tf, "1m")
    d_res = TF_DERIBIT.get(tf, "1")
    book = {
        "name": name, "idx_rows": [], "idx_sw": [], "ce_rows": [], "ce_sw": [],
        "pe_rows": [], "pe_sw": [], "spot": None, "fut": None, "atm": None, "exp": None,
        "ce_tok": None, "pe_tok": None, "ce_sym": "", "pe_sym": "",
        "idx_tok": BINANCE_FUT.get(name), "idx_exch": "BINANCE", "err": "",
        "ce_1m_rows": [], "pe_1m_rows": [],
    }
    try:
        df = fetch_binance_klines(BINANCE_FUT[name], b_int, 500)
        book["idx_rows"] = rows_from_df(df)
        book["idx_sw"] = rebuild_swings(book["idx_rows"])
        if book["idx_rows"]:
            book["fut"] = book["idx_rows"][-1]["price"]
            book["spot"] = book["fut"]
        try:
            book["spot"] = fetch_binance_ticker(BINANCE_FUT[name])
        except Exception:
            pass
        ccy = DERIBIT_CCY.get(name)
        if ccy:
            try:
                dspot = deribit_index(ccy) or book["spot"]
                book["spot"] = dspot or book["spot"]
                atm = pick_atm_deribit(ccy, book["spot"])
                if atm:
                    strike, exp, ce_n, pe_n = atm
                    book["atm"], book["exp"] = strike, exp
                    book["ce_sym"], book["pe_sym"] = ce_n, pe_n
                    book["ce_tok"], book["pe_tok"] = ce_n, pe_n
                    dce = deribit_chart(ce_n, d_res)
                    dpe = deribit_chart(pe_n, d_res)
                    book["ce_rows"] = rows_from_df(dce)
                    book["pe_rows"] = rows_from_df(dpe)
                    book["ce_sw"] = rebuild_swings(book["ce_rows"])
                    book["pe_sw"] = rebuild_swings(book["pe_rows"])
                    if d_res == "1":
                        book["ce_1m_rows"] = book["ce_rows"]
                        book["pe_1m_rows"] = book["pe_rows"]
                    else:
                        book["ce_1m_rows"] = rows_from_df(deribit_chart(ce_n, "1"))
                        book["pe_1m_rows"] = rows_from_df(deribit_chart(pe_n, "1"))
            except Exception as e:
                book["err"] = f"options:{e}"
    except Exception as e:
        book["err"] = str(e)
    books = st.session_state.get("_cry_books") or {}
    books[name] = book
    st.session_state["_cry_books"] = books
    return book


def live_crypto_quotes(books, enabled):
    for name, book in list(books.items()):
        if not enabled.get(name):
            continue
        sym = BINANCE_FUT.get(name)
        if not sym:
            continue
        try:
            px = fetch_binance_ticker(sym)
            apply_live_tape(book, "idx", px, None, None)
            book["spot"] = px
        except Exception:
            pass
    return books


def render_crypto_scalper():
    st.session_state.setdefault("cry_tf", "1 min")
    h1, hL, h2, hTf, h3 = st.columns([1.7, 2.0, 0.85, 0.22, 0.55])
    with h1:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
            "<span style='font-size:1.12rem;font-weight:700;color:#69F0AE;'>Multi Index Crypto Scalper</span>"
            "<span style='color:#90A4AE;font-size:11px;'>Binance + Deribit public</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with hL:
        st.markdown(
            "<div style='font-size:11px;color:#90A4AE;line-height:1.25;'>"
            "<b style='color:#B0BEC5;'>Status</b> WATCH · CONFIRMED · TREND DENIED &nbsp;|&nbsp; "
            "<b style='color:#B0BEC5;'>Flow</b> ABSORB · EXH · ACCEL"
            "</div>",
            unsafe_allow_html=True,
        )
    with h2:
        auto = st.checkbox(
            "Auto-Refresh 5s",
            value=bool(st.session_state.get("enable_main_refresh")),
            key="cb_cry_refresh_main",
        )
    with hTf:
        st.markdown("<div style='padding-top:8px;font-size:12px;color:#B0BEC5;font-weight:700;'>TF</div>", unsafe_allow_html=True)
    with h3:
        tfs = ["1 min", "2 min", "3 min", "5 min", "15 min"]
        cur = st.session_state.get("cry_tf") or "1 min"
        if cur not in tfs:
            cur = "1 min"
        tf = st.selectbox("TF", tfs, index=tfs.index(cur), key="cry_tf_sel", label_visibility="collapsed")
        if tf != st.session_state.get("cry_tf"):
            st.session_state["cry_tf"] = tf
            st.session_state["cry_need_seed"] = True
            st.session_state["_cry_books"] = {}
            st.rerun()
        st.session_state["cry_tf"] = tf
    if auto != bool(st.session_state.get("enable_main_refresh")):
        st.session_state["enable_main_refresh"] = auto
        st.rerun()

    enabled = dict(st.session_state.get("cry_enabled") or {n: True for n in ORDER})
    max_loss = float(st.session_state.get("cry_max_loss") or 2000)
    books = st.session_state.get("_cry_books") or {}
    if st.session_state.get("cry_need_seed"):
        names = [n for n in ORDER if enabled.get(n)]
        bar = st.progress(0.0, text="Seeding crypto tapes…")
        for i, n in enumerate(names):
            try:
                seed_crypto(n)
            except Exception as e:
                books = st.session_state.get("_cry_books") or {}
                books[n] = {"name": n, "err": str(e), "idx_rows": []}
                st.session_state["_cry_books"] = books
            bar.progress((i + 1) / max(len(names), 1), text=f"Seeded {n}")
        bar.empty()
        st.session_state["cry_need_seed"] = False
        books = st.session_state.get("_cry_books") or {}

    if st.session_state.get("enable_main_refresh") and books:
        try:
            books = live_crypto_quotes(books, enabled)
            st.session_state["_cry_books"] = books
        except Exception as e:
            st.session_state["_cry_live_err"] = str(e)[:160]

    summary, setups, packs = [], [], {}
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
            _vote_status(idx["WATCH"]), _vote_status(ce["WATCH"]),
            _vote_status(pe["WATCH"], invert=True),
        ], "TREND")
        f_lab = confluence_label([
            _vote_flow(idx), _vote_flow(ce), _vote_flow(pe, invert=True),
        ], "FLOW")
        packs[name] = (book, idx, ce, pe, t_lab, f_lab)
        strike = book.get("atm") or "—"
        oexp = book.get("exp") or "—"
        summary.append({
            "Index": name,
            "Spot": book.get("spot") if book.get("spot") is not None else "—",
            "Fut LTP": book.get("fut") if book.get("fut") is not None else "—",
            "ATM": strike, "Opt exp": oexp,
            "Idx WATCH": idx["WATCH"], "Idx Flow": idx["Flow"], "Idx Reason": idx["Reason"],
            "CE WATCH": ce["WATCH"], "CE Flow": ce["Flow"], "CE Reason": ce["Reason"], "CE LTP": ce["LTP"],
            "PE WATCH": pe["WATCH"], "PE Flow": pe["Flow"], "PE Reason": pe["Reason"], "PE LTP": pe["LTP"],
            "Trend": t_lab, "Flow x/3": f_lab, "Bar": idx["Bar"],
        })
        lot = CRYPTO_LOTS.get(name, 1)
        for source, pack, rule in (
            ("FUT", idx, {"CONFIRMED LONG": "CE", "CONFIRMED SHORT": "PE"}),
            ("CE", ce, {"CONFIRMED LONG": "CE", "CONFIRMED SHORT": "PE"}),
            ("PE", pe, {"CONFIRMED LONG": "PE", "CONFIRMED SHORT": "CE"}),
        ):
            kind, bar = last_confirmed(pack.get("ann"))
            if not kind:
                continue
            buy = rule[kind]
            buy_pack = ce if buy == "CE" else pe
            fine = book.get("ce_1m_rows") if buy == "CE" else book.get("pe_1m_rows")
            if fine:
                buy_pack = dict(buy_pack)
                buy_pack["ann"] = [
                    {"time": r.get("time"), "high": r.get("high"), "low": r.get("low"), "price": r.get("price")}
                    for r in fine
                ]
            ts = bar.get("time") if bar else ""
            row = setup_row(name, f"{source} {kind}", kind, buy, buy_pack, strike, oexp, lot, ts, max_loss)
            if row:
                setups.append(row)

    try:
        _send_mis_telegram(setups)
    except Exception:
        pass

    _upd = _ist_now().strftime("%d-%m-%Y %H:%M:%S")
    st.caption(f"WATCH LOG    last updated {_upd}")
    if summary:
        st.markdown(watch_log_html(pd.DataFrame(summary)), unsafe_allow_html=True)
    else:
        st.info("Seed crypto tapes in the sidebar.")
    st.caption(f"SETUPS    last updated {_upd}")
    if setups:
        sdf = pd.DataFrame(setups)
        if "Bar" in sdf.columns:
            sdf = sdf.sort_values("Bar", ascending=False).reset_index(drop=True)
        st.dataframe(style_setups(sdf), use_container_width=True, height=38 * 8 + 20, hide_index=True)
    else:
        st.write("No CONFIRMED LONG/SHORT on FUT / CE / PE.")

    def draw_tape(title, pack):
        st.markdown(f"**{title}**  ·  last updated {_upd}")
        if pack["ann"]:
            df = pd.DataFrame(pack["ann"][::-1])
            cols = [c for c in ["time", "status", "flow", "Trend", "Flow x/3", "watch_reason", "price", "CVD", "RDI", "Disp", "volume", "VWAP"] if c in df.columns]
            sty = df[cols].style
            extra = [c for c in ("Trend", "Flow x/3") if c in df.columns]
            if extra:
                sty = sty.map(confluence_css, subset=extra)
            st.dataframe(sty.hide(axis="index"), use_container_width=True, height=260)
        else:
            st.write("No bars.")

    with st.container(height=640):
        for name in ORDER:
            if name not in packs:
                continue
            book, idx, ce, pe, t_lab, f_lab = packs[name]
            strike = book.get("atm") or "—"
            oexp = book.get("exp") or "—"
            if book.get("err"):
                st.caption(book["err"])
            st.markdown(
                f"<div id='cry-sec-{name}' style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:8px 0 4px 0;'>"
                f"<span style='font-size:1.05rem;font-weight:700;color:#69F0AE;'>{name} · {st.session_state.get('cry_tf') or '1 min'}</span>"
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
