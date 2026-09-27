"""Multi Index Crypto Scalper — same desk UI, public Binance + Deribit."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

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


def _bn_keys():
    def g(*names):
        for n in names:
            try:
                if n in st.secrets and str(st.secrets[n]).strip():
                    return str(st.secrets[n]).strip()
            except Exception:
                pass
            v = (os.getenv(n) or "").strip()
            if v:
                return v
        return ""
    return g("BINANCE_API_KEY", "BINANCE_KEY"), g("BINANCE_SECRET_KEY", "BINANCE_SECRET", "BINANCE_API_SECRET")


def _bn_proxies():
    def g(*names):
        for n in names:
            try:
                if n in st.secrets and str(st.secrets[n]).strip():
                    return str(st.secrets[n]).strip()
            except Exception:
                pass
            v = (os.getenv(n) or "").strip()
            if v:
                return v
        return ""
    url = g("BINANCE_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY")
    if not url:
        return None
    return {"http": url, "https": url}


def _bn_headers():
    h = dict(_HDR)
    key, _ = _bn_keys()
    if key:
        h["X-MBX-APIKEY"] = key
    return h


def _get_json(url, params=None, timeout=12, signed=False):
    params = dict(params or {})
    headers = _bn_headers()
    key, secret = _bn_keys()
    if signed and secret:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 10000
        qs = urlencode(params)
        params["signature"] = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers=headers,
        proxies=_bn_proxies(),
    )
    r.raise_for_status()
    return r.json()


def _ist_fmt(ts_sec):
    try:
        from zoneinfo import ZoneInfo
        t = dt.datetime.fromtimestamp(float(ts_sec), ZoneInfo("Asia/Kolkata"))
    except Exception:
        t = dt.datetime.utcfromtimestamp(float(ts_sec)) + dt.timedelta(hours=5, minutes=30)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _klines_from_binance_list(js):
    rows = []
    for k in js or []:
        rows.append({
            "time": _ist_fmt(int(k[0]) / 1000),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]),
        })
    return pd.DataFrame(rows)


def fetch_binance_klines(symbol, interval, limit=500):
    """fapi is 451 in many regions (IN). Fall back to Vision spot, Bybit, Deribit perp."""
    errors = []
    for url in (
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://fapi.binance.com/fapi/v1/klines",
    ):
        try:
            js = _get_json(url, {"symbol": symbol, "interval": interval, "limit": int(limit)})
            df = _klines_from_binance_list(js)
            if not df.empty:
                return df
        except Exception as e:
            errors.append(f"{url.split('/')[2]}:{e}")
    raise RuntimeError(" / ".join(errors[:3]) or "no Binance kline source")


def fetch_binance_ticker(symbol):
    errors = []
    for url in (
        "https://api.binance.com/api/v3/ticker/price",
        "https://api1.binance.com/api/v3/ticker/price",
        "https://fapi.binance.com/fapi/v1/ticker/price",
        "https://data-api.binance.vision/api/v3/ticker/price",
    ):
        try:
            js = _get_json(url, {"symbol": symbol})
            return float(js.get("price") or 0)
        except Exception as e:
            errors.append(str(e)[:80])
    raise RuntimeError(f"no Binance ticker for {symbol}: {errors[:1]}")


def binance_option_info():
    cached = st.session_state.get("_bn_opt_info")
    if cached is not None:
        return cached
    last_err = None
    for url in (
        "https://eapi.binance.com/eapi/v1/exchangeInfo",
        "https://eapi.binance.com/eapi/v1/exchangeInfo",
    ):
        try:
            js = _get_json(url)
            if isinstance(js, dict) and not (js.get("optionSymbols") or js.get("symbols")):
                last_err = js.get("msg") or js.get("code") or "no optionSymbols"
                continue
            st.session_state["_bn_opt_info"] = js
            return js
        except Exception as e:
            last_err = e
    return {"_err": str(last_err or "eapi unavailable")}


def _parse_bn_opt_symbol(sym):
    # BTC-260928-85000-C
    parts = str(sym).split("-")
    if len(parts) < 4:
        return None
    und, ymd, strike, side = parts[0], parts[1], parts[2], parts[-1]
    try:
        exp = dt.datetime.strptime(ymd, "%y%m%d").date()
        k = float(strike)
    except Exception:
        return None
    side = "C" if side.upper().startswith("C") else ("P" if side.upper().startswith("P") else "")
    if not side:
        return None
    return und, exp, k, side, str(sym)


def _candidate_strikes(spot, step):
    base = int(round(float(spot) / step) * step)
    out = []
    for stg in (step, max(step // 2, 1), step * 2):
        b = int(round(float(spot) / stg) * stg)
        for k in (b, b - stg, b + stg, base):
            if k > 0 and k not in out:
                out.append(int(k))
    return out[:10]


def _candidate_expiries():
    today = dt.datetime.utcnow().date()
    days = []
    for i in range(0, 14):
        days.append(today + dt.timedelta(days=i))
    for i in range(0, 28):
        d = today + dt.timedelta(days=i)
        if d.weekday() == 4:
            days.append(d)
    seen, out = set(), []
    for d in days:
        if d >= today and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def pick_atm_by_probe(name, spot):
    """Build BTC-YYMMDD-STRIKE-C and see which ticker lives."""
    und = name.upper()
    step = STEP.get(name, 50)
    for exp in _candidate_expiries():
        for k in _candidate_strikes(spot, step):
            ce = f"{und}-{exp:%y%m%d}-{k}-C"
            pe = f"{und}-{exp:%y%m%d}-{k}-P"
            try:
                last_c = fetch_binance_opt_last(ce)
            except Exception:
                last_c = 0
            if not last_c:
                continue
            try:
                last_p = fetch_binance_opt_last(pe)
            except Exception:
                last_p = 0
            if last_p:
                return int(k), str(exp), ce, pe
    return None


def pick_atm_binance(name, spot):
    if not spot:
        return None
    probed = pick_atm_by_probe(name, spot)
    if probed:
        return probed
    und = name.upper()
    today = dt.datetime.utcnow().date()
    step = STEP.get(name, 50)
    atm = float(int(round(float(spot) / step) * step))
    parsed = []

    info = binance_option_info()
    if info and not info.get("_err"):
        for it in info.get("optionSymbols") or info.get("symbols") or []:
            rec = _parse_bn_opt_symbol(it.get("symbol"))
            if rec:
                parsed.append(rec)
            else:
                try:
                    u = str(it.get("underlying") or "")
                    if not u.upper().startswith(und):
                        continue
                    raw_exp = it.get("expiryDate") or 0
                    exp = dt.datetime.utcfromtimestamp(int(raw_exp) / 1000).date()
                    k = float(it.get("strikePrice") or 0)
                    side = str(it.get("side") or "")
                    side = "C" if "CALL" in side.upper() or side == "C" else ("P" if "PUT" in side.upper() or side == "P" else "")
                    if side:
                        parsed.append((und, exp, k, side, str(it.get("symbol"))))
                except Exception:
                    continue

    if not parsed:
        try:
            tick = _get_json("https://eapi.binance.com/eapi/v1/ticker")
            for it in tick if isinstance(tick, list) else []:
                rec = _parse_bn_opt_symbol(it.get("symbol"))
                if rec and rec[0].upper() == und:
                    parsed.append(rec)
        except Exception:
            pass

    by_exp = {}
    for u, exp, k, side, sym in parsed:
        if u.upper() != und and not str(u).upper().startswith(und):
            continue
        if exp < today:
            continue
        if abs(k - float(spot)) > max(step * 2, float(spot) * 0.02):
            continue
        by_exp.setdefault(exp, []).append((abs(k - float(spot)), k, side, sym))

    for exp in sorted(by_exp):
        legs = by_exp[exp]
        legs.sort()
        ce = next((s for _, k, side, s in legs if side == "C"), None)
        pe = next((s for _, k, side, s in legs if side == "P"), None)
        if ce and pe:
            k_use = next((k for _, k, side, s in legs if s == ce), atm)
            return int(k_use), str(exp), ce, pe
    return None


def fetch_binance_opt_klines(symbol, interval, limit=500):
    last_err = None
    for url in (
        "https://eapi.binance.com/eapi/v1/klines",
        "https://eapi.binance.com/eapi/v1/klines",
    ):
        try:
            js = _get_json(url, {"symbol": symbol, "interval": interval, "limit": int(limit)})
            df = _klines_from_binance_list(js)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return pd.DataFrame()


def fetch_binance_opt_last(symbol):
    for url, params in (
        ("https://eapi.binance.com/eapi/v1/ticker", {"symbol": symbol}),
        ("https://eapi.binance.com/eapi/v1/mark", {"symbol": symbol}),
    ):
        try:
            js = _get_json(url, params)
            if isinstance(js, list):
                js = js[0] if js else {}
            for k in ("lastPrice", "markPrice", "bidPrice", "askPrice"):
                try:
                    v = float(js.get(k) or 0)
                    if v > 0:
                        return v
                except Exception:
                    continue
        except Exception:
            continue
    return 0.0


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
                "time": _ist_fmt(int(t) / 1000),
                "open": float(res["open"][i]),
                "high": float(res["high"][i]),
                "low": float(res["low"][i]),
                "close": float(res["close"][i]),
                "volume": float((res.get("volume") or [0] * len(ticks))[i] or 0),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def deribit_ticker(name):
    js = _get_json(
        "https://www.deribit.com/api/v2/public/ticker",
        {"instrument_name": name},
    )
    return (js or {}).get("result") or {}


def option_usd_df(df, index_px, last_usd=None):
    """Deribit option OHLC is in coin. Convert to USD."""
    try:
        px = float(index_px or 0)
    except Exception:
        px = 0.0
    if df is None or getattr(df, "empty", True):
        if last_usd and last_usd > 0:
            now = _ist_now().strftime("%Y-%m-%d %H:%M:%S")
            return pd.DataFrame([{
                "time": now, "open": last_usd, "high": last_usd,
                "low": last_usd, "close": last_usd, "volume": 0,
            }])
        return pd.DataFrame()
    out = df.copy()
    if px > 0:
        for c in ("open", "high", "low", "close"):
            if c in out.columns:
                out[c] = out[c].astype(float) * px
    if last_usd and last_usd > 0:
        if out.empty:
            now = _ist_now().strftime("%Y-%m-%d %H:%M:%S")
            out = pd.DataFrame([{
                "time": now, "open": last_usd, "high": last_usd,
                "low": last_usd, "close": last_usd, "volume": 0,
            }])
        else:
            out.iloc[-1, out.columns.get_loc("close")] = last_usd
    return out


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
            if abs(k - float(spot)) > max(step * 2, float(spot) * 0.03):
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
        loaded_opt = False
        try:
            atm = pick_atm_binance(name, book.get("spot"))
            if atm:
                strike, exp, ce_n, pe_n = atm
                book["opt_src"] = "binance"
                book["atm"], book["exp"] = strike, exp
                book["ce_sym"], book["pe_sym"] = ce_n, pe_n
                book["ce_tok"], book["pe_tok"] = ce_n, pe_n
                ce_last = fetch_binance_opt_last(ce_n)
                pe_last = fetch_binance_opt_last(pe_n)
                try:
                    dce = fetch_binance_opt_klines(ce_n, b_int)
                except Exception:
                    dce = pd.DataFrame()
                try:
                    dpe = fetch_binance_opt_klines(pe_n, b_int)
                except Exception:
                    dpe = pd.DataFrame()
                now = _ist_now().strftime("%Y-%m-%d %H:%M:%S")
                if (dce is None or dce.empty) and ce_last:
                    dce = pd.DataFrame([{"time": now, "open": ce_last, "high": ce_last, "low": ce_last, "close": ce_last, "volume": 0}])
                elif ce_last and dce is not None and not dce.empty:
                    dce.iloc[-1, dce.columns.get_loc("close")] = ce_last
                if (dpe is None or dpe.empty) and pe_last:
                    dpe = pd.DataFrame([{"time": now, "open": pe_last, "high": pe_last, "low": pe_last, "close": pe_last, "volume": 0}])
                elif pe_last and dpe is not None and not dpe.empty:
                    dpe.iloc[-1, dpe.columns.get_loc("close")] = pe_last
                book["ce_rows"] = rows_from_df(dce)
                book["pe_rows"] = rows_from_df(dpe)
                book["ce_sw"] = rebuild_swings(book["ce_rows"])
                book["pe_sw"] = rebuild_swings(book["pe_rows"])
                if b_int == "1m":
                    book["ce_1m_rows"] = book["ce_rows"]
                    book["pe_1m_rows"] = book["pe_rows"]
                else:
                    book["ce_1m_rows"] = rows_from_df(fetch_binance_opt_klines(ce_n, "1m"))
                    book["pe_1m_rows"] = rows_from_df(fetch_binance_opt_klines(pe_n, "1m"))
                loaded_opt = True
        except Exception as e:
            book["err"] = f"binance-opt:{e}"
        if not loaded_opt and name in DERIBIT_CCY:
            try:
                ccy = DERIBIT_CCY[name]
                dspot = deribit_index(ccy) or book.get("spot")
                book["spot"] = dspot or book.get("spot")
                atm = pick_atm_deribit(ccy, book.get("spot"))
                if not atm:
                    book["err"] = (book.get("err") or "") + " | Deribit ATM not found"
                else:
                    strike, exp, ce_n, pe_n = atm
                    book["opt_src"] = "deribit"
                    book["atm"], book["exp"] = strike, exp
                    book["ce_sym"], book["pe_sym"] = ce_n, pe_n
                    book["ce_tok"], book["pe_tok"] = ce_n, pe_n
                    idx_px = float(book.get("spot") or 0)
                    def _usd_last(iname):
                        tk = deribit_ticker(iname)
                        last = tk.get("last_price") or tk.get("mark_price") or 0
                        try:
                            last = float(last)
                        except Exception:
                            last = 0.0
                        return (last * idx_px) if last and idx_px else 0.0
                    ce_usd, pe_usd = _usd_last(ce_n), _usd_last(pe_n)
                    dce = option_usd_df(deribit_chart(ce_n, d_res), idx_px, ce_usd)
                    dpe = option_usd_df(deribit_chart(pe_n, d_res), idx_px, pe_usd)
                    book["ce_rows"] = rows_from_df(dce)
                    book["pe_rows"] = rows_from_df(dpe)
                    book["ce_sw"] = rebuild_swings(book["ce_rows"])
                    book["pe_sw"] = rebuild_swings(book["pe_rows"])
                    book["ce_1m_rows"] = book["ce_rows"] if d_res == "1" else rows_from_df(option_usd_df(deribit_chart(ce_n, "1"), idx_px, ce_usd))
                    book["pe_1m_rows"] = book["pe_rows"] if d_res == "1" else rows_from_df(option_usd_df(deribit_chart(pe_n, "1"), idx_px, pe_usd))
                    book["err"] = ""
                    loaded_opt = True
            except Exception as e:
                book["err"] = (book.get("err") or "") + f" | deribit:{e}"
        if not loaded_opt and not book.get("err"):
            book["err"] = "No option source (Binance eapi blocked; Deribit only BTC/ETH)"
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
        for side, key in (("ce", "ce_tok"), ("pe", "pe_tok")):
            iname = book.get(key)
            if not iname:
                continue
            try:
                if book.get("opt_src") == "binance":
                    usd = fetch_binance_opt_last(iname)
                else:
                    tk = deribit_ticker(iname)
                    last = tk.get("last_price") or tk.get("mark_price")
                    idx_px = float(book.get("spot") or 0)
                    usd = float(last) * idx_px if last and idx_px else None
                if usd and usd > 0:
                    apply_live_tape(book, side, usd, None, None)
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
            f"<span style='color:#90A4AE;font-size:11px;'>Binance spot/fut · Deribit opts fallback · key {'ON' if _bn_keys()[0] else 'OFF'}</span>"
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
        st.session_state["_cry_books"] = {}
        st.session_state.pop("_bn_opt_info", None)
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
