#!/usr/bin/env python3
"""
Crypto Options Technical Analysis Dashboard
BTC / ETH — Deribit public API (no keys required)

Mirrors the Indian-index Option Chain TA construct:
  GEX / Δ-GEX / VEX / CEX, IV skew, VWAP + VP, OBV / EFI / CVD,
  Superhuman regime engine, 81-state microstructure playbook,
  limit-book Δ, block tape, strategy basket Greeks.

Run:
  pip install streamlit pandas numpy plotly scipy requests pytz
  streamlit run Crypto_Options_Simulator.py
"""

from __future__ import annotations

import os
import math
import time
import json
import datetime
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as plt_go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy import stats as si

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
os.makedirs(".streamlit", exist_ok=True)
_cfg = os.path.join(".streamlit", "config.toml")
if not os.path.exists(_cfg):
    with open(_cfg, "w") as f:
        f.write(
            """[theme]
base="dark"
primaryColor="#00E676"
backgroundColor="#0E1117"
secondaryBackgroundColor="#1E222D"
textColor="#FAFAFA"
"""
        )

logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Crypto Option Chain Technical Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { opacity: 1 !important; }
[data-testid="stStatusWidget"], .stSpinner { display: none !important; }
html, body, [data-testid="stAppViewContainer"] { background-color: #0E1117 !important; color: #FAFAFA !important; }
section[data-testid="stSidebar"] { width: 280px !important; }
.block-container { padding-top: 0.25rem !important; padding-bottom: 0.25rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; height: 2.2rem !important; }
h1, h2, h3, .custom-heading { color: #00E676 !important; font-size: 15px !important; font-weight: 700 !important; margin: 0 !important; }
div[data-testid="stMetricValue"] { font-size: 13px !important; color: #00E676 !important; }
div[data-testid="stMetricLabel"] { font-size: 10px !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.2rem !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
.stMarkdown { margin-bottom: 0 !important; }
hr { margin: 0.25rem 0 !important; }
.status-badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; margin-right: 6px; }
.badge-bullish { background-color: rgba(0,230,118,0.15); color: #00E676; border: 1px solid #00E676; }
.badge-bearish { background-color: rgba(255,82,82,0.15); color: #FF5252; border: 1px solid #FF5252; }
.badge-neutral { background-color: rgba(255,152,0,0.15); color: #FF9800; border: 1px solid #FF9800; }
.sticky-summary { position: sticky; top: 0; z-index: 999; background: #0E1117; border: 1px solid #2A2F3A; border-radius: 6px; padding: 4px 8px; margin-bottom: 4px; }
.micro-hover { position: relative; display: inline-block; cursor: help; }
.micro-hover .micro-tip {
    display: none; position: absolute; left: 0; top: calc(100% + 4px);
    z-index: 5000; width: 360px; max-width: 70vw;
    background: #1A1F2B; color: #FAFAFA; border: 1px solid #00E676;
    border-radius: 8px; padding: 10px 12px; font-size: 12px; line-height: 1.45;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55);
}
.micro-hover:hover .micro-tip { display: block; }
.peco-bar {
    display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px;
    margin: 4px 0 6px 0; padding: 6px 8px;
    background: #11151C; border: 1px solid #2A3340; border-radius: 8px;
}
.peco-chip {
    flex: 1 1 140px; min-width: 140px; text-align: left;
    padding: 6px 8px; background: #1A1F2B; border: 1px solid #3A4150; border-radius: 8px;
}
.chart-card { background: #11151C; border: 1px solid #2A3340; border-radius: 8px; padding: 4px 6px 2px 6px; margin: 0 0 4px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DERIBIT_HOSTS = (
    "https://www.deribit.com/api/v2",
    "https://www.deribit.com/api/v2",
)
DERIBIT = DERIBIT_HOSTS[0]
LAST_API_ERROR = ""
TZ = pytz.timezone("UTC")
IST = pytz.timezone("Asia/Kolkata")


def to_ist(series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(IST)


def ist_label(series, with_date: bool = True) -> pd.Series:
    t = to_ist(series)
    return t.dt.strftime("%d-%b %H:%M" if with_date else "%H:%M")
CONTRACT_SIZE = {"BTC": 1.0, "ETH": 1.0}  # 1 option contract = 1 coin
PERP = {"BTC": "BTC-PERPETUAL", "ETH": "ETH-PERPETUAL"}
DVOL = {"BTC": "BTCDVOL", "ETH": "ETHDVOL"}
BIN_STEP = {"BTC": 250.0, "ETH": 25.0}

SESSION_CACHE_DIR = Path("/home/workdir/artifacts/session_cache_crypto")
try:
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    SESSION_CACHE_DIR = Path("session_cache_crypto")
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
for k, v in {
    "basket_legs": [],
    "selected_timeframe": "5 min",
    "enable_main_refresh": False,
    "liq_delta_history": [],
    "flow_tape": [],
    "heatmap_timeframe": "5 min",
    "gex_heatmap_history": [],
    "perp_window": "1 day",
    "price_alerts": [],
    "last_peco_sent": "",
    "last_peco_ts": 0.0,
    "tg_log": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

INTERVAL_MS = {
    "1 min": 60_000,
    "3 min": 180_000,
    "5 min": 300_000,
    "15 min": 900_000,
    "1 hour": 3_600_000,
}

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "crypto-gex-dashboard/1.0"})


def deribit_get(method: str, params: dict | None = None, timeout: int = 20):
    """Public GET with retry. Does not cache 503 / maintenance as success."""
    global LAST_API_ERROR
    last = None
    for attempt in range(3):
        for base in DERIBIT_HOSTS:
            url = f"{base}/{method}"
            try:
                r = _HTTP.get(url, params=params or {}, timeout=timeout)
                try:
                    js = r.json()
                except Exception:
                    last = f"HTTP {r.status_code} non-JSON"
                    continue
                err = js.get("error")
                if err:
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    LAST_API_ERROR = f"Deribit: {msg}"
                    last = LAST_API_ERROR
                    if str(msg) == "system_maintenance" or r.status_code in (503, 429):
                        time.sleep(0.6 * (attempt + 1))
                        continue
                    return None
                LAST_API_ERROR = ""
                return js.get("result")
            except Exception as e:
                last = str(e)
                continue
        time.sleep(0.4 * (attempt + 1))
    LAST_API_ERROR = last or "Deribit unreachable"
    return None


def telegram_creds():
    token = (
        os.environ.get("TELE_BOTTOKEN")
        or os.environ.get("TELE_BOT_TOKEN")
        or os.environ.get("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()
    chat = (
        os.environ.get("TELE_CHATID")
        or os.environ.get("TELE_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or ""
    ).strip()
    return token, chat


def send_telegram(text: str) -> tuple[bool, str]:
    token, chat = telegram_creds()
    if not token or not chat:
        return False, "TELE_BOTTOKEN / TELE_CHATID not set on Render"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=12,
        )
        js = {}
        try:
            js = r.json()
        except Exception:
            js = {}
        if r.status_code == 200 and js.get("ok"):
            log = list(st.session_state.get("tg_log") or [])
            log.append({"ts": datetime.datetime.now(IST).strftime("%H:%M:%S"), "ok": True, "msg": text[:80]})
            st.session_state["tg_log"] = log[-20:]
            return True, "sent"
        desc = (js.get("description") or r.text or f"HTTP {r.status_code}")[:160]
        return False, desc
    except Exception as e:
        return False, str(e)


def heading_ribbon(title: str, tip_html: str, size: int = 12):
    """Named hover ribbon — title always visible, definition expands on hover."""
    st.markdown(
        f"<div class='micro-hover' style='padding:2px 8px;background:#1A1F2B;"
        f"border:1px solid #3A4150;border-radius:6px;margin:0 0 3px 0;'>"
        f"<span style='font-weight:700;color:#00E676;font-size:{size}px;'>{title}</span>"
        f"<span style='color:#667;font-size:10px;margin-left:6px;'>hover</span>"
        f"<div class='micro-tip'>{tip_html}</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Black–Scholes / Volatility engine  (same construct as the index app)
# ---------------------------------------------------------------------------
class VolatilityEngine:
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)

    @classmethod
    def black_scholes_price(cls, S, K, T, r, sigma, flag="c") -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if flag.lower().startswith("c"):
            return S * cls._norm_cdf(d1) - K * math.exp(-r * T) * cls._norm_cdf(d2)
        return K * math.exp(-r * T) * cls._norm_cdf(-d2) - S * cls._norm_cdf(-d1)

    @classmethod
    def calculate_greeks(cls, S, K, T, r, sigma, flag="c") -> dict:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "charm": 0.0, "vanna": 0.0}
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf = cls._norm_pdf(d1)
        gamma = pdf / (S * sigma * math.sqrt(T))
        vega = (S * pdf * math.sqrt(T)) / 100.0
        vanna = -(pdf * d2) / sigma
        if flag.lower().startswith("c"):
            delta = cls._norm_cdf(d1)
            theta = (-(S * pdf * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls._norm_cdf(d2)) / 365.0
            charm = (pdf * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))) / 365.0
        else:
            delta = cls._norm_cdf(d1) - 1.0
            theta = (-(S * pdf * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls._norm_cdf(-d2)) / 365.0
            charm = (-pdf * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))) / 365.0
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "charm": charm, "vanna": vanna}

    @classmethod
    def calculate_iv(cls, market_price, S, K, T, r=0.0, flag="c") -> float:
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return 0.0
        df = math.exp(-r * T)
        intrinsic = max(0.0, S - K * df) if flag.lower().startswith("c") else max(0.0, K * df - S)
        if market_price <= intrinsic * 0.999:
            return 0.0

        def obj(sig):
            return cls.black_scholes_price(S, K, T, r, sig, flag) - market_price

        try:
            return float(brentq(obj, 1e-4, 8.0, xtol=1e-4))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Playbooks (identical state machines)
# ---------------------------------------------------------------------------
MICRO_PLAYBOOK = {
    ("u", "u", "u", "u"): ("Pure Institutional Aggression", "LONG BREAKOUT (Target: Upper 2.0σ)"),
    ("u", "u", "u", "f"): ("Thin-Book Impulse", "CAUTIOUS LONG (Trailing SL tight)"),
    ("u", "u", "u", "d"): ("Low-Volume Markup", "CAUTIOUS LONG / PREPARE TRAIL"),
    ("u", "u", "f", "u"): ("Passive Wall Sweeping", "HOLD LONG"),
    ("u", "u", "f", "f"): ("Steady Markup", "HOLD LONG"),
    ("u", "u", "f", "d"): ("Low-Volume Steady Drift", "NO ENTRY (Unstable move)"),
    ("u", "u", "d", "u"): ("Bullish Passive Limit Accumulation", "HOLD LONG (Institutional support)"),
    ("u", "u", "d", "f"): ("Absorbed Drift", "HOLD LONG"),
    ("u", "u", "d", "d"): ("Illiquid Short Squeeze", "NO ENTRY (High reversal risk)"),
    ("u", "f", "u", "u"): ("High-Volume Choke", "WARNING: Limit sell wall building"),
    ("u", "f", "u", "f"): ("Mild Buying Friction", "HOLD LONG / NO NEW ENTRIES"),
    ("u", "f", "u", "d"): ("Fading Buying Effort", "BLOCK LONG ENTRIES"),
    ("u", "f", "f", "u"): ("High Volume Neutral Drift", "HOLD LONG"),
    ("u", "f", "f", "f"): ("Low-Volatility Up-Drift", "HOLD EXISTING POSITIONS"),
    ("u", "f", "f", "d"): ("Volume Drying Upbeat", "PREPARE EXIT LONG"),
    ("u", "f", "d", "u"): ("Distribution under Cover", "EXIT LONG / PREPARE SHORT"),
    ("u", "f", "d", "f"): ("Passive Seller Pressure", "EXIT LONG"),
    ("u", "f", "d", "d"): ("Diverging Drift", "EXIT LONG"),
    ("u", "d", "u", "u"): ("Institutional Supply Absorption", "EXIT LONG / DYNAMIC ABSORPTION EXIT"),
    ("u", "d", "u", "f"): ("Buying Force Decay", "EXIT LONG"),
    ("u", "d", "u", "d"): ("Fading Bullish Push", "EXIT LONG"),
    ("u", "d", "f", "u"): ("High Volume Momentum Trap", "PREPARE SHORT"),
    ("u", "d", "f", "f"): ("Momentum Exhaustion", "EXIT LONG"),
    ("u", "d", "f", "d"): ("Low-Volume Top Building", "PREPARE SHORT"),
    ("u", "d", "d", "u"): ("Classic Bearish Institutional Distribution", "SHORT MEAN-REVERSION (At Upper Band)"),
    ("u", "d", "d", "f"): ("Bearish Divergence (Standard)", "SHORT MEAN-REVERSION"),
    ("u", "d", "d", "d"): ("Triple Bearish Divergence", "SHORT MEAN-REVERSION / SHORT ENTRY"),
    ("f", "u", "u", "u"): ("Coil Compression (Bullish Push)", "PRE-BREAKOUT LONG PREPARATION"),
    ("f", "u", "u", "f"): ("Hidden Aggressive Buying", "PRE-BREAKOUT LONG"),
    ("f", "u", "u", "d"): ("Selective Aggressive Buying", "WATCH FOR BREAKOUT"),
    ("f", "u", "f", "u"): ("Force Expansion in Consolidation", "WATCH FOR BREAKOUT"),
    ("f", "u", "f", "f"): ("Quiet Force Building", "NO ENTRY / NO EDGE"),
    ("f", "u", "f", "d"): ("Low-Volume Force Shift", "NO ENTRY"),
    ("f", "u", "d", "u"): ("Passive Limit Floor Building", "BULLISH ABSORPTION WATCH"),
    ("f", "u", "d", "f"): ("Mild Passive Support", "NO ENTRY"),
    ("f", "u", "d", "d"): ("Conflicted Consolidation", "NO ENTRY"),
    ("f", "f", "u", "u"): ("Institutional Accumulation Box", "ACCUMULATION WATCH"),
    ("f", "f", "u", "f"): ("Quiet Delta Accumulation", "ACCUMULATION WATCH"),
    ("f", "f", "u", "d"): ("Low-Volume Delta Push", "NO ENTRY"),
    ("f", "f", "f", "u"): ("High-Volume Equilibrium", "ORDER BOOK REBALANCING"),
    ("f", "f", "f", "f"): ("DEAD MARKET / CHOP", "NO ENTRY (BLOCK ALL SIGNALS)"),
    ("f", "f", "f", "d"): ("Liquidity Drying Up", "NO ENTRY"),
    ("f", "f", "d", "u"): ("Institutional Distribution Box", "DISTRIBUTION WATCH"),
    ("f", "f", "d", "f"): ("Quiet Delta Distribution", "DISTRIBUTION WATCH"),
    ("f", "f", "d", "d"): ("Low-Volume Slippage", "NO ENTRY"),
    ("f", "d", "u", "u"): ("Passive Limit Wall Blocking Force", "NO ENTRY"),
    ("f", "d", "u", "f"): ("Fading Buying Impulse in Range", "NO ENTRY"),
    ("f", "d", "u", "d"): ("Low-Volume Friction", "NO ENTRY"),
    ("f", "d", "f", "u"): ("High-Volume Bearish Force", "WATCH FOR BREAKDOWN"),
    ("f", "d", "f", "f"): ("Quiet Force Decay", "NO ENTRY"),
    ("f", "d", "f", "d"): ("Low-Volume Force Breakdown", "NO ENTRY"),
    ("f", "d", "d", "u"): ("Coil Compression (Bearish Push)", "PRE-BREAKDOWN SHORT PREPARATION"),
    ("f", "d", "d", "f"): ("Hidden Aggressive Selling", "PRE-BREAKDOWN SHORT"),
    ("f", "d", "d", "d"): ("Triple Bearish Compression", "SHORT BREAKDOWN PREPARATION"),
    ("d", "u", "u", "u"): ("Triple Bullish Divergence", "LONG MEAN-REVERSION / LONG ENTRY"),
    ("d", "u", "u", "f"): ("Bullish Divergence (Standard)", "LONG MEAN-REVERSION"),
    ("d", "u", "u", "d"): ("Classic Bullish Institutional Accumulation", "LONG MEAN-REVERSION (At Lower Band)"),
    ("d", "u", "f", "u"): ("High-Volume Bottom Building", "PREPARE LONG"),
    ("d", "u", "f", "f"): ("Momentum Floor", "EXIT SHORT"),
    ("d", "u", "f", "d"): ("Low-Volume Bottom Building", "PREPARE LONG"),
    ("d", "u", "d", "u"): ("Institutional Demand Absorption", "EXIT SHORT / DYNAMIC ABSORPTION EXIT"),
    ("d", "u", "d", "f"): ("Selling Force Decay", "EXIT SHORT"),
    ("d", "u", "d", "d"): ("Fading Bearish Push", "EXIT SHORT"),
    ("d", "f", "u", "u"): ("Accumulation under Cover", "EXIT SHORT / PREPARE LONG"),
    ("d", "f", "u", "f"): ("Passive Buyer Pressure", "EXIT SHORT"),
    ("d", "f", "u", "d"): ("Diverging Down-Drift", "EXIT SHORT"),
    ("d", "f", "f", "u"): ("High Volume Neutral Down-Drift", "HOLD SHORT"),
    ("d", "f", "f", "f"): ("Low-Volatility Down-Drift", "HOLD EXISTING POSITIONS"),
    ("d", "f", "f", "d"): ("Volume Drying Downbeat", "PREPARE EXIT SHORT"),
    ("d", "f", "d", "u"): ("High-Volume Friction", "WARNING: Limit buy wall building"),
    ("d", "f", "d", "f"): ("Mild Selling Friction", "HOLD SHORT / NO NEW ENTRIES"),
    ("d", "f", "d", "d"): ("Fading Selling Effort", "BLOCK SHORT ENTRIES"),
    ("d", "d", "u", "u"): ("Bullish Passive Limit Absorption", "HOLD SHORT (Institutional resistance)"),
    ("d", "d", "u", "f"): ("Absorbed Down-Drift", "HOLD SHORT"),
    ("d", "d", "u", "d"): ("Illiquid Long Squeeze", "NO ENTRY (High reversal risk)"),
    ("d", "d", "f", "u"): ("Passive Wall Sweeping Down", "HOLD SHORT"),
    ("d", "d", "f", "f"): ("Steady Markdown", "HOLD SHORT"),
    ("d", "d", "f", "d"): ("Low-Volume Steady Down-Drift", "NO ENTRY (Unstable move)"),
    ("d", "d", "d", "u"): ("Low-Volume Markdown", "CAUTIOUS SHORT / PREPARE TRAIL"),
    ("d", "d", "d", "f"): ("Thin-Book Down-Impulse", "CAUTIOUS SHORT (Trailing SL tight)"),
    ("d", "d", "d", "d"): ("Pure Institutional Aggression (Bearish)", "SHORT BREAKOUT (Target: Lower 2.0σ)"),
}
EFI_ZERO_PLAYBOOK = {
    ("f", "u", "u"): ("Supply Absorption Trap", "EXIT LONG / PREPARE SHORT"),
    ("f", "u", "f"): ("Aggressive Buyer Friction", "WARNING: No New Longs"),
    ("f", "u", "d"): ("Fading Buying Effort", "BLOCK LONG ENTRIES"),
    ("f", "f", "u"): ("Passive Wall Sweeping", "NEUTRAL WATCH"),
    ("f", "f", "f"): ("Pure Market Stagnation", "NO ENTRY / CHOP"),
    ("f", "f", "d"): ("Volume Drying Neutral", "NO ENTRY"),
    ("f", "d", "u"): ("Covert Distribution", "EXIT LONG / PREPARE SHORT"),
    ("f", "d", "f"): ("Aggressive Seller Friction", "WARNING: No New Shorts"),
    ("f", "d", "d"): ("Demand Absorption Floor", "EXIT SHORT / PREPARE LONG"),
    ("u", "u", "u"): ("Stealth Mark-Up", "CAUTIOUS LONG"),
    ("u", "u", "f"): ("Delta-Driven Grind", "HOLD EXISTING LONGS"),
    ("u", "u", "d"): ("Exhaustion Up-Drift", "TAKE PROFITS"),
    ("u", "f", "u"): ("Passive Ask Pull", "HOLD LONGS"),
    ("u", "f", "f"): ("Low-Volatility Up-Drift", "HOLD EXISTING LONGS"),
    ("u", "f", "d"): ("Illiquid Drift Up", "PREPARE EXIT LONG"),
    ("u", "d", "u"): ("Distribution Under Cover", "EXIT LONG / PREPARE SHORT"),
    ("u", "d", "f"): ("Passive Seller Pressure", "EXIT LONG"),
    ("u", "d", "d"): ("Bearish Diverging Drift", "EXIT LONG / PREPARE SHORT"),
    ("d", "u", "u"): ("Bullish Diverging Slide", "EXIT SHORT / PREPARE LONG"),
    ("d", "u", "f"): ("Passive Absorption Bleed", "WARNING: Downside stalling"),
    ("d", "u", "d"): ("Fading Downside Effort", "EXIT SHORT"),
    ("d", "f", "u"): ("Passive Bid Removal", "HOLD SHORTS"),
    ("d", "f", "f"): ("Low-Volatility Bleed", "HOLD EXISTING SHORTS"),
    ("d", "f", "d"): ("Illiquid Drift Down", "HOLD SHORTS"),
    ("d", "d", "u"): ("Aggressive Mark-Down Friction", "HOLD SHORTS"),
    ("d", "d", "f"): ("Stealth Mark-Down", "HOLD EXISTING SHORTS"),
    ("d", "d", "d"): ("Clean Passive Mark-Down", "HOLD SHORTS (Target −2.0σ VWAP)"),
}
FLOW_PLAYBOOK = {
    ("u", "u", "u", "u"): ("Pure Institutional Call Sweep", "LONG BREAKOUT"),
    ("u", "f", "u", "f"): ("Gamma Squeeze Drift", "HOLD LONG"),
    ("u", "u", "d", "d"): ("Distribution under Cover", "EXIT LONG / PREPARE SHORT"),
    ("u", "d", "d", "d"): ("Institutional Call Liquidation", "PREPARE SHORT"),
    ("f", "u", "u", "u"): ("Coil Compression (Bullish)", "PRE-BREAKOUT LONG"),
    ("f", "f", "f", "f"): ("Gamma Pin / Quiet Regime", "NO ENTRY / SELL NEUTRAL STRADDLE"),
    ("f", "d", "d", "u"): ("Institutional Distribution Box", "PRE-BREAKDOWN SHORT"),
    ("d", "d", "d", "d"): ("Pure Institutional Markdown", "SHORT BREAKOUT"),
    ("d", "f", "d", "f"): ("Short Gamma Drag", "HOLD SHORT"),
    ("d", "d", "u", "u"): ("Institutional Demand Absorption", "EXIT SHORT / PREPARE LONG"),
    ("d", "u", "u", "d"): ("Put Shorting / Floor Building", "PREPARE LONG"),
    ("d", "f", "f", "f"): ("Low-Volume Slippage", "NO ENTRY"),
    ("u", "d", "u", "u"): ("Short Gamma Squeeze", "RIDE LONG"),
    ("d", "u", "d", "u"): ("Short Gamma Unwind", "RIDE SHORT"),
    ("u", "u", "f", "d"): ("Perp-Driven Momentum", "CAUTIOUS LONG"),
    ("u", "d", "u", "d"): ("Short Covering Ramp", "HOLD LONG"),
    ("d", "u", "d", "d"): ("Long Unwinding Bleed", "HOLD SHORT"),
    ("f", "u", "d", "u"): ("Cross / Strangle Creation", "STAND ASIDE (Vol expansion)"),
}
ARROW_GLYPH = {"u": "↑", "f": "→", "d": "↓"}


def _robust_arrow(series: pd.Series, z_th: float = 0.90, ema_span: int = 8) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return "f"
    last = float(s.iloc[-1])
    ema = float(s.ewm(span=ema_span, adjust=False).mean().iloc[-1])
    chg = s.diff().dropna()
    if len(chg) < 6:
        return "f"
    sd = float(chg.tail(20).std(ddof=1) or 0.0)
    if sd <= 1e-12:
        return "f"
    z = float(chg.iloc[-1]) / sd
    lvl_sd = float(s.tail(20).std(ddof=1) or sd)
    z_lvl = (last - float(s.tail(20).mean())) / max(lvl_sd, 1e-12)
    score = 0.65 * z + 0.35 * z_lvl
    if last > ema and score >= z_th:
        return "u"
    if last < ema and score <= -z_th:
        return "d"
    return "f"


def classify_microstructure(dfi: pd.DataFrame) -> dict:
    empty = {"ok": False, "arrows": "→ → → →", "micro": "", "action": "NO ENTRY",
             "key": ("f", "f", "f", "f"), "hover": "", "price": "f", "efi": "f", "cvd": "f", "obv": "f",
             "efi_zero": False, "efi_note": ""}
    if dfi is None or dfi.empty:
        return empty
    d = dfi.copy()
    price_s = d["close"].astype(float)
    efi_s = d["efi13"] if "efi13" in d.columns else pd.Series(dtype=float)
    cvd_s = d["cvd"] if "cvd" in d.columns else pd.Series(dtype=float)
    obv_s = d["obv"] if "obv" in d.columns else pd.Series(dtype=float)
    p_arr = _robust_arrow(price_s, z_th=0.95)
    if "vwap" in d.columns and len(price_s):
        last_px = float(price_s.iloc[-1])
        last_vw = float(d["vwap"].iloc[-1])
        if p_arr == "u" and last_px <= last_vw:
            p_arr = "f"
        if p_arr == "d" and last_px >= last_vw:
            p_arr = "f"
    e_arr = _robust_arrow(efi_s, z_th=0.85) if len(efi_s) else "f"
    c_arr = _robust_arrow(cvd_s, z_th=0.85) if len(cvd_s) else "f"
    o_arr = _robust_arrow(obv_s, z_th=0.85) if len(obv_s) else "f"
    key = (p_arr, e_arr, c_arr, o_arr)
    efi_zero = False
    efi_note = ""
    if len(efi_s) >= 12:
        ev = pd.to_numeric(efi_s, errors="coerce").dropna()
        last_e = float(ev.iloc[-1])
        sd_e = float(ev.tail(20).std(ddof=1) or 0.0)
        if sd_e > 0 and abs(last_e) <= 0.40 * sd_e and e_arr == "f":
            efi_zero = True
            e_arr = "f"
            key = (p_arr, "f", c_arr, o_arr)
            zkey = (p_arr, c_arr, o_arr)
            micro, action = EFI_ZERO_PLAYBOOK.get(zkey, MICRO_PLAYBOOK.get(key, ("Unclassified", "NO ENTRY")))
            efi_note = f"|EFI| {last_e:.0f} ≤ 0.40σ → 27-state EFI≈0 book"
        else:
            micro, action = MICRO_PLAYBOOK.get(key, ("Unclassified tape", "NO ENTRY"))
            efi_note = f"|EFI| {last_e:.0f} vs 0.40σ={0.40 * sd_e:.0f} → 81-state book"
    else:
        micro, action = MICRO_PLAYBOOK.get(key, ("Unclassified tape", "NO ENTRY"))
    glyphs = " ".join(ARROW_GLYPH[k] for k in key)
    hover = (
        f"Perp {ARROW_GLYPH[p_arr]}  EFI {ARROW_GLYPH[e_arr]}  "
        f"CVD {ARROW_GLYPH[c_arr]}  OBV {ARROW_GLYPH[o_arr]}<br>"
        f"<b>{action}</b><br>{micro}<br>{efi_note}"
    )
    return {
        "ok": True, "key": key, "arrows": glyphs, "micro": micro, "action": action,
        "price": p_arr, "efi": e_arr, "cvd": c_arr, "obv": o_arr, "hover": hover,
        "efi_zero": efi_zero, "efi_note": efi_note,
    }


# ---------------------------------------------------------------------------
# Volume profile / indicators
# ---------------------------------------------------------------------------
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 5:
        return df if df is not None else pd.DataFrame()
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.sort_values("time").reset_index(drop=True)
    d["bb_middle"] = d["close"].rolling(20, min_periods=5).mean()
    d["bb_std"] = d["close"].rolling(20, min_periods=5).std()
    d["bb_upper"] = d["bb_middle"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_middle"] - 2 * d["bb_std"]
    d["bb_bandwidth"] = np.where(d["bb_middle"] > 0, (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"], 0)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=5, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=5, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    d["date_group"] = d["time"].dt.date
    d["tp"] = (d["high"] + d["low"] + d["close"]) / 3.0
    d["tp_vol"] = d["tp"] * d["volume"]
    d["cum_vol"] = d.groupby("date_group")["volume"].cumsum()
    d["cum_tp_vol"] = d.groupby("date_group")["tp_vol"].cumsum()
    raw = np.where(d["cum_vol"] > 0, d["cum_tp_vol"] / d["cum_vol"], np.nan)
    d["vwap"] = pd.Series(raw, index=d.index).ffill().fillna(d["close"])
    return d


def _prominence_nodes(vol_at, mids, prominence_factor=0.35):
    n = len(vol_at)
    if n < 3:
        return [], []
    vmax = float(np.max(vol_at)) or 1.0
    hvn_idx = [i for i in range(1, n - 1)
               if vol_at[i] >= vol_at[i - 1] and vol_at[i] >= vol_at[i + 1] and vol_at[i] >= 0.20 * vmax]
    if not hvn_idx:
        hvn_idx = [int(np.argmax(vol_at))]
    inverted = vmax - vol_at
    lvn_idx = []
    for i in range(1, n - 1):
        if not (inverted[i] >= inverted[i - 1] and inverted[i] >= inverted[i + 1]):
            continue
        left = [j for j in hvn_idx if j < i]
        right = [j for j in hvn_idx if j > i]
        if not left or not right:
            continue
        surround = min(float(vol_at[left[-1]]), float(vol_at[right[0]]))
        if surround > 0 and (surround - float(vol_at[i])) >= prominence_factor * surround:
            lvn_idx.append(i)
    return [float(mids[i]) for i in hvn_idx], [float(mids[i]) for i in lvn_idx]


def compute_session_volume_profile(df, bin_step=250.0, prominence_factor=0.35) -> dict:
    empty = {"ok": False}
    if df is None or df.empty or "volume" not in df.columns:
        return empty
    lo = float(min(df["low"].min(), df["close"].min()))
    hi = float(max(df["high"].max(), df["close"].max()))
    if hi <= lo:
        hi = lo + bin_step
    lo = np.floor(lo / bin_step) * bin_step
    hi = np.ceil(hi / bin_step) * bin_step
    edges = np.arange(lo, hi + bin_step * 0.5, bin_step)
    if len(edges) < 3:
        edges = np.array([lo, lo + bin_step, lo + 2 * bin_step])
    n_bins = len(edges) - 1
    vol_at = np.zeros(n_bins)
    for _, r in df.iterrows():
        v = float(r.get("volume") or 0)
        if v <= 0:
            continue
        l, h = float(r["low"]), float(r["high"])
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
    hvn, lvn = _prominence_nodes(vol_at, mids, prominence_factor)
    return {
        "ok": True, "mids": mids, "vol": vol_at, "poc": float(mids[poc_i]),
        "poc_vol": float(vol_at[poc_i]), "wmean": wmean, "wstd": wstd,
        "vah1": wmean + wstd, "val1": wmean - wstd,
        "vah15": wmean + 1.5 * wstd, "val15": wmean - 1.5 * wstd,
        "hvn": hvn, "lvn": lvn, "bin_step": bin_step,
    }


def calculate_max_pain(chain: list) -> float:
    strikes = [row["Strike"] for row in chain]
    if not strikes:
        return 0
    losses = {}
    for hypo in strikes:
        tot = 0.0
        for row in chain:
            k = row["Strike"]
            if hypo > k:
                tot += (hypo - k) * row["C_OI"]
            if hypo < k:
                tot += (k - hypo) * row["P_OI"]
        losses[hypo] = tot
    return min(losses, key=losses.get)


def calculate_support_resistance_targets(chain, spot, max_pain):
    if not chain:
        return {}
    df = pd.DataFrame(chain)
    res_df = df.sort_values("C_OI", ascending=False)
    r1 = res_df.iloc[0]["Strike"] if len(res_df) else spot
    r2 = res_df.iloc[1]["Strike"] if len(res_df) > 1 else r1
    sup_df = df.sort_values("P_OI", ascending=False)
    s1 = sup_df.iloc[0]["Strike"] if len(sup_df) else spot
    s2 = sup_df.iloc[1]["Strike"] if len(sup_df) > 1 else s1
    df_sorted = df.sort_values("Strike").reset_index(drop=True)
    above = df_sorted[df_sorted["Strike"] >= spot]
    below = df_sorted[df_sorted["Strike"] < spot]
    gex_res_row = above.sort_values("Net_GEX_OI", ascending=False).head(1) if not above.empty else pd.DataFrame()
    gex_resistance = gex_res_row.iloc[0]["Strike"] if not gex_res_row.empty else r1
    gex_sup_row = below.sort_values("Net_GEX_OI", ascending=False).head(1) if not below.empty else pd.DataFrame()
    gex_support = gex_sup_row.iloc[0]["Strike"] if not gex_sup_row.empty else s1
    neg = df_sorted.sort_values("Net_GEX_OI", ascending=True).head(1)
    gex_acc = neg.iloc[0]["Strike"] if not neg.empty and neg.iloc[0]["Net_GEX_OI"] < 0 else None
    flips = []
    for i in range(1, len(df_sorted)):
        prev_g, curr_g = df_sorted.loc[i - 1, "Net_GEX_OI"], df_sorted.loc[i, "Net_GEX_OI"]
        if (prev_g < 0 <= curr_g) or (prev_g >= 0 > curr_g):
            chosen = df_sorted.loc[i, "Strike"] if abs(curr_g) < abs(prev_g) else df_sorted.loc[i - 1, "Strike"]
            flips.append(chosen)
    flip = min(flips, key=lambda x: abs(x - spot)) if flips else spot
    atm = df.iloc[(df["Strike"] - spot).abs().argsort()[:1]].iloc[0]
    straddle = float(atm["C_LTP"]) + float(atm["P_LTP"])
    return {
        "S1": s1, "S2": s2, "R1": r1, "R2": r2,
        "GEX_Support": gex_support, "GEX_Resistance": gex_resistance,
        "GEX_Accelerator": gex_acc, "Zero_Gamma_Flip": flip,
        "MaxPain": max_pain,
        "Target_Up": round(spot + straddle, 2),
        "Target_Down": round(spot - straddle, 2),
        "Straddle_Cost": round(straddle, 4),
    }


# ---------------------------------------------------------------------------
# Deribit data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=180, show_spinner=False)
def _fetch_instruments_ok(currency: str) -> pd.DataFrame:
    res = deribit_get("public/get_instruments", {"currency": currency, "kind": "option", "expired": "false"})
    if not res:
        raise RuntimeError(LAST_API_ERROR or "no instruments")
    df = pd.DataFrame(res)
    if df.empty:
        raise RuntimeError("empty instrument list")
    df["expiry_dt"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["opt_type"] = df["option_type"].str.lower().map({"call": "C", "put": "P"})
    return df


def fetch_instruments(currency: str) -> pd.DataFrame:
    try:
        return _fetch_instruments_ok(currency)
    except Exception:
        try:
            _fetch_instruments_ok.clear()
        except Exception:
            pass
        return pd.DataFrame()


def _coingecko_spot(currency: str) -> float:
    ids = {"BTC": "bitcoin", "ETH": "ethereum"}
    try:
        r = _HTTP.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ids.get(currency, "bitcoin"), "vs_currencies": "usd"},
            timeout=10,
        )
        js = r.json()
        return float(js[ids[currency]]["usd"])
    except Exception:
        return 0.0


@st.cache_data(ttl=15, show_spinner=False)
def fetch_index_price(currency: str) -> float:
    res = deribit_get("public/get_index_price", {"index_name": f"{currency.lower()}_usd"})
    if res and "index_price" in res:
        return float(res["index_price"])
    t = deribit_get("public/ticker", {"instrument_name": PERP[currency]})
    if t:
        return float(t.get("index_price") or t.get("last_price") or 0)
    return _coingecko_spot(currency)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_book_summaries(currency: str, kind: str = "option") -> list:
    res = deribit_get("public/get_book_summary_by_currency", {"currency": currency, "kind": kind})
    return res or []


@st.cache_data(ttl=20, show_spinner=False)
def fetch_candles(instrument: str, resolution_ms: int, lookback_hours: int = 72) -> pd.DataFrame:
    now = int(time.time() * 1000)
    start = now - lookback_hours * 3600 * 1000
    res = deribit_get(
        "public/get_tradingview_chart_data",
        {
            "instrument_name": instrument,
            "start_timestamp": start,
            "end_timestamp": now,
            "resolution": str(max(int(resolution_ms / 60_000), 1)),
        },
    )
    if not res or res.get("status") not in ("ok", "OK", None) and not res.get("ticks"):
        # some responses wrap under status=ok
        pass
    ticks = (res or {}).get("ticks") or []
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(res["ticks"], unit="ms", utc=True),
            "open": res.get("open"),
            "high": res.get("high"),
            "low": res.get("low"),
            "close": res.get("close"),
            "volume": res.get("volume"),
        }
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return compute_technical_indicators(df)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_dvol_candles(currency: str, lookback_hours: int = 72) -> pd.DataFrame:
    # DVOL is an index; use index chart if available, else skip
    name = f"{currency}_DVOL"  # not a tradeable instrument on all endpoints
    # Fallback: public/get_volatility_index_data
    now = int(time.time() * 1000)
    start = now - lookback_hours * 3600 * 1000
    res = deribit_get(
        "public/get_volatility_index_data",
        {
            "currency": currency,
            "start_timestamp": start,
            "end_timestamp": now,
            "resolution": "60",
        },
    )
    if not res:
        return pd.DataFrame()
    data = res.get("data") if isinstance(res, dict) else res
    if not data:
        return pd.DataFrame()
    # [timestamp, open, high, low, close]
    rows = []
    for row in data:
        if isinstance(row, (list, tuple)) and len(row) >= 5:
            rows.append({"time": pd.to_datetime(row[0], unit="ms", utc=True), "close": float(row[4])})
        elif isinstance(row, dict):
            rows.append({"time": pd.to_datetime(row.get("timestamp"), unit="ms", utc=True), "close": float(row.get("close", 0))})
    return pd.DataFrame(rows)


def fetch_order_book(instrument: str, depth: int = 20) -> dict:
    res = deribit_get("public/get_order_book", {"instrument_name": instrument, "depth": depth})
    return res or {}


def fetch_last_trades(instrument: str, count: int = 50) -> list:
    res = deribit_get("public/get_last_trades_by_instrument", {"instrument_name": instrument, "count": count})
    if not res:
        return []
    return res.get("trades") or []


def coin_price_from_option(mark_or_last: float, index_price: float) -> float:
    """Deribit option prices are in the coin (BTC/ETH). Convert to USD."""
    if mark_or_last is None:
        return 0.0
    return float(mark_or_last) * float(index_price)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------
def build_chain(currency: str, expiry_ts: int, spot: float, r: float, strikes_below: int, strikes_above: int):
    inst = fetch_instruments(currency)
    if inst.empty:
        return [], 0.0, 0.0
    exp_df = inst[inst["expiration_timestamp"] == expiry_ts].copy()
    if exp_df.empty:
        return [], 0.0, 0.0

    summaries = {s["instrument_name"]: s for s in fetch_book_summaries(currency, "option")}
    now = datetime.datetime.now(datetime.timezone.utc)
    exp_dt = datetime.datetime.fromtimestamp(expiry_ts / 1000.0, tz=datetime.timezone.utc)
    T = max((exp_dt - now).total_seconds() / (365.0 * 24 * 3600), 1e-6)

    strikes = sorted(exp_df["strike"].dropna().unique().tolist())
    atm = min(strikes, key=lambda x: abs(x - spot)) if strikes else spot
    atm_idx = strikes.index(atm)
    window = strikes[max(0, atm_idx - strikes_below): min(len(strikes), atm_idx + strikes_above + 1)]

    lot = CONTRACT_SIZE.get(currency, 1.0)
    by_strike = {}
    for _, row in exp_df.iterrows():
        k = float(row["strike"])
        if k not in window:
            continue
        name = row["instrument_name"]
        sm = summaries.get(name, {})
        flag = "c" if row["opt_type"] == "C" else "p"
        mark_coin = float(sm.get("mark_price") or sm.get("last") or sm.get("mid_price") or 0)
        last_coin = float(sm.get("last") or mark_coin)
        bid_coin = float(sm.get("bid_price") or 0)
        ask_coin = float(sm.get("ask_price") or 0)
        oi = float(sm.get("open_interest") or 0)
        vol = float(sm.get("volume") or 0)
        usd = coin_price_from_option(mark_coin if mark_coin > 0 else last_coin, spot)
        mark_iv = sm.get("mark_iv")
        if mark_iv is not None:
            try:
                iv = float(mark_iv) / 100.0 if float(mark_iv) > 2 else float(mark_iv)
            except Exception:
                iv = 0.0
        else:
            iv = VolatilityEngine.calculate_iv(usd, spot, k, T, r, flag)
        if iv <= 0:
            iv = VolatilityEngine.calculate_iv(usd, spot, k, T, r, flag)
        g = VolatilityEngine.calculate_greeks(spot, k, T, r, max(iv, 1e-4), flag)
        rec = by_strike.setdefault(k, {
            "Strike": k,
            "C_LTP": 0.0, "P_LTP": 0.0,
            "C_OI": 0.0, "P_OI": 0.0,
            "C_Vol": 0.0, "P_Vol": 0.0,
            "C_IV": 0.0, "P_IV": 0.0,
            "C_Δ": 0.0, "P_Δ": 0.0,
            "C_γ": 0.0, "P_γ": 0.0,
            "C_Bid": 0.0, "C_Ask": 0.0, "P_Bid": 0.0, "P_Ask": 0.0,
            "C_Name": "", "P_Name": "",
        })
        if flag == "c":
            rec.update({
                "C_LTP": usd, "C_OI": oi, "C_Vol": vol, "C_IV": iv * 100.0,
                "C_Δ": g["delta"], "C_γ": g["gamma"],
                "C_Bid": coin_price_from_option(bid_coin, spot),
                "C_Ask": coin_price_from_option(ask_coin, spot),
                "C_Name": name, "C_theta": g["theta"], "C_vega": g["vega"],
                "C_vanna": g["vanna"], "C_charm": g["charm"],
            })
        else:
            rec.update({
                "P_LTP": usd, "P_OI": oi, "P_Vol": vol, "P_IV": iv * 100.0,
                "P_Δ": g["delta"], "P_γ": g["gamma"],
                "P_Bid": coin_price_from_option(bid_coin, spot),
                "P_Ask": coin_price_from_option(ask_coin, spot),
                "P_Name": name, "P_theta": g["theta"], "P_vega": g["vega"],
                "P_vanna": g["vanna"], "P_charm": g["charm"],
            })

    chain = []
    total_call_oi = total_put_oi = 0.0
    total_net_gex_oi = total_net_gex_vol = 0.0
    ivs = []
    for k in sorted(by_strike):
        r0 = by_strike[k]
        # Dealer GEX convention: call gamma long dealers if they are short calls? Standard retail GEX:
        # Net GEX ≈ (Call_γ * Call_OI − Put_γ * Put_OI) * S^2 * 0.01 * contract
        gex_oi = (r0["C_γ"] * r0["C_OI"] - r0["P_γ"] * r0["P_OI"]) * lot * (spot ** 2) * 0.01
        gex_vol = (r0["C_γ"] * r0["C_Vol"] - r0["P_γ"] * r0["P_Vol"]) * lot * (spot ** 2) * 0.01
        d_gex = gex_oi * max(abs(r0["C_Δ"]) + abs(r0["P_Δ"]), 0.15)
        dex_oi = (float(r0["C_Δ"]) * float(r0["C_OI"]) + float(r0["P_Δ"]) * float(r0["P_OI"])) * lot * spot
        vex = ((r0.get("C_vanna", 0) or 0) * r0["C_OI"] + (r0.get("P_vanna", 0) or 0) * r0["P_OI"]) * lot
        cex = ((r0.get("C_charm", 0) or 0) * r0["C_OI"] + (r0.get("P_charm", 0) or 0) * r0["P_OI"]) * lot
        prem_c = float(r0["C_LTP"]) * float(r0["C_OI"]) * lot
        prem_p = float(r0["P_LTP"]) * float(r0["P_OI"]) * lot
        r0.update({
            "Net_GEX_OI": gex_oi, "Net_GEX_Vol": gex_vol,
            "Net_Delta_GEX_OI": d_gex, "DEX_OI": dex_oi,
            "VEX": vex, "CEX": cex, "Prem_C": prem_c, "Prem_P": prem_p,
        })
        chain.append(r0)
        total_call_oi += r0["C_OI"]
        total_put_oi += r0["P_OI"]
        total_net_gex_oi += gex_oi
        total_net_gex_vol += gex_vol
        if r0["C_IV"] > 1:
            ivs.append(r0["C_IV"])
        if r0["P_IV"] > 1:
            ivs.append(r0["P_IV"])

    return chain, T, float(np.median(ivs) if ivs else 50.0), total_call_oi, total_put_oi, total_net_gex_oi, total_net_gex_vol


def hv_from_candles(df: pd.DataFrame, days: int = 30) -> float:
    if df is None or df.empty or "close" not in df.columns:
        return 0.55
    # resample daily
    d = df.copy()
    d = d.set_index("time")["close"].resample("1D").last().dropna()
    if len(d) < 6:
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(rets) < 5:
            return 0.55
        # scale 5-min to annual
        bars_day = 288
        return float(rets.std(ddof=1) * np.sqrt(bars_day * 365))
    lr = np.log(d / d.shift(1)).dropna()
    return float(lr.tail(days).std(ddof=1) * np.sqrt(365)) if len(lr) >= 5 else 0.55


# ---------------------------------------------------------------------------
# Superhuman + directional trigger (same scoring weights)
# ---------------------------------------------------------------------------
def evaluate_directional_trigger(data, df_candles):
    levels = data.get("levels", {}) or {}
    spot = float(data.get("spot_price") or 0)
    gex_sup = float(levels.get("GEX_Support") or spot or 0)
    gex_res = float(levels.get("GEX_Resistance") or spot or 0)
    checks = []
    px = vwap = obv = obv_ma = cvd = efi = None
    lvn = []
    df = pd.DataFrame()
    if df_candles is not None and not getattr(df_candles, "empty", True):
        df = df_candles.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        if "close" in df.columns:
            px = float(df["close"].iloc[-1])
            if "volume" in df.columns:
                tp = (df["high"] + df["low"] + df["close"]) / 3.0 if {"high", "low"}.issubset(df.columns) else df["close"]
                vol = df["volume"].astype(float)
                cum_v = vol.cumsum().replace(0, np.nan)
                vwap = float((tp.astype(float) * vol).cumsum().iloc[-1] / cum_v.iloc[-1]) if len(cum_v) else px
                direction = np.sign(df["close"].astype(float).diff().fillna(0.0))
                obv_s = (direction * vol).cumsum()
                obv = float(obv_s.iloc[-1])
                obv_ma = float(obv_s.rolling(20, min_periods=5).mean().iloc[-1])
                hl = (df["high"] - df["low"]).replace(0, np.nan)
                loc = ((df["close"] - df["low"]) / hl * 2.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
                cvd = float((vol * loc).cumsum().iloc[-1])
                efi = float((df["close"].astype(float).diff() * vol).ewm(span=13, adjust=False).mean().iloc[-1])
                try:
                    vp = compute_session_volume_profile(df, bin_step=BIN_STEP.get(data.get("currency", "BTC"), 250))
                    lvn = list(vp.get("lvn") or [])
                except Exception:
                    lvn = []
            else:
                vwap = px

    loc_long = loc_short = False
    loc_note = "GEX walls unavailable"
    if spot > 0 and gex_sup and gex_res:
        near_lvn = any(abs(spot - lv) <= spot * 0.002 for lv in lvn)
        near_call = abs(spot - gex_res) <= max(50.0, 0.0015 * spot)
        if spot > gex_res:
            loc_long, loc_short = True, False
            loc_note = f"spot {spot:.0f} above call-wall {gex_res:.0f}"
        elif spot < gex_sup:
            loc_long, loc_short = False, True
            loc_note = f"spot {spot:.0f} below put-wall {gex_sup:.0f}"
        elif near_call:
            loc_long, loc_short = False, True
            loc_note = f"spot {spot:.0f} rejected at call-wall {gex_res:.0f}"
        else:
            loc_long, loc_short = True, False
            loc_note = f"spot {spot:.0f} between walls {gex_sup:.0f}/{gex_res:.0f}"
        if near_lvn:
            loc_long = True
            loc_note += " · through LVN"
    checks.append({"name": "Location (GEX)", "long": loc_long, "short": loc_short, "note": loc_note})

    vwap_long = vwap_short = False
    vwap_note = "VWAP unavailable"
    if px and vwap:
        vwap_long, vwap_short = px > vwap, px < vwap
        vwap_note = f"px {px:.1f} vs VWAP {vwap:.1f}"
    checks.append({"name": "Trend (VWAP)", "long": vwap_long, "short": vwap_short, "note": vwap_note})

    obv_long = obv_short = False
    obv_note = "OBV unavailable"
    if obv is not None and obv_ma is not None and not np.isnan(obv_ma):
        obv_long, obv_short = obv > obv_ma, obv < obv_ma
        obv_note = f"OBV {obv:.0f} vs MA20 {obv_ma:.0f}"
    checks.append({"name": "Macro Flow (OBV)", "long": obv_long, "short": obv_short, "note": obv_note})

    cvd_long = cvd_short = False
    cvd_note = "CVD unavailable"
    if cvd is not None and not df.empty and "volume" in df.columns:
        cvd_long, cvd_short = cvd > 0, cvd < 0
        cvd_note = f"CVD {cvd:.0f}"
    checks.append({"name": "Order Delta (CVD)", "long": cvd_long, "short": cvd_short, "note": cvd_note})

    efi_long = efi_short = False
    efi_note = "EFI unavailable"
    if efi is not None and not np.isnan(efi):
        efi_long, efi_short = efi > 0, efi < 0
        efi_note = f"EFI13 {efi:.1f}"
    checks.append({"name": "Execution (EFI 13)", "long": efi_long, "short": efi_short, "note": efi_note})

    setup_l = sum(1 for ch in checks[:4] if ch["long"])
    setup_s = sum(1 for ch in checks[:4] if ch["short"])
    if setup_l >= 3 and efi_long:
        trigger, colour, summary = "LONG TRIGGER", "#00E676", f"Setup {setup_l}/4 long + EFI>0."
    elif setup_s >= 3 and efi_short:
        trigger, colour, summary = "SHORT TRIGGER", "#FF5252", f"Setup {setup_s}/4 short + EFI<0."
    elif setup_l >= 3:
        trigger, colour, summary = "LONG BIAS (no fire)", "#80CBC4", f"Setup {setup_l}/4 long but EFI not >0."
    elif setup_s >= 3:
        trigger, colour, summary = "SHORT BIAS (no fire)", "#EF9A9A", f"Setup {setup_s}/4 short but EFI not <0."
    elif setup_l > setup_s:
        trigger, colour, summary = "LONG LEAN (no fire)", "#80CBC4", f"Only {setup_l}/4 setup long."
    elif setup_s > setup_l:
        trigger, colour, summary = "SHORT LEAN (no fire)", "#EF9A9A", f"Only {setup_s}/4 setup short."
    else:
        trigger, colour, summary = "NO DIRECTIONAL TRIGGER", "#FF9800", f"Setup split {setup_l}L/{setup_s}S."
    return {
        "trigger": trigger, "colour": colour, "summary": summary,
        "long_hits": sum(1 for c in checks if c["long"]),
        "short_hits": sum(1 for c in checks if c["short"]),
        "checks": checks,
    }


def compute_superhuman_scores(data, df_candles):
    levels = data.get("levels", {}) or {}
    chain = data.get("chain_results", []) or []
    spot = float(data.get("spot_price", 0) or 0)
    if spot <= 0 or not chain:
        return {"error": "Insufficient data for scoring"}

    now = datetime.datetime.now(datetime.timezone.utc)
    T = float(data.get("T", 0) or 0)
    dte = max(min(T * 365.0 if T > 0 else 3.0, 30.0), 0.05)
    tod_factor = 1.0  # 24/7 market — no cash-session decay

    total_delta_gex = float(data.get("total_net_gex_oi", 0) or 0)
    # scale: BTC GEX is in $ terms via S^2
    gamma_raw = total_delta_gex / max(abs(total_delta_gex), 1e7) * 70
    gamma_regime_score = float(np.clip(gamma_raw, -100, 100))
    if dte <= 1.0:
        gamma_regime_score = float(np.clip(gamma_regime_score * 1.25, -100, 100))
    elif dte >= 7.0:
        gamma_regime_score = float(np.clip(gamma_regime_score * 0.85, -100, 100))

    straddle = float(levels.get("Straddle_Cost", 0) or 0)
    expected_move_pct = (straddle / spot * 100.0) if spot > 0 else 0.0
    realised_range_pct = 0.0
    day_high = day_low = day_open = None
    df_session = pd.DataFrame()
    if df_candles is not None and not getattr(df_candles, "empty", True):
        df_tmp = df_candles.copy()
        df_tmp["time"] = pd.to_datetime(df_tmp["time"], utc=True)
        latest_date = df_tmp["time"].dt.date.max()
        df_session = df_tmp[df_tmp["time"].dt.date == latest_date].sort_values("time")
        if not df_session.empty:
            day_high = float(df_session["high"].max())
            day_low = float(df_session["low"].min())
            day_open = float(df_session["open"].iloc[0])
            if day_open > 0:
                realised_range_pct = (day_high - day_low) / day_open * 100.0
    if expected_move_pct > 0.12:
        move_score = float(np.clip((expected_move_pct - realised_range_pct * 1.15) / expected_move_pct * 80, -100, 100))
    else:
        move_score = 0.0

    total_vex = sum(float(r.get("VEX", 0) or 0) for r in chain)
    total_cex = sum(float(r.get("CEX", 0) or 0) for r in chain)
    flow_score = float(np.clip(np.tanh((total_vex / 50.0) + (total_cex / 50.0)) * 85, -100, 100))

    or_score = 0.0
    or_high = or_low = None
    gex_sup = float(levels.get("GEX_Support") or spot)
    gex_res = float(levels.get("GEX_Resistance") or spot)
    if not df_session.empty and len(df_session) >= 5:
        n_or = min(12, max(5, len(df_session) // 6))
        or_bars = df_session.head(n_or)
        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        if spot > or_high and spot > gex_res:
            or_score = 70.0
        elif spot < or_low and spot < gex_sup:
            or_score = -70.0
        elif or_high < gex_res and or_low > gex_sup:
            or_score = 45.0
        elif spot > gex_res or spot < gex_sup:
            or_score = -25.0
        else:
            or_score = 8.0

    flip = float(levels.get("Zero_Gamma_Flip") or spot)
    dist_pct = abs(spot - flip) / spot * 100.0 if spot else 0.0
    if gamma_regime_score > 20 and dist_pct < 0.35:
        flip_score = 40.0
    elif gamma_regime_score < -20 and dist_pct > 0.6:
        flip_score = -40.0
    else:
        flip_score = float(np.clip((0.4 - dist_pct) * 50, -30, 30))

    composite = float(np.clip(
        0.38 * gamma_regime_score + 0.22 * move_score + 0.15 * flow_score + 0.15 * or_score + 0.10 * flip_score,
        -100, 100,
    ))
    big_range = realised_range_pct > expected_move_pct * 1.5 if expected_move_pct > 0.1 else realised_range_pct > 3.0
    quiet_range = realised_range_pct < expected_move_pct * 0.85 if expected_move_pct > 0.1 else realised_range_pct < 1.5
    strong_long_g = gamma_regime_score >= 55
    strong_short_g = gamma_regime_score <= -55

    if composite >= 42 and strong_long_g and quiet_range:
        bias, action, colour, clarity = "QUIET PIN", "Non-directional → short straddle / iron fly", "#00E676", "Strong long-gamma + quiet range."
    elif composite >= 35 and strong_long_g and big_range:
        bias, action, colour, clarity = "GAMMA REVERSION", "Fade extended moves into close of UTC day", "#26A69A", "Long-gamma but already travelled far."
    elif composite <= -42 or (strong_short_g and big_range):
        bias, action, colour, clarity = "TREND / BREAKOUT", "Directional → debit spreads / perps", "#FF5252", "Short-gamma or wall break."
    elif abs(composite) <= 15:
        bias, action, colour, clarity = "NO EDGE", "Stay out or tight range only", "#FF9800", "No clear dealer positioning."
    else:
        bias, action, colour, clarity = "MILD DIRECTIONAL", "Defined-risk spreads / calendars", "#2196F3", "Moderate bias."

    return {
        "gamma_regime_score": round(gamma_regime_score, 1),
        "move_score": round(move_score, 1),
        "flow_score": round(flow_score, 1),
        "or_score": round(or_score, 1),
        "flip_score": round(flip_score, 1),
        "composite": round(composite, 1),
        "bias": bias, "action": action, "colour": colour, "clarity": clarity,
        "expected_move_pct": round(expected_move_pct, 2),
        "realised_range_pct": round(realised_range_pct, 2),
        "straddle": round(straddle, 2),
        "total_delta_gex_cr": round(total_delta_gex / 1e6, 2),
        "or_high": or_high, "or_low": or_low,
        "gex_support": gex_sup, "gex_resistance": gex_res,
        "dte": round(dte, 2), "dist_to_flip_pct": round(dist_pct, 3),
        "tod_factor": tod_factor,
        "day_open": day_open, "day_high": day_high, "day_low": day_low,
        "big_range": big_range, "quiet_range": quiet_range,
        "timestamp_ist": now.strftime("%d-%b-%Y %H:%M:%S UTC"),
        "dir_trigger": evaluate_directional_trigger(data, df_candles),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Parameters & Strategy Builder")
st.sidebar.caption("Public Deribit REST · no API key")

currency = st.sidebar.selectbox("Underlying", ["BTC", "ETH"])
r_rate = st.sidebar.number_input("Risk-free rate (r)", 0.0, 0.15, 0.04, 0.01)

if st.sidebar.button("↻ Refresh instruments", use_container_width=True):
    try:
        _fetch_instruments_ok.clear()
        fetch_index_price.clear()
    except Exception:
        pass
    st.rerun()

inst_df = fetch_instruments(currency)
spot_live = fetch_index_price(currency)
if inst_df.empty:
    st.sidebar.warning(LAST_API_ERROR or "Deribit instruments unavailable")

expiries = []
if not inst_df.empty:
    exp_uniq = sorted(inst_df["expiration_timestamp"].unique())
    now_ms = int(time.time() * 1000)
    exp_uniq = [e for e in exp_uniq if e >= now_ms]
    expiries = exp_uniq

def _exp_label(ts):
    dt = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%d%b%Y %H:%M UTC")

exp_choice = st.sidebar.selectbox(
    "Expiry",
    expiries if expiries else [0],
    format_func=lambda x: _exp_label(x) if x else "N/A",
)

c3, c4 = st.sidebar.columns(2)
with c3:
    strikes_below = st.number_input("Below ATM", 1, 40, 12, 1)
with c4:
    strikes_above = st.number_input("Above ATM", 1, 40, 12, 1)
hv_days = st.sidebar.slider("HV lookback (days)", 10, 90, 30, 5)

exp_df = inst_df[inst_df["expiration_timestamp"] == exp_choice] if not inst_df.empty else pd.DataFrame()
all_strikes = sorted(exp_df["strike"].dropna().unique().tolist()) if not exp_df.empty else []

with st.sidebar.expander("2. Build Strategy Basket", expanded=True):
    sel_k = st.selectbox("Strike", all_strikes if all_strikes else [spot_live or 100000], format_func=lambda x: f"{int(x)}")
    b1, b2 = st.columns(2)
    with b1:
        opt_type = st.selectbox("Type", ["C", "P"])
    with b2:
        trade_action = st.selectbox("Action", ["BUY", "SELL"])
    entry_px = st.number_input("Entry USD (0 = mark)", 0.0, value=0.0, step=10.0)
    qty = st.number_input("Contracts", 0.1, value=1.0, step=0.1)
    a1, a2 = st.columns(2)
    with a1:
        if st.button("+ Add Leg", use_container_width=True):
            st.session_state["basket_legs"].append({
                "strike": float(sel_k), "type": opt_type, "action": trade_action,
                "entry_price": entry_px, "qty": qty,
            })
            st.rerun()
    with a2:
        if st.button("Clear Basket", use_container_width=True):
            st.session_state["basket_legs"] = []
            st.rerun()

run_btn = st.sidebar.button("🚀 Fetch Chain & Greeks", use_container_width=True)
tf_label = st.session_state["selected_timeframe"]


# ---------------------------------------------------------------------------
# Fetch live bundle
# ---------------------------------------------------------------------------
def fetch_live_bundle(tf_label: str):
    spot = fetch_index_price(currency)
    if spot <= 0:
        return None
    res_ms = INTERVAL_MS.get(tf_label, 300_000)
    df_perp = fetch_candles(PERP[currency], res_ms, lookback_hours=96)
    df_spot = df_perp.copy()  # index ≈ perp for crypto; both shown
    hv = hv_from_candles(df_perp, hv_days)

    built = build_chain(currency, exp_choice, spot, r_rate, strikes_below, strikes_above)
    if not built or not built[0]:
        return None
    chain, T, med_iv, tcoi, tpoi, tgex_oi, tgex_vol = built
    max_pain = calculate_max_pain(chain)
    levels = calculate_support_resistance_targets(chain, spot, max_pain)
    pcr = (tpoi / tcoi) if tcoi else 0.0

    # ATM IV percentile vs chain
    ivs = [r["C_IV"] for r in chain if r["C_IV"] > 1] + [r["P_IV"] for r in chain if r["P_IV"] > 1]
    atm = min(chain, key=lambda r: abs(r["Strike"] - spot))
    atm_iv = (atm["C_IV"] + atm["P_IV"]) / 2.0
    iv_pct = float(si.percentileofscore(ivs, atm_iv)) if ivs else 50.0

    perp_book = fetch_order_book(PERP[currency], 20)
    perp_ticker = deribit_get("public/ticker", {"instrument_name": PERP[currency]}) or {}
    fut_ltp = float(perp_ticker.get("last_price") or perp_ticker.get("mark_price") or spot)
    basis = fut_ltp - spot

    dvol = fetch_dvol_candles(currency)

    return {
        "currency": currency,
        "spot_price": spot,
        "F": fut_ltp,
        "T": T,
        "index_hv": hv,
        "chain_results": chain,
        "levels": levels,
        "max_pain_strike": int(max_pain) if max_pain else 0,
        "total_call_oi": tcoi,
        "total_put_oi": tpoi,
        "total_net_gex_oi": tgex_oi,
        "total_net_gex_vol": tgex_vol,
        "pcr": pcr,
        "iv_percentile": iv_pct,
        "df_candles": df_spot,
        "df_futures": df_perp,
        "dvol": dvol,
        "perp_book": perp_book,
        "basis_info": {
            "fut_token": PERP[currency],
            "fut_expiry": "PERPETUAL",
            "spot_ltp": spot,
            "fut_ltp": fut_ltp,
            "basis": round(basis, 2),
        },
        "timestamp": datetime.datetime.now(IST).strftime("%d-%b %H:%M:%S IST"),
        "selected_expiry": _exp_label(exp_choice) if exp_choice else "",
        "fut_fallback_msg": "24/7 Deribit perpetual",
        "fut_is_fallback": False,
        "market_data": {},
    }


if run_btn or "data_store" not in st.session_state:
    with st.spinner("Fetching Deribit chain + perp candles…"):
        bundle = fetch_live_bundle(st.session_state["selected_timeframe"])
        if bundle:
            st.session_state["data_store"] = bundle
        elif "data_store" not in st.session_state:
            why = LAST_API_ERROR or "no chain for this expiry"
            st.error(
                f"Deribit public API failed: **{why}**.\n\n"
                "If the message is `system_maintenance`, Deribit is down globally "
                "(HTTP 503) — not a Render env-var problem. Wait and click "
                "**↻ Refresh instruments** in the sidebar."
            )
            st.stop()


# ---------------------------------------------------------------------------
# Liquidity Δ from perp book
# ---------------------------------------------------------------------------
def update_liq_from_book(book: dict, ccy: str):
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid_qty = sum(float(x[1]) for x in bids if isinstance(x, (list, tuple)) and len(x) >= 2)
    ask_qty = sum(float(x[1]) for x in asks if isinstance(x, (list, tuple)) and len(x) >= 2)
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    ltp = float(book.get("last_price") or book.get("mark_price") or 0)
    hist = list(st.session_state.get("liq_delta_history") or [])
    prev = hist[-1] if hist else None
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc),
        "index": ccy,
        "bid_qty_lots": bid_qty,
        "ask_qty_lots": ask_qty,
        "bid_change": bid_qty - float(prev["bid_qty_lots"]) if prev else 0.0,
        "ask_change": ask_qty - float(prev["ask_qty_lots"]) if prev else 0.0,
        "net_liq": 0.0,
        "best_bid": best_bid, "best_ask": best_ask, "ltp": ltp,
    }
    rec["net_liq"] = rec["bid_change"] - rec["ask_change"]
    hist.append(rec)
    st.session_state["liq_delta_history"] = hist[-180:]
    return st.session_state["liq_delta_history"]


def classify_liq_alert(bid_change, ask_change, bid_s, ask_s, k=1.5):
    if bid_s is None or ask_s is None:
        return {"label": "NO SIGNAL", "color": "#888888", "hint": "Need more snapshots", "bid_thr": None}
    bid_thr, ask_thr = k * bid_s, k * ask_s
    if bid_change <= -bid_thr:
        return {"label": "BIDS PULLED", "color": "#FF5252", "hint": "Liquidity vacuum", "bid_thr": bid_thr}
    if ask_change <= -ask_thr:
        return {"label": "ASKS PULLED", "color": "#00E676", "hint": "Offers lifted", "bid_thr": bid_thr}
    if bid_change >= bid_thr:
        return {"label": "BIDS STACKED", "color": "#2196F3", "hint": "Passive absorption", "bid_thr": bid_thr}
    return {"label": "NO SIGNAL", "color": "#888888", "hint": f"Inside ±{k}σ", "bid_thr": bid_thr}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@st.fragment(run_every=8 if st.session_state.get("enable_main_refresh") else None)
def live_dashboard():
    if st.session_state.get("enable_main_refresh"):
        refreshed = fetch_live_bundle(st.session_state["selected_timeframe"])
        if refreshed:
            st.session_state["data_store"] = refreshed
    data = st.session_state["data_store"]
    lvls = data.get("levels", {}) or {}
    chain = data.get("chain_results") or []
    df_chain = pd.DataFrame(chain)
    df_fut = data.get("df_futures", pd.DataFrame())
    df_full = data.get("df_candles", pd.DataFrame())

    st.markdown("<div class='sticky-summary'>", unsafe_allow_html=True)
    h_l, h_r = st.columns([0.72, 0.28])
    with h_l:
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;'>"
            f"<h1 class='custom-heading' style='margin:0;font-size:17px;'>₿ {data.get('currency')} Market Summary</h1>"
            f"<span style='color:#7CB342;font-size:11px;'>Updated {data.get('timestamp','')} · Deribit</span></div>",
            unsafe_allow_html=True,
        )
    with h_r:
        cb = st.checkbox("Auto-Refresh 8s", value=st.session_state["enable_main_refresh"], key="cb_main_refresh")
        if cb != st.session_state["enable_main_refresh"]:
            st.session_state["enable_main_refresh"] = cb
            st.rerun()

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Index (Perp)", f"{data['spot_price']:.0f} ({data['F']:.0f})")
    c2.metric("Max Pain", f"{data['max_pain_strike']}")
    c3.metric("Net GEX (OI)", f"${data['total_net_gex_oi']/1e6:.1f}M")
    c4.metric("ATM IV Rank", f"{data['iv_percentile']:.0f}%")
    c5.metric("PCR", f"{data['pcr']:.2f}")
    c6.metric("C/P OI", f"{data['total_call_oi']:.0f}/{data['total_put_oi']:.0f}")
    c7.metric("Flip", f"{lvls.get('Zero_Gamma_Flip', '–')}")
    c8.metric("Straddle $", f"{lvls.get('Straddle_Cost', 0):.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

    scores = compute_superhuman_scores(data, df_fut if df_fut is not None and not df_fut.empty else df_full)
    if "error" not in scores:
        trig = scores.get("dir_trigger") or {}
        st.markdown(
            f"<div style='background:#1A1F2B;border:1px solid #2A2F3A;border-radius:8px;"
            f"padding:6px 12px;margin:6px 0 8px 0;display:flex;flex-wrap:wrap;align-items:center;gap:14px;'>"
            f"<span style='font-weight:700;color:#00E676;font-size:13px;'>🧠 Superhuman</span>"
            f"<span style='font-weight:800;color:{scores['colour']};font-size:14px;'>{scores['bias']} ({scores['composite']:+.0f})</span>"
            f"<span style='color:#AAA;font-size:12px;'>{scores.get('clarity','')}</span>"
            f"<span style='font-weight:800;color:{trig.get('colour','#FF9800')};font-size:13px;'>⚡ {trig.get('trigger','NO TRIGGER')}</span>"
            f"<span style='color:#CCC;font-size:12px;'>{trig.get('summary','')}</span></div>",
            unsafe_allow_html=True,
        )
        with st.expander("▼ Trigger · Decision tree", expanded=False):
            st.caption("LONG TRIGGER = ≥3 of GEX/VWAP/OBV/CVD bullish AND EFI>0. Same construct as the index desk.")
            lines = ["| Metric | Now |", "|---|---|"]
            for ch in trig.get("checks") or []:
                side = "L" if ch.get("long") and not ch.get("short") else ("S" if ch.get("short") and not ch.get("long") else "—")
                lines.append(f"| {ch['name']} | **{side}** {ch.get('note','')} |")
            st.markdown("\n".join(lines))
            st.markdown(
                f"**{scores['bias']} ({scores['composite']:+.0f})** — {scores['action']}"
            )

    # Technicals header
    badge_html = ""
    if not df_full.empty and len(df_full) >= 20:
        latest_row = df_full.iloc[-1]
        rsi_val = float(latest_row.get("rsi") or 50)
        rsi_status = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
        rsi_cls = "badge-bearish" if rsi_val > 70 else ("badge-bullish" if rsi_val < 30 else "badge-neutral")
        macd_val = float(latest_row.get("macd") or 0)
        macd_sig = float(latest_row.get("macd_signal") or 0)
        macd_status = "Bullish XO" if macd_val > macd_sig else "Bearish XO"
        macd_cls = "badge-bullish" if macd_val > macd_sig else "badge-bearish"
        recent_bw = df_full["bb_bandwidth"].tail(20)
        is_sqz = float(latest_row.get("bb_bandwidth") or 0) <= float(recent_bw.quantile(0.20) or 0)
        sqz_cls = "badge-neutral" if is_sqz else "badge-bullish"
        badge_html = (
            f"<span class='status-badge {sqz_cls}'>BB: {'Squeeze' if is_sqz else 'Expand'}</span>"
            f"<span class='status-badge {macd_cls}'>MACD: {macd_status}</span>"
            f"<span class='status-badge {rsi_cls}'>RSI: {rsi_val:.0f} ({rsi_status})</span>"
        )

    tech_h, tech_b, tf_col = st.columns([0.28, 0.52, 0.20])
    with tech_h:
        heading_ribbon("📈 Underlying Technicals", "Perp + Bollinger(20,2). MACD 12/26/9. RSI 14.")
    with tech_b:
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)
    with tf_col:
        selected_tf = st.selectbox(
            "TF", ["1 min", "3 min", "5 min", "15 min"],
            index=["1 min", "3 min", "5 min", "15 min"].index(st.session_state["selected_timeframe"])
            if st.session_state["selected_timeframe"] in ["1 min", "3 min", "5 min", "15 min"] else 2,
            key="tf_select_frag", label_visibility="collapsed",
        )
        if selected_tf != st.session_state["selected_timeframe"]:
            st.session_state["selected_timeframe"] = selected_tf
            upd = fetch_live_bundle(selected_tf)
            if upd:
                st.session_state["data_store"] = upd
                st.rerun()

    if not df_full.empty and len(df_full) >= 10:
        df_chart = df_full.tail(400).copy()
        df_chart["time_str"] = ist_label(df_chart["time"], with_date=True)
        min_p = min(df_chart["close"].min(), df_chart["bb_lower"].min())
        max_p = max(df_chart["close"].max(), df_chart["bb_upper"].max())
        pad = (max_p - min_p) * 0.05
        tech_left, tech_right = st.columns([0.58, 0.42])
        with tech_left:
            fig_px = plt_go.Figure()
            fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["close"], mode="lines", name="Index/Perp", line=dict(color="#00E676", width=2)))
            fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_upper"], mode="lines", name="BB Upper", line=dict(color="rgba(33,150,243,0.5)", width=1)))
            fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_lower"], mode="lines", name="BB Lower", line=dict(color="rgba(33,150,243,0.5)", width=1), fill="tonexty", fillcolor="rgba(33,150,243,0.05)"))
            fig_px.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                 height=220, margin=dict(l=8, r=8, t=12, b=8), hovermode="x unified",
                                 legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                                 yaxis=dict(range=[min_p - pad, max_p + pad]), xaxis=dict(type="category", nticks=8))
            st.plotly_chart(fig_px, use_container_width=True, key="chart_spot_bb")
        with tech_right:
            fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45])
            colors_macd = np.where(df_chart["macd_hist"] >= 0, "#00E676", "#FF5252")
            fig_ind.add_trace(plt_go.Bar(x=df_chart["time_str"], y=df_chart["macd_hist"], marker_color=colors_macd, showlegend=False))
            fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd"], mode="lines", name="MACD", line=dict(color="#2196F3", width=1.3)))
            fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd_signal"], mode="lines", name="Sig", line=dict(color="#FF9800", width=1.3)))
            fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["rsi"], mode="lines", name="RSI", line=dict(color="#E040FB", width=1.3)), row=2, col=1)
            fig_ind.add_hline(y=70, line_dash="dash", line_color="#FF5252", line_width=1, row=2, col=1)
            fig_ind.add_hline(y=30, line_dash="dash", line_color="#00E676", line_width=1, row=2, col=1)
            fig_ind.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                  height=220, margin=dict(l=8, r=8, t=12, b=8), hovermode="x unified",
                                  legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)))
            fig_ind.update_xaxes(type="category", nticks=6)
            st.plotly_chart(fig_ind, use_container_width=True, key="chart_macd_rsi")

    left_col, right_col = st.columns([0.70, 0.30])

    with left_col:
        hdr1, hdr2, hdr3 = st.columns([0.52, 0.20, 0.28])
        with hdr1:
            heading_ribbon(
                f"PERP · VWAP · VP  ({currency})",
                "<b>PERP</b> — Deribit perpetual last.<br>"
                "<b>VWAP</b> — session Σ(TP×Vol)/ΣVol, TP=(H+L+C)/3. Orange line + ±σ bands.<br>"
                "<b>VP</b> — volume-at-price histogram (right). Yellow bar = POC.<br>"
                "<b>Basis</b> — Perp − Index. Contango +, backwardation −.",
            )
        with hdr2:
            basis = data.get("basis_info") or {}
            if basis.get("basis") is not None:
                bc = "#00E676" if basis["basis"] >= 0 else "#FF5252"
                st.markdown(
                    f"<div style='font-size:12px;padding-top:2px;'>Basis "
                    f"<span style='color:{bc};font-weight:800;'>{basis['basis']:+.1f}</span></div>",
                    unsafe_allow_html=True,
                )
        with hdr3:
            w1, w2 = st.columns(2)
            with w1:
                sigma_mult = st.selectbox(
                    "VWAP bands", [1.0, 1.5, 2.0], index=1,
                    format_func=lambda x: f"±{x}σ", key="vwap_sigma",
                    label_visibility="collapsed",
                )
            with w2:
                win_label = st.selectbox(
                    "Window", ["6 hrs", "1 day", "2 days"],
                    index=["6 hrs", "1 day", "2 days"].index(st.session_state.get("perp_window", "1 day"))
                    if st.session_state.get("perp_window") in ["6 hrs", "1 day", "2 days"] else 1,
                    key="perp_window_sel",
                    label_visibility="collapsed",
                )
                st.session_state["perp_window"] = win_label

        al1, al2, al3, al4 = st.columns([0.28, 0.18, 0.22, 0.32])
        with al1:
            alert_px = st.number_input(
                "Alert", min_value=0.0, value=0.0, step=50.0,
                key="tg_alert_px", label_visibility="visible",
            )
        with al2:
            arm = st.button("Send alert", key="tg_arm_btn", use_container_width=True)
        with al3:
            peco_tg = st.checkbox("PECO → TG", value=True, key="tg_peco_on")
        with al4:
            token_ok, chat_ok = telegram_creds()
            if token_ok and chat_ok:
                st.caption("TG live")
            else:
                st.caption("Set TELE_BOTTOKEN + TELE_CHATID")
        if arm:
            if alert_px <= 0:
                st.warning("Enter a futures price > 0")
            else:
                last_px = 0.0
                if df_fut is not None and not df_fut.empty:
                    last_px = float(df_fut["close"].iloc[-1])
                side = "above" if alert_px >= last_px else "below"
                alerts = list(st.session_state.get("price_alerts") or [])
                alerts.append({
                    "price": float(alert_px),
                    "side": side,
                    "armed_at": last_px,
                    "ccy": currency,
                    "done": False,
                })
                st.session_state["price_alerts"] = alerts
                ok, msg = send_telegram(
                    f"🔔 <b>{currency} alert armed</b>\n"
                    f"Trigger {side} <b>{alert_px:,.1f}</b>\n"
                    f"Spot now {last_px:,.1f}"
                )
                if ok:
                    st.success(f"Armed {side} {alert_px:,.1f}")
                else:
                    st.error(f"Armed locally, Telegram failed: {msg}")

        live_alerts = [a for a in (st.session_state.get("price_alerts") or []) if not a.get("done") and a.get("ccy") == currency]
        if live_alerts:
            bits = " · ".join(f"{a['side']} {a['price']:,.0f}" for a in live_alerts)
            st.caption(f"Armed: {bits}")

        if df_fut is not None and not df_fut.empty:
            dfi = df_fut.copy().reset_index(drop=True)
            dfi["time"] = to_ist(dfi["time"])
            dfi["ist_date"] = dfi["time"].dt.date
            now_ist = datetime.datetime.now(IST)
            hours = {"6 hrs": 6, "1 day": 24, "2 days": 48}.get(st.session_state.get("perp_window", "1 day"), 24)
            cut = now_ist - datetime.timedelta(hours=hours)
            dfi = dfi[dfi["time"] >= cut].reset_index(drop=True)
            if dfi.empty:
                dfi = df_fut.copy().reset_index(drop=True)
                dfi["time"] = to_ist(dfi["time"])
                dfi["ist_date"] = dfi["time"].dt.date
                dfi = dfi.tail(max(12, hours * 2)).reset_index(drop=True)
            dfi["tp"] = (dfi["high"] + dfi["low"] + dfi["close"]) / 3.0

            vwaps, stds = [], []
            prev_day = None
            cum_vol = cum_tp = cum_sq = 0.0
            for i in range(len(dfi)):
                day = dfi.loc[i, "ist_date"]
                if day != prev_day:
                    cum_vol = cum_tp = cum_sq = 0.0
                    prev_day = day
                vol = float(dfi.loc[i, "volume"] or 0)
                tp = float(dfi.loc[i, "tp"])
                cum_vol += vol
                cum_tp += tp * vol
                vwap = cum_tp / cum_vol if cum_vol > 0 else tp
                cum_sq += vol * (tp - vwap) ** 2
                vwaps.append(vwap)
                stds.append(math.sqrt(max(cum_sq / cum_vol if cum_vol else 0.0, 0.0)))
            dfi["vwap"] = vwaps
            dfi["vwap_std"] = stds
            dfi["vwap_upper"] = dfi["vwap"] + sigma_mult * dfi["vwap_std"]
            dfi["vwap_lower"] = dfi["vwap"] - sigma_mult * dfi["vwap_std"]
            close = dfi["close"].astype(float)
            vol = dfi["volume"].astype(float)
            dfi["obv"] = (np.sign(close.diff().fillna(0)) * vol).cumsum()
            dfi["obv_ma20"] = dfi["obv"].rolling(20, min_periods=1).mean()
            dfi["efi13"] = (close.diff() * vol).ewm(span=13, adjust=False).mean()
            hl = (dfi["high"] - dfi["low"]).replace(0, np.nan)
            loc = ((dfi["close"] - dfi["low"]) / hl * 2 - 1).fillna(0).clip(-1, 1)
            dfi["cvd"] = (vol * loc).cumsum()
            dfi["time_str"] = dfi["time"].dt.strftime("%d-%b %H:%M")
            vp = compute_session_volume_profile(dfi, bin_step=BIN_STEP.get(currency, 250))
            y0 = float(min(dfi["close"].min(), dfi["vwap_lower"].min())) * 0.998
            y1 = float(max(dfi["close"].max(), dfi["vwap_upper"].max())) * 1.002

            fig_stack = make_subplots(
                rows=4, cols=2, column_widths=[0.86, 0.14],
                row_heights=[0.44, 0.16, 0.20, 0.20],
                shared_xaxes=True, horizontal_spacing=0.008, vertical_spacing=0.012,
                specs=[[{}, {}], [{}, None], [{}, None], [{}, None]],
            )
            fig_stack.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["vwap_upper"], mode="lines", showlegend=False, hoverinfo="skip",
                                               line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot")), row=1, col=1)
            fig_stack.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["vwap_lower"], mode="lines", showlegend=False, hoverinfo="skip",
                                               line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
                                               fill="tonexty", fillcolor="rgba(255,152,0,0.08)"), row=1, col=1)
            fig_stack.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["vwap"], mode="lines", name="VWAP", line=dict(color="#FF9800", width=2)), row=1, col=1)
            fig_stack.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["close"], mode="lines", name="Perp", line=dict(color="#2196F3", width=2)), row=1, col=1)
            if vp.get("ok"):
                mids, vols, colors = [], [], []
                for m, v in zip(vp["mids"], vp["vol"]):
                    if y0 <= float(m) <= y1:
                        mids.append(float(m)); vols.append(float(v))
                        colors.append("#FFD54F" if abs(m - vp["poc"]) < 1e-6 else "rgba(100,181,246,0.7)")
                fig_stack.add_trace(plt_go.Bar(x=vols, y=mids, orientation="h", showlegend=False, marker=dict(color=colors)), row=1, col=2)
            fig_stack.add_trace(plt_go.Scatter(
                x=dfi["time_str"], y=dfi["obv"].where(dfi["obv"] >= 0),
                mode="lines", showlegend=False, name="OBV+",
                line=dict(color="#00E676", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.22)",
            ), row=2, col=1)
            fig_stack.add_trace(plt_go.Scatter(
                x=dfi["time_str"], y=dfi["obv"].where(dfi["obv"] < 0),
                mode="lines", showlegend=False, name="OBV-",
                line=dict(color="#FF5252", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,82,82,0.22)",
            ), row=2, col=1)
            fig_stack.add_trace(plt_go.Scatter(
                x=dfi["time_str"], y=dfi["obv_ma20"], mode="lines", showlegend=False,
                line=dict(color="#FFF176", width=1.4),
            ), row=2, col=1)
            efi_col = np.where(dfi["efi13"] >= 0, "#00E676", "#FF5252")
            fig_stack.add_trace(plt_go.Bar(
                x=dfi["time_str"], y=dfi["efi13"],
                marker_color=efi_col, marker_line_width=0,
                opacity=1.0, showlegend=False, name="EFI13",
            ), row=3, col=1)
            cvd_last = float(dfi["cvd"].iloc[-1])
            fig_stack.add_trace(plt_go.Scatter(
                x=dfi["time_str"], y=dfi["cvd"].where(dfi["cvd"] >= 0),
                mode="lines", showlegend=False, name="CVD+",
                line=dict(color="#00E676", width=1.8),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.20)",
            ), row=4, col=1)
            fig_stack.add_trace(plt_go.Scatter(
                x=dfi["time_str"], y=dfi["cvd"].where(dfi["cvd"] < 0),
                mode="lines", showlegend=False, name="CVD-",
                line=dict(color="#FF5252", width=1.8),
                fill="tozeroy", fillcolor="rgba(255,82,82,0.22)",
            ), row=4, col=1)
            for rr in (2, 3, 4):
                fig_stack.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot", row=rr, col=1)
            fig_stack.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                height=500, margin=dict(l=36, r=4, t=4, b=12), hovermode="x unified",
                legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
                bargap=0.05,
            )
            fig_stack.update_yaxes(range=[y0, y1], title_text="PX", title_font=dict(size=9), tickfont=dict(size=8), row=1, col=1)
            fig_stack.update_yaxes(range=[y0, y1], showticklabels=False, row=1, col=2)
            fig_stack.update_yaxes(title_text="OBV", title_font=dict(size=9), tickfont=dict(size=8), row=2, col=1)
            fig_stack.update_yaxes(title_text="EFI13", title_font=dict(size=9), tickfont=dict(size=8), row=3, col=1)
            fig_stack.update_yaxes(title_text="CVD", title_font=dict(size=9), tickfont=dict(size=8), row=4, col=1)
            fig_stack.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=dfi["time_str"].tolist(),
                nticks=8, tickfont=dict(size=8), row=4, col=1,
            )
            # Snapshot DEX / premium for the time-series tab
            tot_dex = float(df_chain["DEX_OI"].sum()) if not df_chain.empty and "DEX_OI" in df_chain.columns else 0.0
            tot_pc = float(df_chain["Prem_C"].sum()) if not df_chain.empty and "Prem_C" in df_chain.columns else 0.0
            tot_pp = float(df_chain["Prem_P"].sum()) if not df_chain.empty and "Prem_P" in df_chain.columns else 0.0
            tape = list(st.session_state.get("flow_tape") or [])
            tape.append({
                "ts": now_ist.strftime("%d-%b %H:%M"),
                "dex": tot_dex,
                "prem_c": tot_pc,
                "prem_p": tot_pp,
            })
            st.session_state["flow_tape"] = tape[-180:]

            tab_flow, tab_dex = st.tabs(["Perp · OBV · EFI · CVD", "Perp · DEX · Premium"])
            with tab_flow:
                st.plotly_chart(fig_stack, use_container_width=True, key="chart_perp_stack")
            with tab_dex:
                heading_ribbon(
                    "DEX flow · net premium",
                    "<b>DEX</b> = Σ (Δ_call×OI_call + Δ_put×OI_put) × S. Signed dollar delta from OI.<br>"
                    "<b>Call / Put prem</b> = Σ LTP×OI. Needs Auto-Refresh to build a time path.",
                )
                fig_dex = make_subplots(
                    rows=3, cols=2, column_widths=[0.86, 0.14],
                    row_heights=[0.46, 0.27, 0.27],
                    shared_xaxes=False, vertical_spacing=0.06, horizontal_spacing=0.01,
                    specs=[[{}, {}], [{}, None], [{}, None]],
                )
                fig_dex.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["vwap"], mode="lines", name="VWAP",
                                                line=dict(color="#FF9800", width=2)), row=1, col=1)
                fig_dex.add_trace(plt_go.Scatter(x=dfi["time_str"], y=dfi["close"], mode="lines", name="Perp",
                                                line=dict(color="#2196F3", width=2)), row=1, col=1)
                if vp.get("ok"):
                    fig_dex.add_trace(plt_go.Bar(
                        x=vols if vp.get("ok") else [], y=mids if vp.get("ok") else [],
                        orientation="h", showlegend=False,
                        marker=dict(color=colors if vp.get("ok") else "#64B5F6"),
                    ), row=1, col=2)
                if tape:
                    txs = [t["ts"] for t in tape]
                    dex_s = pd.Series([t["dex"] for t in tape], dtype=float)
                    bar_c = np.where(dex_s >= 0, "#00E676", "#FF5252")
                    fig_dex.add_trace(plt_go.Bar(x=txs, y=dex_s, name="DEX $", marker_color=bar_c, opacity=0.85), row=2, col=1)
                    fig_dex.add_trace(plt_go.Scatter(
                        x=txs, y=dex_s.ewm(span=5, adjust=False).mean(), name="DEX EMA",
                        line=dict(color="#FFD54F", width=2),
                    ), row=2, col=1)
                    fig_dex.add_trace(plt_go.Scatter(
                        x=txs, y=[t["prem_c"] / 1e6 for t in tape], name="Call prem $M",
                        line=dict(color="#00E676", width=2),
                    ), row=3, col=1)
                    fig_dex.add_trace(plt_go.Scatter(
                        x=txs, y=[t["prem_p"] / 1e6 for t in tape], name="Put prem $M",
                        line=dict(color="#FF5252", width=2),
                    ), row=3, col=1)
                    fig_dex.add_hline(y=0, line_dash="dot", line_color="#FFF", row=2, col=1)
                fig_dex.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    height=500, margin=dict(l=36, r=4, t=8, b=12), hovermode="x unified",
                    legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9)),
                )
                fig_dex.update_yaxes(title_text="PX", tickfont=dict(size=8), row=1, col=1)
                fig_dex.update_yaxes(title_text="DEX $", tickfont=dict(size=8), row=2, col=1)
                fig_dex.update_yaxes(title_text="Prem $M", tickfont=dict(size=8), row=3, col=1)
                st.plotly_chart(fig_dex, use_container_width=True, key="chart_perp_dex")
                if len(tape) < 4:
                    st.caption("DEX / premium path needs a few Auto-Refresh snaps. Keep refresh on.")
                else:
                    st.caption(f"DEX tape · {len(tape)} snaps · last DEX ${tot_dex:,.0f}")
            cap = (
                f"IST · Perp {float(dfi['close'].iloc[-1]):,.1f} · VWAP {float(dfi['vwap'].iloc[-1]):,.1f} "
                f"· CVD {cvd_last:,.0f}"
            )
            if vp.get("ok"):
                cap += f" · POC {vp['poc']:.0f} · VA {vp['val1']:.0f}–{vp['vah1']:.0f}"
            st.caption(cap)

            # PECO trigger sits directly under VP / price stack
            micro = classify_microstructure(dfi)
            trig = scores.get("dir_trigger") if isinstance(scores, dict) else {}
            last_close = float(dfi["close"].iloc[-1])

            # Price alerts
            new_alerts = []
            for a in list(st.session_state.get("price_alerts") or []):
                if a.get("done") or a.get("ccy") != currency:
                    new_alerts.append(a)
                    continue
                hit = (a["side"] == "above" and last_close >= a["price"]) or (
                    a["side"] == "below" and last_close <= a["price"]
                )
                if hit:
                    ok, _ = send_telegram(
                        f"🎯 <b>{currency} PRICE HIT</b>\n"
                        f"Alert {a['side']} {a['price']:,.1f}\n"
                        f"Perp now <b>{last_close:,.1f}</b>\n"
                        f"VWAP {float(dfi['vwap'].iloc[-1]):,.1f}"
                    )
                    a = dict(a)
                    a["done"] = True
                    a["hit"] = last_close
                new_alerts.append(a)
            st.session_state["price_alerts"] = new_alerts

            # PECO long/short → Telegram (cooldown 8 min per state)
            act = str(micro.get("action") or "")
            peco_side = "LONG" if "LONG" in act else ("SHORT" if "SHORT" in act else "")
            if peco_tg and peco_side:
                sig = f"{currency}:{peco_side}:{act}"
                now_s = time.time()
                last_sig = st.session_state.get("last_peco_sent") or ""
                last_ts = float(st.session_state.get("last_peco_ts") or 0)
                if sig != last_sig or (now_s - last_ts) > 480:
                    ok, _ = send_telegram(
                        f"{'🟢' if peco_side == 'LONG' else '🔴'} <b>PECO {peco_side}</b> {currency}\n"
                        f"{act}\n"
                        f"P{ARROW_GLYPH[micro.get('price','f')]} "
                        f"E{ARROW_GLYPH[micro.get('efi','f')]} "
                        f"C{ARROW_GLYPH[micro.get('cvd','f')]} "
                        f"O{ARROW_GLYPH[micro.get('obv','f')]}\n"
                        f"Perp {last_close:,.1f}  VWAP {float(dfi['vwap'].iloc[-1]):,.1f}\n"
                        f"Prepare to {peco_side.lower()}"
                    )
                    if ok:
                        st.session_state["last_peco_sent"] = sig
                        st.session_state["last_peco_ts"] = now_s
            peco_arrows = (
                f"P{ARROW_GLYPH[micro.get('price','f')]} "
                f"E{ARROW_GLYPH[micro.get('efi','f')]} "
                f"C{ARROW_GLYPH[micro.get('cvd','f')]} "
                f"O{ARROW_GLYPH[micro.get('obv','f')]}"
            )
            act_col = "#00E676" if "LONG" in str(micro.get("action", "")) else (
                "#FF5252" if "SHORT" in str(micro.get("action", "")) else "#FF9800"
            )
            tcol = (trig or {}).get("colour", "#FF9800")
            st.markdown(
                f"<div class='peco-bar'>"
                f"<div class='micro-hover peco-chip'>"
                f"<div style='color:#8FA4B8;font-size:10px;font-weight:700;letter-spacing:.04em;'>PECO TRIGGER</div>"
                f"<div style='color:{act_col};font-weight:800;font-size:13px;'>{peco_arrows}</div>"
                f"<div style='color:#EEE;font-size:11px;font-weight:700;'>{micro.get('action','NO ENTRY')}</div>"
                f"<div class='micro-tip'><b>PECO = Price · EFI · CVD · OBV</b><br>"
                f"{micro.get('micro','')}<br>{micro.get('efi_note','')}<br>"
                f"↑ up · → flat · ↓ down. EFI≈0 uses the 27-state book.</div></div>"
                f"<div class='micro-hover peco-chip'>"
                f"<div style='color:#8FA4B8;font-size:10px;font-weight:700;letter-spacing:.04em;'>SUPERHUMAN</div>"
                f"<div style='color:{scores.get('colour','#FF9800') if isinstance(scores, dict) else '#FF9800'};font-weight:800;font-size:13px;'>"
                f"{scores.get('bias','—') if isinstance(scores, dict) else '—'} "
                f"({scores.get('composite',0):+.0f})</div>"
                f"<div style='color:#AAA;font-size:11px;'>{scores.get('clarity','') if isinstance(scores, dict) else ''}</div>"
                f"<div class='micro-tip'>Quiet+long γ → PIN · big range+long γ → REVERSION · "
                f"short γ / wall break → TREND · |C|≤15 → NO EDGE.</div></div>"
                f"<div class='micro-hover peco-chip'>"
                f"<div style='color:#8FA4B8;font-size:10px;font-weight:700;letter-spacing:.04em;'>DIR TRIGGER</div>"
                f"<div style='color:{tcol};font-weight:800;font-size:13px;'>{(trig or {}).get('trigger','NO TRIGGER')}</div>"
                f"<div style='color:#AAA;font-size:11px;'>{(trig or {}).get('summary','')}</div>"
                f"<div class='micro-tip'>LONG = ≥3 of GEX/VWAP/OBV/CVD bullish AND EFI&gt;0.<br>"
                f"SHORT = ≥3 bearish AND EFI&lt;0.</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Perp candles unavailable.")

    with right_col:
        heading_ribbon(
            "GEX / OI  ·  GEX / Volume",
            "<b>GEX</b> = (Call γ × Call OI − Put γ × Put OI) × 1 coin × S² × 0.01.<br>"
            "Green net bar = long-gamma pin. Red = short-gamma accelerator.<br>"
            "Left axis: call OI up / put OI down. Right axis: net GEX $.<br>"
            "<b>Volume tab</b> swaps session volume for OI — today's prints, not inventory.",
        )
        if not df_chain.empty:
            min_s = float(df_chain["Strike"].min()) - 200
            max_s = float(df_chain["Strike"].max()) + 200

            def _one_gex(c_col, p_col, net_col, h):
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                cols = np.where(df_chain[net_col] >= 0, "#006400", "#8B0000")
                fig.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain[c_col], marker_color="#2E7D32", opacity=0.55, showlegend=False), secondary_y=False)
                fig.add_trace(plt_go.Bar(x=df_chain["Strike"], y=-df_chain[p_col], marker_color="#C62828", opacity=0.55, showlegend=False), secondary_y=False)
                fig.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain[net_col], marker_color=cols, opacity=0.85, width=25, showlegend=False), secondary_y=True)
                fig.add_hline(y=0, line_width=1, line_color="#FFFFFF")
                fig.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA")
                fig.update_layout(template="plotly_dark", paper_bgcolor="#11151C", plot_bgcolor="#0E1117",
                                  height=h, barmode="overlay", margin=dict(l=8, r=8, t=8, b=18), hovermode="x unified")
                fig.update_xaxes(type="linear", tickformat="d", range=[min_s, max_s])
                return fig

            g_oi, g_vol = st.tabs(["GEX / OI", "GEX / Volume"])
            with g_oi:
                st.plotly_chart(_one_gex("C_OI", "P_OI", "Net_GEX_OI", 250), use_container_width=True, key="chart_gex_oi")
            with g_vol:
                st.plotly_chart(_one_gex("C_Vol", "P_Vol", "Net_GEX_Vol", 250), use_container_width=True, key="chart_gex_vol")

            heading_ribbon(
                "Δ-GEX (OI)  ·  DEX / OI",
                "<b>Δ-GEX</b> = Net GEX × (|Δc|+|Δp|). ATM bars dominate.<br>"
                "<b>DEX / OI</b> = (Δc×OIc + Δp×OIp) × S. Signed dollar delta inventory.",
            )
            diffs = df_chain["Strike"].sort_values().diff().dropna()
            bar_w = float(diffs.median()) * 0.72 if len(diffs) else 400.0

            def _strike_bars(ycol, key_name, h=240):
                y = df_chain[ycol].astype(float)
                cols = np.where(y >= 0, "#00E676", "#FF5252")
                fig = plt_go.Figure()
                fig.add_trace(plt_go.Bar(
                    x=df_chain["Strike"], y=y, marker_color=cols,
                    marker_line_width=0, width=bar_w, opacity=1.0,
                    hovertemplate="K %{x:.0f}<br>%{y:,.0f}<extra></extra>",
                ))
                fig.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
                fig.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA")
                fig.add_vline(x=lvls.get("Zero_Gamma_Flip", data["spot_price"]), line_dash="dot", line_color="#FF9800")
                ymax = float(y.abs().max() or 1.0)
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="#11151C", plot_bgcolor="#0E1117",
                    height=h, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                    bargap=0.15,
                )
                fig.update_xaxes(range=[min_s, max_s], tickformat="d")
                fig.update_yaxes(range=[-ymax * 1.15, ymax * 1.15], zeroline=True, zerolinecolor="#FFFFFF")
                st.plotly_chart(fig, use_container_width=True, key=key_name)

            t_dg, t_dex = st.tabs(["Δ-GEX (OI)", "DEX / OI"])
            with t_dg:
                _strike_bars("Net_Delta_GEX_OI", "chart_delta_gex")
            with t_dex:
                _strike_bars("DEX_OI", "chart_dex_oi")
        else:
            st.info("GEX unavailable.")

    # Book + tape
    book_l, tape_r = st.columns(2)
    with book_l:
        heading_ribbon("📘 Perp Limit Book Δ", "Resting book size on BTC/ETH-PERPETUAL. Δ = this snap − last snap.")
        book = data.get("perp_book") or {}
        hist = update_liq_from_book(book, currency) if book else list(st.session_state.get("liq_delta_history") or [])
        if hist:
            last = hist[-1]
            bids = [float(h.get("bid_change") or 0) for h in hist]
            asks = [float(h.get("ask_change") or 0) for h in hist]
            bid_s = float(np.std(bids, ddof=1)) if len(bids) > 8 else None
            ask_s = float(np.std(asks, ddof=1)) if len(asks) > 8 else None
            alert = classify_liq_alert(last["bid_change"], last["ask_change"], bid_s, ask_s)
            bq, aq = last["bid_qty_lots"], last["ask_qty_lots"]
            imb = (bq - aq) / (bq + aq) if (bq + aq) else 0
            st.caption(f"Bid {bq:,.1f} / Ask {aq:,.1f} · Imb {imb:+.2f} · {alert['label']}")
            times = [h["ts"].strftime("%H:%M:%S") if hasattr(h["ts"], "strftime") else str(h["ts"]) for h in hist]
            fig = plt_go.Figure()
            nets = [h["net_liq"] for h in hist]
            fig.add_trace(plt_go.Bar(x=times, y=nets, marker_color=["#00E676" if v >= 0 else "#FF5252" for v in nets], opacity=0.7))
            fig.add_trace(plt_go.Scatter(x=times, y=[h["bid_change"] for h in hist], name="Bid Δ", line=dict(color="#2196F3", width=1.5)))
            fig.add_trace(plt_go.Scatter(x=times, y=[h["ask_change"] for h in hist], name="Ask Δ", line=dict(color="#FF9800", width=1.5)))
            fig.add_hline(y=0, line_dash="dot", line_color="#FFF")
            fig.update_layout(template="plotly_dark", height=170, margin=dict(l=4, r=4, t=4, b=24),
                              paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", hovermode="x unified",
                              legend=dict(orientation="h", y=-0.25, font=dict(size=9)))
            fig.update_xaxes(type="category", nticks=5)
            st.plotly_chart(fig, use_container_width=True, key="chart_liq_delta")
        else:
            st.caption("Enable Auto-Refresh to accumulate book snapshots.")

    with tape_r:
        heading_ribbon("📜 Perp tape (last prints)", "public/get_last_trades_by_instrument on the perpetual.")
        trades = fetch_last_trades(PERP[currency], 40)
        if trades:
            rows = []
            for t in trades:
                rows.append({
                    "Time": datetime.datetime.fromtimestamp(t["timestamp"] / 1000, tz=datetime.timezone.utc).astimezone(IST).strftime("%H:%M:%S"),
                    "Side": (t.get("direction") or "—").upper(),
                    "Px": t.get("price"),
                    "Amt": t.get("amount"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=170)
        else:
            st.caption("No trades returned.")

    # VEX/CEX + IV
    if not df_chain.empty:
        st.markdown("---")
        d_left, d_right = st.columns([0.45, 0.55])
        with d_left:
            heading_ribbon(
                "VEX / CEX",
                "<b>VEX (vanna)</b> ≈ −pdf(d1)×d2/σ × OI — dealer vanna vs spot/IV.<br>"
                "<b>CEX (charm)</b> ≈ ∂Δ/∂t × OI — delta decay into expiry.<br>"
                "Positive often supports a pin; large negative supports acceleration.",
            )
            fig_vc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vc.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["VEX"], name="VEX", marker_color="#00E676", opacity=0.75), secondary_y=False)
            fig_vc.add_trace(plt_go.Scatter(x=df_chain["Strike"], y=df_chain["CEX"], name="CEX", line=dict(color="#2196F3", width=2), mode="lines+markers", marker=dict(size=4)), secondary_y=True)
            fig_vc.add_hline(y=0, line_color="#FFF")
            fig_vc.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA")
            fig_vc.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                 height=280, margin=dict(l=8, r=8, t=10, b=8), hovermode="x unified",
                                 legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)))
            st.plotly_chart(fig_vc, use_container_width=True, key="chart_vex_cex")
        with d_right:
            heading_ribbon(
                f"IV SKEW  ·  {data.get('selected_expiry','')}",
                "<b>OTM</b> — puts below spot + calls at/above (clean smile).<br>"
                "<b>Raw</b> — full CE and PE IV curves.<br>"
                "<b>DVOL</b> — Deribit volatility index (VIX analog).<br>"
                "IV from Deribit mark_iv when present, else Brent inversion on USD premium.",
            )
            tab1, tab2, tab3 = st.tabs(["OTM Skew", "Raw CE vs PE", "DVOL"])
            with tab1:
                puts = df_chain[df_chain["Strike"] < data["spot_price"]][["Strike", "P_IV"]].rename(columns={"P_IV": "IV_%"})
                calls = df_chain[df_chain["Strike"] >= data["spot_price"]][["Strike", "C_IV"]].rename(columns={"C_IV": "IV_%"})
                skew = pd.concat([puts, calls]).sort_values("Strike")
                fig_s = px.line(skew, x="Strike", y="IV_%", markers=True, color_discrete_sequence=["#00bfff"])
                fig_s.add_vline(x=data["spot_price"], line_dash="dash", line_color="white")
                fig_s.update_layout(template="plotly_dark", height=260, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                    margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True, key="chart_iv_otm")
            with tab2:
                raw = pd.concat([
                    df_chain[["Strike", "C_IV"]].assign(Option_Type="C").rename(columns={"C_IV": "IV_%"}),
                    df_chain[["Strike", "P_IV"]].assign(Option_Type="P").rename(columns={"P_IV": "IV_%"}),
                ])
                fig_r = px.line(raw, x="Strike", y="IV_%", color="Option_Type", markers=True,
                                color_discrete_map={"C": "#00cc96", "P": "#ff4136"})
                fig_r.add_vline(x=data["spot_price"], line_dash="dash", line_color="white")
                fig_r.update_layout(template="plotly_dark", height=280, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                    margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_r, use_container_width=True, key="chart_iv_raw")
            with tab3:
                dv = data.get("dvol", pd.DataFrame())
                if dv is None or dv.empty:
                    st.info("DVOL series unavailable from Deribit volatility-index endpoint.")
                else:
                    last = float(dv["close"].iloc[-1])
                    st.markdown(f"<div style='font-size:18px;font-weight:800;color:#FFD54F;'>{currency} DVOL {last:.2f}</div>", unsafe_allow_html=True)
                    fig_v = plt_go.Figure()
                    fig_v.add_trace(plt_go.Scatter(x=dv["time"], y=dv["close"], mode="lines", line=dict(color="#FFD54F", width=2)))
                    fig_v.update_layout(template="plotly_dark", height=240, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                        margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True, key="chart_dvol")

    # Basket
    st.markdown("---")
    heading_ribbon("🧺 Basket Greeks", "Δ Γ Θ Vega across legs × contracts. Premiums in USD (coin price × index).")
    if not st.session_state.get("basket_legs"):
        st.caption("No legs — add from the sidebar.")
    else:
        F_val = data.get("F", data["spot_price"])
        T_val = data.get("T", 1e-5)
        hv_val = data.get("index_hv", 0.55)
        rows = []
        tot = dict(pnl=0, delta=0, gamma=0, theta=0, vega=0)
        by_k = {r["Strike"]: r for r in chain}
        for i, leg in enumerate(st.session_state["basket_legs"]):
            k, t, act, q = leg["strike"], leg["type"], leg["action"], leg["qty"]
            rec = min(chain, key=lambda r: abs(r["Strike"] - k)) if chain else {}
            ltp = rec.get("C_LTP" if t == "C" else "P_LTP", 0.0)
            entry = leg["entry_price"] if leg["entry_price"] > 0 else ltp
            iv = (rec.get("C_IV") if t == "C" else rec.get("P_IV", 0)) / 100.0 or hv_val
            g = VolatilityEngine.calculate_greeks(F_val, k, T_val, r_rate, max(iv, 1e-4), "c" if t == "C" else "p")
            mult = 1.0 if act == "BUY" else -1.0
            pnl = ((ltp - entry) if act == "BUY" else (entry - ltp)) * q
            rows.append({
                "Leg": i + 1, "Action": act, "Strike": int(k), "Type": t, "Qty": q,
                "Entry $": round(entry, 2), "Mark $": round(ltp, 2), "P&L $": round(pnl, 2),
                "Δ": round(g["delta"] * q * mult, 3),
                "γ": round(g["gamma"] * q * mult, 6),
                "θ/d": round(g["theta"] * q * mult, 2),
                "ν": round(g["vega"] * q * mult, 2),
            })
            tot["pnl"] += pnl
            tot["delta"] += g["delta"] * q * mult
            tot["gamma"] += g["gamma"] * q * mult
            tot["theta"] += g["theta"] * q * mult
            tot["vega"] += g["vega"] * q * mult
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("P&L $", f"{tot['pnl']:,.0f}")
        m2.metric("Δ", f"{tot['delta']:,.2f}")
        m3.metric("γ", f"{tot['gamma']:.6f}")
        m4.metric("θ/d", f"{tot['theta']:,.1f}")
        m5.metric("ν", f"{tot['vega']:,.1f}")
        tbl, dels = st.columns([0.92, 0.08])
        with tbl:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with dels:
            for i in range(len(st.session_state["basket_legs"])):
                if st.button("✕", key=f"del_{i}"):
                    st.session_state["basket_legs"].pop(i)
                    st.rerun()


live_dashboard()

st.caption(
    "Data: Deribit public REST (`get_instruments`, `get_book_summary_by_currency`, "
    "`get_tradingview_chart_data`, `get_order_book`, `get_last_trades_by_instrument`, "
    "`get_volatility_index_data`, `get_index_price`). "
    "Option marks are coin-denominated on Deribit and converted to USD via the index. "
    "GEX sign convention matches the attached index desk (call γ×OI − put γ×OI). "
    "Not exchange TBT. Not investment advice."
)
