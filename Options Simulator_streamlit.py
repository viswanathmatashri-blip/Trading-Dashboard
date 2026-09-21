import os
from pathlib import Path
import logging
import warnings
import time
import datetime
import math
import json
import re
import random
import numpy as np
import pandas as pd
import scipy.stats as si
import pyotp
import pytz
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as plt_go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from scipy.optimize import brentq
from scipy.stats import norm
from SmartApi import SmartConnect

# Setup .streamlit/config.toml programmatically for dark theme
os.makedirs(".streamlit", exist_ok=True)
config_path = os.path.join(".streamlit", "config.toml")
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        f.write(
            """[theme]
base="dark"
primaryColor="#00E676"
backgroundColor="#0E1117"
secondaryBackgroundColor="#1E222D"
textColor="#FAFAFA"
"""
        )

logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

load_dotenv()

st.set_page_config(
    page_title="Option Chain Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
custom_css = """
<style>
.stApp { opacity: 1 !important; }
[data-testid="stStatusWidget"], .stSpinner { display: none !important; }
div[data-testid="stFragment"] { opacity: 1 !important; }
html, body, [data-testid="stAppViewContainer"] { background-color: #0E1117 !important; color: #FAFAFA !important; }
section[data-testid="stSidebar"] { width: 310px !important; }
.block-container { padding-top: 2.4rem !important; padding-bottom: 0.6rem !important; }
header[data-testid="stHeader"] { background-color: rgba(0, 0, 0, 0) !important; pointer-events: none !important; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a { pointer-events: auto !important; }
h1, h2, h3, .custom-heading { color: #00E676 !important; font-size: 18px !important; font-weight: 700 !important; margin-bottom: 0.15rem !important; }
div[data-testid="stMetricValue"] { font-size: 15px !important; color: #00E676 !important; }
div[data-testid="stMetricLabel"] { font-size: 11px !important; }
div[data-testid="stMetricDelta"] { font-size: 11px !important; }
.update-timestamp { font-size: 12px; color: #00E676; font-weight: 600; text-align: right; }
div[data-baseweb="select"] > div { background-color: #1E222D !important; color: #FAFAFA !important; border-color: #363C4E !important; }
.stButton>button { border-radius: 6px; font-weight: 600; }
.status-badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; margin-right: 6px; }
.badge-bullish { background-color: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; }
.badge-bearish { background-color: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid #FF5252; }
.badge-neutral { background-color: rgba(255, 152, 0, 0.15); color: #FF9800; border: 1px solid #FF9800; }
/* Sticky compact market-summary ribbon */
.sticky-summary {
    position: sticky; top: 0; z-index: 999;
    background: #0E1117; border: 1px solid #2A2F3A; border-radius: 8px;
    padding: 8px 12px 6px 12px; margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.45);
}
.sticky-summary .stMetric { padding: 2px 0 !important; }
.micro-hover { position: relative; display: block; cursor: help; }
.micro-hover .micro-tip {
    display: none; position: absolute; left: 104%; top: 0;
    z-index: 4000; width: 340px; max-width: 42vw;
    background: #1A1F2B; color: #FAFAFA; border: 1px solid #00E676;
    border-radius: 8px; padding: 10px 12px; font-size: 12px; line-height: 1.4;
    box-shadow: 0 8px 24px rgba(0,0,0,0.55); text-align: left;
}
.micro-hover:hover .micro-tip { display: block; }
.micro-float {
    position: relative; z-index: 25;
    margin-top: -355px; margin-bottom: 8px;
    margin-left: auto; margin-right: 2px;
    width: 200px; text-align: center;
}
.micro-chip {
    display: block; margin: 0 auto 6px auto; padding: 6px 8px;
    background: #1A1F2B;
    border: 1px solid #3A4150; border-radius: 8px;
    width: 218px; text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35);
}
.chart-card {
    background: #11151C;
    border: 1px solid #2A3340;
    border-radius: 10px;
    padding: 8px 8px 4px 8px;
    margin: 0 0 8px 0;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
}
.chart-card .card-title {
    color: #8FA4B8; font-size: 11px; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
    margin: 0 0 4px 2px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def _secret_or_env(*names, default=""):
    """Streamlit Cloud secrets first, then env / .env. Supports flat or [smartapi] tables."""
    for name in names:
        try:
            if name in st.secrets:
                val = st.secrets.get(name)
                if val is not None and str(val).strip():
                    return str(val).strip()
        except Exception:
            pass
        try:
            sect = st.secrets.get("smartapi", {})
            if isinstance(sect, dict) and sect.get(name):
                return str(sect.get(name)).strip()
        except Exception:
            pass
        val = os.getenv(name, "")
        if val and str(val).strip():
            return str(val).strip()
    return default


API_KEY = _secret_or_env("API_KEY", "SMARTAPI_KEY", "ANGEL_API_KEY")
CLIENT_CODE = _secret_or_env("CLIENT_CODE", "CLIENTID", "CLIENT_ID")
PIN = _secret_or_env("PIN", "MPIN", "PASSWORD")
TOTP_SECRET = _secret_or_env("TOTP_SECRET", "TOTP", "TOKEN")

# Initialise Session State Variables
if "basket_legs" not in st.session_state:
    st.session_state["basket_legs"] = []
if "selected_timeframe" not in st.session_state:
    st.session_state["selected_timeframe"] = "3 min"
if "chart_window" not in st.session_state:
    st.session_state["chart_window"] = "Session (6h)"
if "replay_session_on" not in st.session_state:
    st.session_state["replay_session_on"] = False
if "replay_session_date" not in st.session_state:
    st.session_state["replay_session_date"] = None
if "px_alert_on" not in st.session_state:
    st.session_state["px_alert_on"] = False
if "px_alert_lvl" not in st.session_state:
    st.session_state["px_alert_lvl"] = 0.0
if "avwap_on" not in st.session_state:
    st.session_state["avwap_on"] = False
if "avwap_time" not in st.session_state:
    st.session_state["avwap_time"] = None
if "gemini_enabled" not in st.session_state:
    st.session_state["gemini_enabled"] = False
if "gemini_interval_min" not in st.session_state:
    st.session_state["gemini_interval_min"] = 5
if "gemini_regular" not in st.session_state:
    st.session_state["gemini_regular"] = ""
if "gemini_trigger" not in st.session_state:
    st.session_state["gemini_trigger"] = ""
if "gemini_regular_ts" not in st.session_state:
    st.session_state["gemini_regular_ts"] = 0.0
if "enable_main_refresh" not in st.session_state:
    st.session_state["enable_main_refresh"] = False
if "enable_zscore_refresh" not in st.session_state:
    st.session_state["enable_zscore_refresh"] = False
if "zscore_data_store" not in st.session_state:
    st.session_state["zscore_data_store"] = pd.DataFrame()
if "heatmap_timeframe" not in st.session_state:
    st.session_state["heatmap_timeframe"] = "5 min"
if "gex_heatmap_history" not in st.session_state:
    st.session_state["gex_heatmap_history"] = []
if "alert_metrics_history" not in st.session_state:
    st.session_state["alert_metrics_history"] = []  # snapshots for Exit / Long alert criteria
if "liq_delta_history" not in st.session_state:
    st.session_state["liq_delta_history"] = []
if "tick_cvd_history" not in st.session_state:
    st.session_state["tick_cvd_history"] = []
if "flow_tape" not in st.session_state:
    st.session_state["flow_tape"] = []
if "multi_index_mode" not in st.session_state:
    st.session_state["multi_index_mode"] = False
if "multi_tf" not in st.session_state:
    st.session_state["multi_tf"] = "15 min"
if "multi_enabled" not in st.session_state:
    st.session_state["multi_enabled"] = {
        "NIFTY": True, "BANKNIFTY": True, "FINNIFTY": False,
        "MIDCPNIFTY": False, "SENSEX": True, "GOLDM": False, "CRUDEOIL": False,
    }
if "multi_store" not in st.session_state:
    st.session_state["multi_store"] = {}
if "multi_gex_ts" not in st.session_state:
    st.session_state["multi_gex_ts"] = 0.0
if "app_view" not in st.session_state:
    st.session_state["app_view"] = "default"

# ---------- Loading status (sidebar) ----------
def update_load_status(msg: str):
    if "load_status_placeholder" not in st.session_state:
        st.session_state["load_status_placeholder"] = st.sidebar.empty()
    try:
        st.session_state["load_status_placeholder"].info(f"⏳ {msg}")
    except Exception:
        pass

def clear_load_status():
    if "load_status_placeholder" in st.session_state:
        try:
            st.session_state["load_status_placeholder"].empty()
        except Exception:
            pass


# Streamlit Cache Persistence Handlers
@st.cache_data(ttl=86400)
def get_cached_basket():
    return []

def save_basket_to_cache(basket):
    get_cached_basket.clear()
    @st.cache_data(ttl=86400)
    def _inner():
        return basket
    _inner()

# LocalStorage Sync via HTML/JS Fragment
def sync_local_storage():
    basket_json = json.dumps(st.session_state["basket_legs"])
    js_code = f"""
    <script>
        // Save current basket state to browser localStorage
        localStorage.setItem("streamlit_strategy_basket", '{basket_json}');
    </script>
    """
    components.html(js_code, height=0, width=0)

LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
    "SENSEX": 20,
    "GOLDM": 100,
    "CRUDEOIL": 100,
}

INDEX_TOKEN_MAP = {
    "NIFTY": ("99926000", "NSE", "NFO"),
    "BANKNIFTY": ("99926009", "NSE", "NFO"),
    "FINNIFTY": ("99926037", "NSE", "NFO"),
    "MIDCPNIFTY": ("99926074", "NSE", "NFO"),
    "SENSEX": ("99919000", "BSE", "BFO"),
    "GOLDM": ("", "MCX", "MCX"),
    "CRUDEOIL": ("", "MCX", "MCX"),
}
INDIA_VIX_TOKEN = "99926017"

# --- RATE LIMIT SAFEGUARD WRAPPER ---
def safe_api_call(func, *args, max_retries=4, base_delay=0.6, **kwargs):
    """Executes SmartAPI calls with dynamic retry logic and exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            res = func(*args, **kwargs)
            if isinstance(res, dict) and not res.get("status"):
                msg = str(res.get("message", "")).lower()
                if "access denied" in msg or "rate" in msg or "exceeding" in msg:
                    raise Exception(f"Rate Limit Hit: {res.get('message')}")
            return res
        except Exception as e:
            err_str = str(e).lower()
            if "access denied" in err_str or "rate" in err_str or "exceeding" in err_str:
                if attempt == max_retries - 1:
                    return None
                sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 0.4)
                time.sleep(sleep_time)
            else:
                return None
    return None

# --- BETA MODULE HELPER FUNCTIONS ---
def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def calculate_implied_volatility(price, S, K, T, r, option_type="CE"):
    if T <= 1e-5 or price <= 0:
        return 0.15
    discounted_K = K * np.exp(-r * T)
    
    def bs_error(sigma):
        d1, d2 = d1_d2(S, K, T, r, sigma)
        if option_type == "CE":
            bs_price = S * si.norm.cdf(d1) - discounted_K * si.norm.cdf(d2)
        else:
            bs_price = discounted_K * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        return bs_price - price

    try:
        f_low = bs_error(0.0001)
        f_high = bs_error(10.0)
        if f_low * f_high <= 0:
            return brentq(bs_error, a=0.0001, b=10.0, xtol=1e-5)
        return 0.01 if f_low > 0 else 2.5
    except Exception:
        return 0.15

def compute_greeks(row, K, r, option_type="CE"):
    S = row['Spot_Price']
    T = row['T']
    price = row['Close']
    
    if T <= 1e-5 or S <= 0:
        return pd.Series([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False])
    
    discounted_K = K * np.exp(-r * T)
    intrinsic = max(0.0, S - discounted_K) if option_type == "CE" else max(0.0, discounted_K - S)
    extrinsic = max(0.0, price - intrinsic)
    is_pure_intrinsic = extrinsic <= 1.0
    
    sigma = calculate_implied_volatility(price, S, K, T, r, option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    
    if option_type == "CE":
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365.0
        charm = (si.norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))) / 365.0
    else:
        theta = (- (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365.0
        charm = (-si.norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))) / 365.0
                 
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vanna = - (si.norm.pdf(d1) * d2) / sigma
    theta_decay_pct = (abs(theta) / price) * 100.0 if price > 0 else 0.0
    
    return pd.Series([sigma * 100.0, theta, theta_decay_pct, gamma, vanna, charm, extrinsic, is_pure_intrinsic])

# MODIFIED: Updated to 1-hour interval for 1-hr data points while keeping daily greeks
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(_api, token, days, spot_token="99926000", spot_exchange="NSE"):
    to_date = datetime.datetime.now()
    from_date = to_date - datetime.timedelta(days=days)
    
    opt_params = {"exchange": "NFO", "symboltoken": str(token), "interval": "ONE_HOUR", 
                  "fromdate": from_date.strftime("%Y-%m-%d 09:15"), "todate": to_date.strftime("%Y-%m-%d 15:30")}
    opt_res = safe_api_call(_api.getCandleData, opt_params)
    time.sleep(0.40)
    
    spot_params = {"exchange": spot_exchange, "symboltoken": str(spot_token), "interval": "ONE_HOUR", 
                   "fromdate": from_date.strftime("%Y-%m-%d 09:15"), "todate": to_date.strftime("%Y-%m-%d 15:30")}
    spot_res = safe_api_call(_api.getCandleData, spot_params)
    time.sleep(0.40)
    
    if opt_res and opt_res.get('data') and spot_res and spot_res.get('data'):
        df_opt = pd.DataFrame(opt_res['data'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_spot = pd.DataFrame(spot_res['data'], columns=['Timestamp', 'Open', 'High', 'Low', 'Close_Spot', 'Vol_Spot'])
        
        df_opt['Timestamp'] = pd.to_datetime(df_opt['Timestamp'])
        df_spot['Timestamp'] = pd.to_datetime(df_spot['Timestamp'])
        
        df = pd.merge(df_opt[['Timestamp', 'Close']], df_spot[['Timestamp', 'Close_Spot']], on='Timestamp', how='inner')
        df.rename(columns={'Close_Spot': 'Spot_Price'}, inplace=True)
        df['Date'] = df['Timestamp'].dt.strftime('%d-%b %H:%M')
        df['Raw_Timestamp'] = df['Timestamp']
        return df.sort_values('Raw_Timestamp', ascending=True).reset_index(drop=True)
    return pd.DataFrame()

# OPTIMIZED: Cached & Paced with BATCH API requests to prevent Rate Limit Exceeded
@st.cache_data(ttl=60, show_spinner=False)
def fetch_and_compute_full_chain_iv(_api, df_master, index_name, exchange, expiry_str, current_spot, r, strikes_below=10, strikes_above=10):
    target_dt = pd.to_datetime(expiry_str, format="%d%b%Y", errors='coerce')
    
    # Filter strictly by Index Name, Exchange, and Selected Expiry
    df_expiry = df_master[
        (df_master['name'] == index_name) & 
        (df_master['exch_seg'] == exchange) & 
        (df_master['expiry_dt'] == target_dt) &
        (df_master['instrumenttype'].isin(['OPTIDX', 'OPTSTK']))
    ].copy()

    if df_expiry.empty:
        return pd.DataFrame()

    df_expiry['strike_clean'] = pd.to_numeric(df_expiry['strike'], errors='coerce') / (100.0 if exchange == "NFO" else 1.0)
    if df_expiry['strike_clean'].max() > 1000000:
        df_expiry['strike_clean'] = df_expiry['strike_clean'] / 100.0

    expiry_dt = df_expiry.iloc[0]['expiry_dt'].date()
    today_dt = datetime.datetime.now().date()
    dte = max((expiry_dt - today_dt).days, 0.001)
    T = dte / 365.0

    all_strikes = sorted(df_expiry['strike_clean'].dropna().unique())
    if not all_strikes:
        return pd.DataFrame()

    atm_strike = min(all_strikes, key=lambda x: abs(x - current_spot))
    atm_idx = all_strikes.index(atm_strike)

    start_idx = max(0, atm_idx - strikes_below)
    end_idx = min(len(all_strikes), atm_idx + strikes_above + 1)
    target_strikes = set(all_strikes[start_idx:end_idx])

    df_filtered = df_expiry[df_expiry['strike_clean'].isin(target_strikes)].copy()

    # Optimized Batch Fetch via getMarketData to avoid hundreds of separate candle requests
    tokens = [str(t) for t in df_filtered['token'].dropna().unique() if str(t) != "nan"]
    market_data = {}
    chunk_size = 40

    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        res = safe_api_call(_api.getMarketData, "FULL", {exchange: chunk})
        if res and res.get("status") and res.get("data") and res["data"].get("fetched"):
            for item in res["data"]["fetched"]:
                market_data[str(item["symbolToken"])] = float(item.get("ltp", 0.0))
        time.sleep(0.40)

    results = []
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        opt_type = "CE" if str(row['symbol']).endswith("CE") else "PE"
        token = str(row['token'])
        strike = row['strike_clean']

        latest_ltp = market_data.get(token, 0.0)
        if latest_ltp <= 1.0:
            continue
            
        iv = calculate_implied_volatility(latest_ltp, current_spot, strike, T, r, option_type=opt_type)
        iv_pct = iv * 100.0
        
        discounted_K = strike * np.exp(-r * T)
        intrinsic = max(0.0, current_spot - discounted_K) if opt_type == "CE" else max(0.0, discounted_K - current_spot)
        extrinsic = max(0.0, latest_ltp - intrinsic)
        is_pure_intrinsic = extrinsic <= 1.0
        is_vol_crash = is_pure_intrinsic or (iv_pct <= 2.0)

        if 2.0 <= iv_pct <= 80.0 or is_vol_crash:
            results.append({
                'Strike': strike,
                'Option_Type': opt_type,
                'LTP': latest_ltp,
                'IV_%': iv_pct,
                'Extrinsic_Val': extrinsic,
                'Is_Pure_Intrinsic': is_pure_intrinsic,
                'Vol_Crash_Flag': is_vol_crash,
                'Expiry': expiry_str,
                'Token': token
            })

    df_chain_iv = pd.DataFrame(results)
    if df_chain_iv.empty:
        return pd.DataFrame()

    return df_chain_iv.sort_values('Strike', ascending=True)

def get_clean_otm_skew(df_chain_iv, current_spot):
    puts_otm = df_chain_iv[(df_chain_iv['Option_Type'] == 'PE') & (df_chain_iv['Strike'] < current_spot)].copy()
    calls_otm = df_chain_iv[(df_chain_iv['Option_Type'] == 'CE') & (df_chain_iv['Strike'] >= current_spot)].copy()
    df_skew = pd.concat([puts_otm, calls_otm]).sort_values('Strike').reset_index(drop=True)
    return df_skew

def plot_line_chart(df, y_col, title, y_label, color="#1f77b4"):
    fig = px.line(df, x='Date', y=y_col, title=title, markers=True,
                  hover_data=['Date', 'Spot_Price', 'Close', 'Extrinsic_Val', 'Is_Pure_Intrinsic'])
    fig.update_traces(line_color=color)
    fig.update_xaxes(type="category", autorange=True, title="Date (Oldest ➔ Present/Today)")
    fig.update_yaxes(title=y_label)
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- VOLATILITY & GREEKS ENGINE ---
class VolatilityEngine:
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

    @classmethod
    def black_scholes_price(cls, S: float, K: float, T: float, r: float, sigma: float, flag: str = "c") -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if flag.lower() == "c":
            return S * cls._norm_cdf(d1) - K * math.exp(-r * T) * cls._norm_cdf(d2)
        return K * math.exp(-r * T) * cls._norm_cdf(-d2) - S * cls._norm_cdf(-d1)

    @classmethod
    def calculate_greeks(cls, S: float, K: float, T: float, r: float, sigma: float, flag: str = "c") -> dict:
        if T <= 0 or sigma <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "charm": 0.0}

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf_d1 = cls._norm_pdf(d1)

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = (S * pdf_d1 * math.sqrt(T)) / 100.0

        if flag.lower() == "c":
            delta = cls._norm_cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls._norm_cdf(d2)) / 365.0
            charm = (pdf_d1 * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))) / 365.0
        else:
            delta = cls._norm_cdf(d1) - 1.0
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls._norm_cdf(-d2)) / 365.0
            charm = (-pdf_d1 * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))) / 365.0

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "charm": charm}

    @classmethod
    def calculate_iv(cls, market_price: float, S: float, K: float, T: float, r: float = 0.10, flag: str = "c") -> float:
        if market_price <= 0.05 or T <= 0:
            return 0.0
        df = math.exp(-r * T)
        intrinsic = max(0.0, S - K * df) if flag.lower() == "c" else max(0.0, K * df - S)

        if market_price <= intrinsic:
            return 0.0

        def objective_function(sigma: float) -> float:
            return cls.black_scholes_price(S, K, T, r, sigma, flag) - market_price

        try:
            return float(brentq(objective_function, a=1e-4, b=10.0, xtol=1e-4))
        except (ValueError, RuntimeError):
            return 0.0

    @classmethod
    def calculate_hv(cls, smart_api, symbol_token: str, exchange: str = "NSE", days: int = 30) -> float:
        try:
            to_date = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
            from_date = to_date - datetime.timedelta(days=int(days * 1.6) + 10)

            param = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": "ONE_DAY",
                "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
                "todate": to_date.strftime("%Y-%m-%d 15:30")
            }
            hist_data = safe_api_call(smart_api.getCandleData, param)
            time.sleep(0.40)
            if hist_data and hist_data.get("status") and hist_data.get("data"):
                df_hist = pd.DataFrame(hist_data["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df_hist["close"] = df_hist["close"].astype(float)
                df_hist = df_hist.tail(days + 1)
                log_returns = np.log(df_hist["close"] / df_hist["close"].shift(1)).dropna()

                if len(log_returns) >= 5:
                    return float(np.std(log_returns, ddof=1) * np.sqrt(252))
        except Exception:
            pass
        return 0.15

    @staticmethod
    def calculate_max_pain(chain_data: list) -> int:
        strikes = [row["Strike"] for row in chain_data]
        if not strikes:
            return 0

        losses = {}
        for spot_hypo in strikes:
            total_loss = 0.0
            for row in chain_data:
                k = row["Strike"]
                c_oi = row["C_OI"]
                p_oi = row["P_OI"]

                if spot_hypo > k:
                    total_loss += (spot_hypo - k) * c_oi
                if spot_hypo < k:
                    total_loss += (k - spot_hypo) * p_oi

            losses[spot_hypo] = total_loss

        return min(losses, key=losses.get)

# --- OPTION CHAIN LEVEL CALCULATOR ---
def calculate_support_resistance_targets(chain_data: list, spot_price: float, max_pain: int):
    if not chain_data:
        return {}

    df = pd.DataFrame(chain_data)

    res_df = df.sort_values(by="C_OI", ascending=False)
    r1 = res_df.iloc[0]["Strike"] if len(res_df) > 0 else spot_price
    r2 = res_df.iloc[1]["Strike"] if len(res_df) > 1 else r1

    sup_df = df.sort_values(by="P_OI", ascending=False)
    s1 = sup_df.iloc[0]["Strike"] if len(sup_df) > 0 else spot_price
    s2 = sup_df.iloc[1]["Strike"] if len(sup_df) > 1 else s1

    df_sorted = df.sort_values(by="Strike").reset_index(drop=True)

    above_spot = df_sorted[df_sorted["Strike"] >= spot_price]
    below_spot = df_sorted[df_sorted["Strike"] < spot_price]

    gex_res_row = above_spot.sort_values(by="Net_GEX_OI", ascending=False).head(1) if not above_spot.empty else pd.DataFrame()
    gex_resistance = gex_res_row.iloc[0]["Strike"] if not gex_res_row.empty and gex_res_row.iloc[0]["Net_GEX_OI"] > 0 else r1

    gex_sup_row = below_spot.sort_values(by="Net_GEX_OI", ascending=False).head(1) if not below_spot.empty else pd.DataFrame()
    gex_support = gex_sup_row.iloc[0]["Strike"] if not gex_sup_row.empty and gex_sup_row.iloc[0]["Net_GEX_OI"] > 0 else s1

    neg_gex_row = df_sorted.sort_values(by="Net_GEX_OI", ascending=True).head(1)
    gex_accelerator = neg_gex_row.iloc[0]["Strike"] if not neg_gex_row.empty and neg_gex_row.iloc[0]["Net_GEX_OI"] < 0 else None

    flip_candidates = []
    for i in range(1, len(df_sorted)):
        prev_gex = df_sorted.loc[i-1, "Net_GEX_OI"]
        curr_gex = df_sorted.loc[i, "Net_GEX_OI"]

        if (prev_gex < 0 and curr_gex >= 0) or (prev_gex >= 0 and curr_gex < 0):
            chosen_strike = df_sorted.loc[i, "Strike"] if abs(curr_gex) < abs(prev_gex) else df_sorted.loc[i-1, "Strike"]
            flip_candidates.append(chosen_strike)

    zero_gamma_strike = min(flip_candidates, key=lambda x: abs(x - spot_price)) if flip_candidates else spot_price

    atm_row = df.iloc[(df['Strike'] - spot_price).abs().argsort()[:1]].iloc[0]
    atm_straddle_cost = atm_row["C_LTP"] + atm_row["P_LTP"]

    target_upside = round(spot_price + atm_straddle_cost, 2)
    target_downside = round(spot_price - atm_straddle_cost, 2)

    return {
        "S1": s1, "S2": s2, "R1": r1, "R2": r2,
        "GEX_Support": gex_support,
        "GEX_Resistance": gex_resistance,
        "GEX_Accelerator": gex_accelerator,
        "Zero_Gamma_Flip": zero_gamma_strike,
        "MaxPain": max_pain,
        "Target_Up": target_upside,
        "Target_Down": target_downside,
        "Straddle_Cost": round(atm_straddle_cost, 2)
    }



ARROW_GLYPH = {"u": "↑", "f": "→", "d": "↓"}
LEVEL_GLYPH = {"++": "⇈", "+": "↑", "-": "↓", "--": "⇊", "=": "→"}


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


# --- VALUE-AREA REGIME PLAYBOOK (replaces PDEC / Flow18 / candle books) ---
# Four models:
#   M1-S  VAH mean-reversion SHORT   (range)
#   M1-L  VAL mean-reversion LONG    (range)
#   M2-L  VAH acceptance TREND LONG  (trend)
#   M2-S  VAL acceptance TREND SHORT (trend)
# Location tests are statistical (z vs session σ, persistence, ATR buffer).

VA_PLAYBOOK = {
    "M1S_WATCH": (
        "Range · VAH probe — price statistically above value, absorption forming",
        "WATCH SHORT (VAH fade)",
    ),
    "M1S_ENTRY": (
        "Range · VAH mean-reversion SHORT — absorption + EFI divergence at/above VAH",
        "SHORT MEAN-REVERSION (Target: POC → VAL)",
    ),
    "M1S_ADD": (
        "Range · VAH failed acceptance — close back inside value, ΔV still offered",
        "ADD SHORT (Retest VAH from below)",
    ),
    "M1L_WATCH": (
        "Range · VAL probe — price statistically below value, demand absorbing",
        "WATCH LONG (VAL bounce)",
    ),
    "M1L_ENTRY": (
        "Range · VAL mean-reversion LONG — absorption + EFI divergence at/below VAL",
        "LONG MEAN-REVERSION (Target: POC → VAH)",
    ),
    "M1L_ADD": (
        "Range · VAL failed breakdown — close back inside value, ΔV still bid",
        "ADD LONG (Retest VAL from above)",
    ),
    "M2L_WATCH": (
        "Trend · VAH break — price holding above value, force not yet confirmed",
        "WATCH LONG (VAH acceptance)",
    ),
    "M2L_ENTRY": (
        "Trend · VAH acceptance LONG — density building above VAH + EFI expansion + ΔV bid",
        "LONG BREAKOUT (Hold above VAH / trail under box)",
    ),
    "M2L_ADD": (
        "Trend · VAH retest from above — shallow pullback, ΔV stays non-negative",
        "ADD LONG (Retest old VAH / new box low)",
    ),
    "M2S_WATCH": (
        "Trend · VAL break — price holding below value, force not yet confirmed",
        "WATCH SHORT (VAL acceptance)",
    ),
    "M2S_ENTRY": (
        "Trend · VAL acceptance SHORT — density building below VAL + EFI expansion + ΔV offered",
        "SHORT BREAKDOWN (Hold below VAL / trail above box)",
    ),
    "M2S_ADD": (
        "Trend · VAL retest from below — shallow bounce, ΔV stays non-positive",
        "ADD SHORT (Retest old VAL / new box high)",
    ),
    "INSIDE": (
        "Inside value — no edge at the edge of the profile",
        "NO ENTRY",
    ),
    "CHOP": (
        "Regime mixed / low efficiency — neither clean range nor trend",
        "NO ENTRY",
    ),
}


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
        p = float(2.0 * si.t.sf(abs(t), dof))
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


def classify_market_regime(dfi: pd.DataFrame, data: dict = None) -> dict:
    """Range vs trend using profile density, efficiency, GEX, VWAP stretch, ATR."""
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

    # Range score: high density in value, low efficiency, long-gamma / pin, mid VWAP
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
    """Statistical location vs value area. Buffer = max(0.35σ, 0.35 ATR)."""
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
    efi_div_up = px_sl > 0 and efi_sl <= 0  # higher prices, EFI not expanding
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
        # chop: only fire watch at statistically extreme location
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
    hover = (
        f"{glyphs}<br><b>{action}</b><br>{micro}<br>{note}<br>"
        f"VAH {vah} VAL {val} POC {poc}"
    )
    return {
        "ok": True, "key": (p_arr, d_arr, e_arr, c_arr), "arrows": glyphs,
        "micro": micro + " · " + note, "action": action,
        "price": p_arr, "delta": d_arr, "efi": e_arr, "cvd": c_arr, "obv": d_arr,
        "hover": hover, "efi_zero": False, "efi_note": note,
        "regime": reg["regime"], "model": code, "loc": loc, "loc_z": loc_z,
        "vah": vah, "val": val, "poc": poc,
    }


def classify_microstructure(dfi: pd.DataFrame, data: dict = None) -> dict:
    """Value-area models (range fade vs trend acceptance). Signature kept for call sites."""
    rec = classify_va_setup(dfi, data if data is not None else st.session_state.get("data_store"))
    return rec


def pdec_session_history(dfi: pd.DataFrame, min_bars: int = 16) -> list:
    """Causal VA action. Incremental + last-60-bar window (full-day loop was the load stall)."""
    out = []
    if dfi is None or dfi.empty:
        return out
    n = len(dfi)
    data = st.session_state.get("data_store") if "st" in dir() else None
    sig = (
        str(dfi["time"].iloc[-1]) if "time" in dfi.columns else n,
        int(n),
        float(dfi["close"].iloc[-1]),
    )
    prev_sig = st.session_state.get("_va_hist_sig")
    prev_out = list(st.session_state.get("_va_hist") or [])
    start = min_bars - 1
    if prev_sig and prev_out and prev_sig[1] <= n and prev_sig[0] != sig[0]:
        start = max(start, int(prev_out[-1]["i"]))
        out = [r for r in prev_out if r["i"] < n]
    elif prev_sig == sig and prev_out:
        return prev_out
    for i in range(start, n):
        sl = dfi.iloc[max(0, i + 1 - 60): i + 1]
        try:
            rec = classify_va_setup(sl, data)
        except Exception:
            continue
        if not rec.get("ok"):
            continue
        act = rec.get("action") or ""
        hi = sl["high"].iloc[-1] if "high" in sl.columns else sl["close"].iloc[-1]
        t = sl["time_str"].iloc[-1] if "time_str" in sl.columns else str(i)
        out.append({
            "i": i, "t": t, "action": act,
            "key": rec.get("key"), "micro": rec.get("micro") or "",
            "glyphs": rec.get("arrows") or "",
            "y": float(hi) if pd.notna(hi) else None,
            "model": rec.get("model"),
        })
    st.session_state["_va_hist"] = out
    st.session_state["_va_hist_sig"] = sig
    return out


def classify_flow_playbook(dfi: pd.DataFrame, data: dict) -> dict:
    """Compat shim — flow-18 book removed. Surface the VA model as flow chip."""
    rec = classify_va_setup(dfi, data)
    if not rec.get("ok"):
        return {"ok": False, "action": "", "micro": "", "hover": ""}
    return {
        "ok": True, "action": rec.get("action") or "NO ENTRY",
        "micro": rec.get("micro") or "", "hover": rec.get("hover") or "",
        "price": "f", "cvd": "f", "dex": "f", "prem": "f",
    }


def detect_candle_pattern(df: pd.DataFrame) -> dict:
    """Candle book removed — kept as inert stub for leftover call sites."""
    return {"ok": False, "name": "—", "bias": 0, "conf": "Low", "note": "candle book removed"}


def confirm_pdec_with_candle(action: str, pat: dict) -> tuple:
    return action or "NO ENTRY", "VA model (no candle gate)"


def _telegram_creds():
    tok = (os.getenv("TELE_BOTTOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.getenv("TELE_CHATID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    return tok, chat


def send_telegram_alert(text: str) -> bool:
    tok, chat = _telegram_creds()
    if not tok or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text, "disable_web_page_preview": True},
            timeout=8,
        )
        return r.ok
    except Exception:
        return False


def process_telegram_alerts(data, dfi, scores, micro, flow, cvd_st, index_name=None):
    """Edge-triggered alerts. No token in logs. Skip when market closed."""
    idx = index_name or st.session_state.get("_last_index") or "INDEX"
    live, _, _, _ = market_session_state(index_name=idx)
    if not live:
        return
    tok, chat = _telegram_creds()
    if not tok or not chat:
        return
    if dfi is None or getattr(dfi, "empty", True):
        return
    prev = st.session_state.get("tg_prev") or {}
    now_ts = time.time()
    cool = dict(st.session_state.get("tg_cool") or {})

    def cooled(key, sec=180):
        last = cool.get(key, 0)
        if now_ts - last < sec:
            return True
        cool[key] = now_ts
        return False

    spot = float(data.get("spot_price") or dfi.get("spot_px", dfi["close"]).iloc[-1] or 0)
    px = float(dfi["spot_px"].iloc[-1]) if "spot_px" in dfi.columns and pd.notna(dfi["spot_px"].iloc[-1]) else spot
    vw = None
    if "vwap_idx" in dfi.columns:
        vw = float(dfi["vwap_idx"].iloc[-1])
    elif "vwap" in dfi.columns:
        vw = float(dfi["vwap"].iloc[-1])
    efi = float(dfi["efi13"].iloc[-1]) if "efi13" in dfi.columns else None
    flip = float((data.get("levels") or {}).get("Zero_Gamma_Flip") or 0)
    peco = (micro or {}).get("action") or "—"
    peco_arr = (micro or {}).get("arrows") or ""
    pcd = (flow or {}).get("action") or "—"
    header = (
        f"NIFTY {px:,.1f}\n"
        f"PECO {peco_arr}  {peco}\n"
        f"PCD$ {pcd}\n"
    )
    events = []

    if vw:
        side = "ABOVE" if px >= vw else "BELOW"
        prev_side = prev.get("vwap_side")
        if prev_side and side != prev_side:
            if not cooled("vwap"):
                events.append(f"VWAP CROSS {prev_side} → {side}  (VWAP {vw:,.1f})")
        prev["vwap_side"] = side

    if efi is not None:
        efi_side = "POS" if efi >= 0 else "NEG"
        if prev.get("efi_side") and prev["efi_side"] != efi_side:
            if not cooled("efi"):
                events.append(f"EFI FLIP {prev['efi_side']} → {efi_side}  (EFI {efi:,.0f})")
        prev["efi_side"] = efi_side

    if peco and peco != prev.get("peco"):
        interesting = any(w in peco.upper() for w in (
            "SHORT", "LONG", "PREPARE", "BREAKOUT", "REVERSION", "MEAN", "CAUTIOUS", "PRE-"))
        if interesting and peco.upper() != "NO ENTRY" and not cooled("peco"):
            events.append(f"PECO  {prev.get('peco') or '—'} → {peco}")
        prev["peco"] = peco

    if pcd and pcd != prev.get("pcd"):
        interesting = any(w in pcd.upper() for w in ("PREPARE", "ENTER", "LONG", "SHORT"))
        if False and interesting and "NO ENTRY" not in pcd.upper() and not cooled("pcd"):
            events.append(f"PCD$  {prev.get('pcd') or '—'} → {pcd}")
        prev["pcd"] = pcd

    std = None
    if "vwap_std" in dfi.columns:
        std = float(dfi["vwap_std"].iloc[-1] or 0)
    elif "vwap_upper_idx" in dfi.columns and vw:
        std = abs(float(dfi["vwap_upper_idx"].iloc[-1]) - vw) / max(float(st.session_state.get("vwap_sigma_mult") or 1.5), 0.5)
    if vw and std and std > 0:
        z = (px - vw) / std
        band = "INSIDE"
        if abs(z) >= 2:
            band = "2SIG+"
        elif abs(z) >= 1:
            band = "1SIG+"
        if prev.get("sigma_band") and band != prev["sigma_band"]:
            if band != "INSIDE" and not cooled("sigma"):
                events.append(f"SIGMA  {prev['sigma_band']} → {band}  z={z:+.2f}")
        prev["sigma_band"] = band

    tgt = st.session_state.get("px_alert_lvl")
    if st.session_state.get("px_alert_on") and tgt:
        tgt = float(tgt)
        hit_side = "ABOVE" if px >= tgt else "BELOW"
        prev_hit = st.session_state.get("px_alert_side")
        if prev_hit and hit_side != prev_hit and not cooled("pxlvl", 60):
            events.append(f"PRICE ALERT  hit {tgt:,.1f}  ({prev_hit} → {hit_side})")
            st.session_state["px_alert_on"] = False
        st.session_state["px_alert_side"] = hit_side

    if flip:
        flip_side = "ABOVE_FLIP" if px >= flip else "BELOW_FLIP"
        if prev.get("flip_side") and flip_side != prev["flip_side"]:
            if not cooled("flip"):
                events.append(f"GAMMA FLIP CROSS  {prev['flip_side']} → {flip_side}  flip={flip:,.1f}")
        prev["flip_side"] = flip_side

    div = (cvd_st or {}).get("div") or ""
    if div.startswith("HIGH-PROB") and div != prev.get("cvd_div"):
        if not cooled("cvd_div", 300):
            events.append(f"CVD {div}\n{(cvd_st or {}).get('div_note') or ''}")
        prev["cvd_div"] = div
    else:
        prev["cvd_div"] = div

    st.session_state["tg_prev"] = prev
    st.session_state["tg_cool"] = cool
    digest = build_gemini_digest(data, dfi, scores if isinstance(scores, dict) else {}, micro, flow, cvd_st)
    for ev in events:
        gfb = ""
        try:
            gfb = gemini_trigger_feedback(ev, digest)
        except Exception:
            gfb = ""
        msg = f"OS ALERT\n{ev}\n{header}"
        if gfb:
            msg += "\n--- Trigger feedback ---\n" + gfb[:2500]
        send_telegram_alert(msg)



GEMINI_TRIGGER_MODELS = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite",
    "gemini-2.5-flash", "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
]
GEMINI_REGULAR_MODELS = [
    "gemini-3.5-flash-lite", "gemini-3.5-flash",
    "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash",
    "gemini-3.1-flash-lite",
]


def _gemini_key():
    try:
        return _secret_or_env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    except Exception:
        return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def gemini_generate(prompt: str, models: list) -> tuple:
    key = _gemini_key()
    if not key:
        return "", "NO_KEY"
    last_err = ""
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            r = requests.post(
                url,
                params={"key": key},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1800}},
                timeout=25,
            )
            if r.status_code == 429:
                last_err = f"{model} 429"
                continue
            if r.status_code >= 400:
                last_err = f"{model} {r.status_code}"
                continue
            js = r.json()
            parts = (((js.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
            txt = "".join(p.get("text", "") for p in parts).strip()
            if txt:
                return txt, model
        except Exception as e:
            last_err = f"{model} {e}"
            continue
    return "", last_err or "ALL_FAILED"


def _ser_stats(s, name):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return f"{name}: NA"
    last = float(s.iloc[-1])
    first = float(s.iloc[0])
    return (f"{name}: last={last:.2f} sess_chg={last-first:.2f} "
            f"min={float(s.min()):.2f} max={float(s.max()):.2f} "
            f"sign={'+' if last>=0 else '-'}")


def build_gemini_digest(data, dfi, scores, micro, flow, cvd_st) -> str:
    lv = data.get("levels") or {}
    px = data.get("spot_price")
    lines = [
        f"Index {data.get('index_name', '')} expiry {data.get('selected_expiry', '')}",
        f"Spot {px} Fut {data.get('F')} basis {data.get('basis_info', {}).get('basis')}",
        f"Flip {lv.get('Zero_Gamma_Flip')} GEX_sup {lv.get('GEX_Support')} GEX_res {lv.get('GEX_Resistance')}",
        f"Net GEX OI {data.get('total_net_gex_oi')} Net GEX Vol {data.get('total_net_gex_vol')} PCR {data.get('pcr')}",
        f"Scores composite {scores.get('composite') if scores else ''} bias {scores.get('bias') if scores else ''}",
        f"PECO {(micro or {}).get('arrows')} {(micro or {}).get('action')} {(micro or {}).get('micro')}",
        f"PCD$ {(flow or {}).get('action')} {(flow or {}).get('micro')}",
        f"CVD div {(cvd_st or {}).get('div')} {(cvd_st or {}).get('div_note')}",
        f"DirTrigger {(scores or {}).get('dir_trigger', {})}",
    ]
    if dfi is not None and not getattr(dfi, "empty", True):
        last = dfi.iloc[-1]
        vw = last.get("vwap_idx", last.get("vwap"))
        sp = last.get("spot_px", last.get("close"))
        std = last.get("vwap_std")
        z = ""
        try:
            if vw is not None and std and float(std) > 0 and sp is not None:
                z = f" zVWAP={(float(sp)-float(vw))/float(std):+.2f}"
        except Exception:
            z = ""
        lines.append(
            f"Last bar t={last.get('time_str')} spot={sp} fut={last.get('close')} "
            f"vwap_idx={last.get('vwap_idx')} vwap_fut={last.get('vwap')}{z}"
        )
        for col, name in (("efi13", "EFI13"), ("cvd", "CVD"), ("obv", "OBV")):
            if col in dfi.columns:
                lines.append(_ser_stats(dfi[col], name))
        if "obv_ma20" in dfi.columns:
            lines.append(f"OBV vs MA20: last={float(dfi['obv'].iloc[-1]):.1f} ma={float(dfi['obv_ma20'].iloc[-1]):.1f}")
        # Intra-day path (evenly sampled) so slope/divergence is visible
        n = len(dfi)
        step = max(1, n // 16)
        path = []
        for i in range(0, n, step):
            r = dfi.iloc[i]
            path.append(
                f"{r.get('time_str','?')} S={float(r.get('spot_px', r.get('close',0)) or 0):.0f}"
                f" V={float(r.get('vwap_idx', r.get('vwap',0)) or 0):.0f}"
                f" EFI={float(r.get('efi13',0) or 0):.0f}"
                f" CVD={float(r.get('cvd',0) or 0):.0f}"
                f" OBV={float(r.get('obv',0) or 0):.0f}"
            )
        r = dfi.iloc[-1]
        lastp = (
            f"{r.get('time_str','?')} S={float(r.get('spot_px', r.get('close',0)) or 0):.0f}"
            f" V={float(r.get('vwap_idx', r.get('vwap',0)) or 0):.0f}"
            f" EFI={float(r.get('efi13',0) or 0):.0f}"
            f" CVD={float(r.get('cvd',0) or 0):.0f}"
            f" OBV={float(r.get('obv',0) or 0):.0f}"
        )
        if path[-1] != lastp:
            path.append(lastp)
        lines.append("PATH " + " | ".join(path))
        if cvd_st:
            lines.append(
                f"CVD_STATS px_slope={cvd_st.get('px_slope')} p={cvd_st.get('px_p')} "
                f"cvd_slope={cvd_st.get('cvd_slope')} p={cvd_st.get('cvd_p')} "
                f"rho={cvd_st.get('spearman')} div={cvd_st.get('div')}"
            )
    tape = list(st.session_state.get("flow_tape") or [])
    if tape:
        t = tape[-1]
        lines.append(
            f"DEX last={t.get('dex')} CallPrem={t.get('prem_c')} PutPrem={t.get('prem_p')} "
            f"snaps={len(tape)} last_ts={t.get('ts')}"
        )
        if len(tape) >= 3:
            dex = pd.Series([x.get("dex", 0) for x in tape], dtype=float)
            lines.append(_ser_stats(dex, "DEX_tape"))
    else:
        lines.append("DEX/premium tape: EMPTY (no intra-day snaps)")
    return "\n".join(str(x) for x in lines)


def _scalper_tape_digest(df, name: str) -> str:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return f"{name}: NO TAPE"
    d = df.copy()
    if "cvd" not in d.columns or "efi13" not in d.columns:
        try:
            d = attach_bar_flow(d, rebuild=True)
        except Exception:
            pass
    last = d.iloc[-1]
    first = d.iloc[0]
    cl = pd.to_numeric(d["close"], errors="coerce")
    hi = float(pd.to_numeric(d.get("high", cl), errors="coerce").max())
    lo = float(pd.to_numeric(d.get("low", cl), errors="coerce").min())
    last_px = float(cl.iloc[-1])
    open_px = float(pd.to_numeric(d.get("open", cl), errors="coerce").iloc[0])
    chg = last_px - open_px
    pct = (chg / open_px * 100.0) if open_px else 0.0
    va = ""
    try:
        rec = classify_microstructure(d)
        va = f"{rec.get('regime','')} | {rec.get('action','')} | VAH {rec.get('vah')} VAL {rec.get('val')} POC {rec.get('poc')}"
    except Exception:
        va = ""
    path = []
    n = len(d)
    step = max(1, n // 12)
    for i in range(0, n, step):
        r = d.iloc[i]
        t = r.get("time_str") or str(r.get("time", ""))[-5:]
        path.append(
            f"{t} px={float(r.get('close',0) or 0):.2f}"
            f" efi={float(r.get('efi13',0) or 0):.1f}"
            f" cvd={float(r.get('cvd',0) or 0):.0f}"
        )
    r = d.iloc[-1]
    path.append(
        f"{r.get('time_str', 'now')} px={last_px:.2f}"
        f" efi={float(r.get('efi13',0) or 0):.1f}"
        f" cvd={float(r.get('cvd',0) or 0):.0f}"
    )
    vw = last.get("vwap") or last.get("vwap_idx")
    return "\n".join([
        f"{name}: last={last_px:.2f} open={open_px:.2f} chg={chg:+.2f} ({pct:+.2f}%) high={hi:.2f} low={lo:.2f} vwap={vw}",
        f"{name} VA: {va}",
        _ser_stats(d.get("efi13"), f"{name} EFI") if "efi13" in d.columns else f"{name} EFI: NA",
        _ser_stats(d.get("cvd"), f"{name} CVD") if "cvd" in d.columns else f"{name} CVD: NA",
        f"{name} PATH: " + " | ".join(path[-14:]),
    ])


def build_scalper_gemini_digest() -> str:
    data = st.session_state.get("data_store") or {}
    lv = data.get("levels") or {}
    lines = [
        f"INDEX {Index_Name} expiry {data.get('selected_expiry') or selected_expiry_str}",
        f"Spot {data.get('spot_price')} Fut {data.get('F')} ATM strike {data.get('atm_strike')}",
        f"Flip {lv.get('Zero_Gamma_Flip')} GEX_sup {lv.get('GEX_Support')} GEX_res {lv.get('GEX_Resistance')} MaxPain {data.get('max_pain_strike')}",
        f"Net GEX OI {data.get('total_net_gex_oi')} PCR {data.get('pcr')} Straddle {lv.get('Straddle_Cost')}",
        f"TF {st.session_state.get('selected_timeframe')}",
        _scalper_tape_digest(st.session_state.get("_scalp_spot_df"), "SPOT/NIFTY"),
        _scalper_tape_digest(st.session_state.get("_scalp_ce_df"), "ATM CE"),
        _scalper_tape_digest(st.session_state.get("_scalp_pe_df"), "ATM PE"),
    ]
    return "\n".join(str(x) for x in lines)


def maybe_gemini_scalper_setups():
    if st.session_state.get("app_view") != "scalper":
        return
    if not st.session_state.get("gemini_enabled"):
        return
    if not _gemini_key():
        st.session_state["gemini_regular"] = "No GEMINI_API_KEY in Streamlit secrets."
        return
    now = time.time()
    last = float(st.session_state.get("gemini_regular_ts") or 0)
    wait_s = int(float(st.session_state.get("gemini_interval_min") or 5) * 60)
    if now - last < wait_s:
        st.session_state["gemini_regular_wait"] = int(wait_s - (now - last))
        return
    if st.session_state.get("gemini_in_flight"):
        return
    st.session_state["gemini_in_flight"] = True
    digest = build_scalper_gemini_digest()
    prompt = (
        "You are an options scalper for Indian index options (NIFTY/BANKNIFTY/SENSEX etc). "
        "Use ONLY the DATA snapshot of today's SPOT tape, ATM CE tape and ATM PE tape. "
        "Do not invent prices that are not in DATA. Prefer high-probability structures "
        "(breakout, breakdown, mean-reversion at VA/VWAP). "
        "If there is no edge write a single setup titled NO TRADE.\n\n"
        "Output ONLY trade setups. Do NOT write MARKET STATE, PECO, PCD$, PLAYERS, or BIAS sections.\n"
        "Output 1 or 2 setups MAX. Each setup MUST use this exact numbered shape and nothing else:\n"
        "SETUP n\n"
        "1. Type : <Long BO | Short BD | Long mean-reversion | Short mean-reversion | NO TRADE>\n"
        "2. Trigger : <Nifty/spot close above/below a level from DATA>\n"
        "3. Entry : <ATM CE or ATM PE price band, e.g. 88-100 ATM CE>\n"
        "4. Target : <price> (<pct %>)\n"
        "5. Stoploss : <price> (<pct %>)\n"
        "6. Risk : Reward : <ratio like 1 : 2.1>\n"
        "7. Logic behind setup : <3-5 short sentences using spot VA/VWAP/EFI/CVD and option tape>\n\n"
        "Rules: Entry, target, stop must be on the SAME option (CE or PE). "
        "Percentages vs the mid of the entry band. R:R = reward pct / risk pct. "
        "If CE is missing say so and use PE or NO TRADE.\n\nDATA:\n" + digest
    )
    try:
        txt, model = gemini_generate(prompt, GEMINI_REGULAR_MODELS)
        st.session_state["gemini_regular_ts"] = time.time()
        if txt:
            st.session_state["gemini_regular"] = f"[{model}]\n{txt}"
        else:
            st.session_state["gemini_regular"] = f"(no model answered: {model})"
    finally:
        st.session_state["gemini_in_flight"] = False
        st.session_state["gemini_regular_wait"] = wait_s


def _parse_gemini_setups(text: str) -> list:
    raw = str(text or "")
    raw = raw.split("\n", 1)[1] if raw.startswith("[") and "]\n" in raw[:80] else raw
    raw = raw.replace("\r", "")
    if not raw.strip():
        return []
    chunks = re.split(r"(?:^|\n)\s*(?:SETUP\s*\d+\s*[:.\-]?\s*)", raw, flags=re.IGNORECASE)
    parts = [c.strip() for c in chunks if c.strip() and re.search(r"1\.\s*Type|Type\s*:", c, re.I)]
    if not parts:
        parts = [p.strip() for p in re.split(r"\n(?=\s*1\.\s*Type)", raw) if p.strip()]
    if not parts and raw.strip():
        parts = [raw.strip()]
    out = []
    for p in parts[:2]:
        if re.search(r"MARKET STATE|PECO|PCD\$", p, re.I) and not re.search(r"1\.\s*Type", p, re.I):
            continue
        out.append(p)
    return out


def _setup_card_html(text: str, idx: int) -> str:
    safe = (
        str(text or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    short = safe.upper()
    col = "#FF5252" if "SHORT" in short else ("#00E676" if "LONG" in short else "#90CAF9")
    return (
        f"<div style='font-size:11px;line-height:1.35;white-space:pre-wrap;"
        f"background:#11151C;border:1px solid {col};border-radius:8px;"
        f"padding:8px 8px;color:#ECEFF1;max-height:620px;overflow:auto;'>"
        f"<div style='color:{col};font-weight:800;margin-bottom:4px;'>SETUP {idx}</div>"
        f"{safe}</div>"
    )


def maybe_gemini_regular(digest: str):
    if st.session_state.get("app_view") == "scalper":
        maybe_gemini_scalper_setups()
    return
    now = time.time()
    last = float(st.session_state.get("gemini_regular_ts") or 0)
    if now - last < 60:
        st.session_state["gemini_regular_wait"] = int(60 - (now - last))
        return
    if st.session_state.get("gemini_in_flight"):
        return
    st.session_state["gemini_in_flight"] = True
    st.session_state["gemini_regular_wait"] = 60
    prompt = (
        "You are a Nifty cash/futures/options desk analyst. "
        "Use ONLY the snapshot below (spot, futures VWAP/sigma, EFI, OBV, CVD, "
        "DEX/premium if present, GEX/Δ-GEX/DEX-OI, flip, PECO, PCD$). "
        "Do not invent ticks, OI, or prints that are not in DATA.\n\n"
        "Write these sections, short bullets:\n"
        "1) MARKET STATE — spot vs VWAP/sigma, EFI/CVD/OBV agreement, GEX walls and flip.\n"
        "2) PECO — VALIDATE or INVALIDATE the current PECO action. Why. What would flip it.\n"
        "3) PCD$ — VALIDATE or INVALIDATE. Why. What would flip it.\n"
        "4) PLAYERS — Retail (OTM premium/chase), Dealers (GEX/DEX/gamma flip), "
        "Institutions (CVD/DEX/premium/book if present). Label confidence Low/Med.\n"
        "5) HIGH-PROB ENTRIES — at most two: structure, side, invalidation, skip-if. "
        "If no edge, say NO TRADE.\n"
        "6) EXPECT / WATCH — next 30-90 min: pin vs expansion, levels that matter.\n"
        "End with one line: BIAS | INVALIDATION.\n\nDATA:\n" + digest
    )
    try:
        txt, model = gemini_generate(prompt, GEMINI_REGULAR_MODELS)
        st.session_state["gemini_regular_ts"] = time.time()
        if txt:
            st.session_state["gemini_regular"] = f"[{model}]\n{txt}"
        else:
            st.session_state["gemini_regular"] = f"(no model answered: {model})"
    finally:
        st.session_state["gemini_in_flight"] = False


def gemini_trigger_feedback(event_text: str, digest: str) -> str:
    if not st.session_state.get("gemini_enabled"):
        return ""
    prompt = (
        "Critical TRIGGER on a Nifty desk. Use ONLY DATA. No invented tape.\n"
        "Sections:\n"
        "1) TRIGGER — name it and whether it is confirmed by VWAP/sigma, EFI, CVD, GEX/flip.\n"
        "2) PECO — VALIDATE or INVALIDATE vs this trigger.\n"
        "3) PCD$ — VALIDATE or INVALIDATE vs this trigger.\n"
        "4) FADE vs FOLLOW — which is higher probability and the one invalidation print.\n"
        "5) ENTRY — one plan or WAIT. Include stop idea in points not fantasy targets.\n"
        "6) PLAYERS — retail / dealers / institutions in one line each.\n"
        "End: BIAS | INVALIDATION.\n\n"
        f"TRIGGER:\n{event_text}\n\nDATA:\n{digest}"
    )
    txt, model = gemini_generate(prompt, GEMINI_TRIGGER_MODELS)
    if txt:
        out = f"[{model}]\n{txt}"
        st.session_state["gemini_trigger"] = out
        return out
    st.session_state["gemini_trigger"] = f"(trigger model failed: {model})"
    return ""


def fmt_compact_num(x) -> str:
    try:
        v = float(x or 0)
    except Exception:
        return "—"
    sgn = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12:
        return f"{sgn}{a/1e12:.2f}T"
    if a >= 1e9:
        return f"{sgn}{a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sgn}{a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sgn}{a/1e3:.1f}K"
    return f"{v:.0f}"


def heading_ribbon(title: str, tip_html: str, size: int = 13):
    """Compact hoverable chart-title ribbon."""
    st.markdown(
        f"<div class='micro-hover' style='display:inline-block;padding:3px 10px;"
        f"background:#1A1F2B;border:1px solid #3A4150;border-radius:8px;margin:0 0 4px 0;'>"
        f"<span style='font-weight:700;color:#00E676;font-size:{size}px;'>{title}</span>"
        f"<div class='micro-tip'>{tip_html}</div></div>",
        unsafe_allow_html=True,
    )


def evaluate_directional_trigger(data: dict, df_candles: pd.DataFrame) -> dict:
    """Long/short trigger from GEX location, VWAP, OBV, CVD, EFI."""
    levels = data.get("levels", {}) or {}
    spot = float(data.get("spot_price") or 0)
    gex_sup = float(levels.get("GEX_Support") or spot or 0)
    gex_res = float(levels.get("GEX_Resistance") or spot or 0)
    checks = []
    long_n = short_n = 0

    px = None
    vwap = None
    obv = obv_ma = None
    cvd = None
    efi = None
    lvn = []

    df = pd.DataFrame()
    if df_candles is not None and not df_candles.empty:
        try:
            df, _, _ = pick_last_nse_session(df_candles, min_bars=10)
        except Exception:
            df = df_candles.copy()
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if not df.empty and "close" in df.columns:
        df = df.copy().reset_index(drop=True)
        px = float(df["close"].iloc[-1])
        if "volume" in df.columns:
            tp = (df["high"] + df["low"] + df["close"]) / 3.0 if {"high", "low"}.issubset(df.columns) else df["close"]
            vol = df["volume"].astype(float)
            cum_v = vol.cumsum().replace(0, np.nan)
            vwap = float((tp.astype(float) * vol).cumsum().iloc[-1] / cum_v.iloc[-1]) if cum_v.iloc[-1] == cum_v.iloc[-1] else px
            direction = np.sign(df["close"].astype(float).diff().fillna(0.0))
            obv_s = (direction * vol).cumsum()
            obv = float(obv_s.iloc[-1])
            obv_ma = float(obv_s.rolling(20, min_periods=5).mean().iloc[-1])
            hl = (df["high"] - df["low"]).replace(0, np.nan)
            loc = ((df["close"] - df["low"]) / hl * 2.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
            cvd_s = (vol * loc).cumsum()
            cvd = float(cvd_s.iloc[-1])
            efi = float((df["close"].astype(float).diff() * vol).ewm(span=13, adjust=False).mean().iloc[-1])
            try:
                vp = compute_session_volume_profile(df, bin_step=2.0, prominence_factor=0.35)
                lvn = list(vp.get("lvn") or [])
            except Exception:
                lvn = []
        else:
            vwap = px

    # 1 Location GEX — INDEX SPOT vs GEX walls (never futures)
    loc_long = loc_short = False
    loc_note = "GEX walls unavailable"
    if spot > 0 and gex_sup and gex_res:
        near_lvn = any(abs(spot - lv) <= 15 for lv in lvn)
        near_call = abs(spot - gex_res) <= max(12.0, 0.0006 * spot)
        if spot > gex_res:
            loc_long, loc_short = True, False
            loc_note = f"spot {spot:.0f} above call-wall {gex_res:.0f} (break)"
        elif spot < gex_sup:
            loc_long, loc_short = False, True
            loc_note = f"spot {spot:.0f} below put-wall {gex_sup:.0f}"
        elif near_call:
            loc_long, loc_short = False, True
            loc_note = f"spot {spot:.0f} rejected at call-wall {gex_res:.0f}"
        else:
            loc_long, loc_short = True, False
            loc_note = f"spot {spot:.0f} above put-wall {gex_sup:.0f} · below call {gex_res:.0f}"
        if near_lvn:
            loc_long = True
            loc_note += " · through LVN"
    checks.append({"name": "Location (GEX)", "long": loc_long, "short": loc_short, "note": loc_note})

    # 2 VWAP
    vwap_long = vwap_short = False
    vwap_note = "VWAP unavailable"
    if px and vwap:
        vwap_long = px > vwap
        vwap_short = px < vwap
        vwap_note = f"px {px:.1f} vs VWAP {vwap:.1f}"
    checks.append({"name": "Trend (VWAP)", "long": vwap_long, "short": vwap_short, "note": vwap_note})

    # 3 OBV
    obv_long = obv_short = False
    obv_note = "OBV unavailable"
    if obv is not None and obv_ma is not None and not np.isnan(obv_ma):
        obv_long = obv > obv_ma
        obv_short = obv < obv_ma
        obv_note = f"OBV {obv:.0f} vs MA20 {obv_ma:.0f}"
    checks.append({"name": "Macro Flow (OBV)", "long": obv_long, "short": obv_short, "note": obv_note})

    # 4 CVD structure
    cvd_long = cvd_short = False
    cvd_note = "CVD unavailable"
    if cvd is not None and not df.empty and "volume" in df.columns:
        half = max(len(df) // 2, 3)
        cvd_s = (df["volume"].astype(float) * loc).cumsum() if "volume" in df.columns else None
        if cvd_s is not None and len(cvd_s) >= 6:
            early = float(cvd_s.iloc[:half].max())
            late = float(cvd_s.iloc[half:].max())
            early_lo = float(cvd_s.iloc[:half].min())
            late_lo = float(cvd_s.iloc[half:].min())
            cvd_long = late > early and cvd > 0
            cvd_short = late_lo < early_lo and cvd < 0
            cvd_note = f"CVD {cvd:.0f} · HH={cvd_long} LL={cvd_short}"
    checks.append({"name": "Order Delta (CVD)", "long": cvd_long, "short": cvd_short, "note": cvd_note})

    # 5 EFI execution
    efi_long = efi_short = False
    efi_note = "EFI unavailable"
    if efi is not None and not np.isnan(efi):
        efi_long = efi > 0
        efi_short = efi < 0
        efi_note = f"EFI13 {efi:.1f}"
    checks.append({"name": "Execution (EFI 13)", "long": efi_long, "short": efi_short, "note": efi_note})

    for ch in checks:
        if ch["long"]:
            long_n += 1
        if ch["short"]:
            short_n += 1

    setup_l = sum(1 for ch in checks[:4] if ch["long"])
    setup_s = sum(1 for ch in checks[:4] if ch["short"])
    if setup_l >= 3 and efi_long:
        trigger, colour = "LONG TRIGGER", "#00E676"
        summary = f"Setup {setup_l}/4 long + EFI>0. Directional long."
    elif setup_s >= 3 and efi_short:
        trigger, colour = "SHORT TRIGGER", "#FF5252"
        summary = f"Setup {setup_s}/4 short + EFI<0. Directional short."
    elif setup_l >= 3 and not efi_long:
        trigger, colour = "LONG BIAS (no fire)", "#80CBC4"
        summary = f"Setup {setup_l}/4 long but EFI is not >0 — no entry yet."
    elif setup_s >= 3 and not efi_short:
        trigger, colour = "SHORT BIAS (no fire)", "#EF9A9A"
        summary = f"Setup {setup_s}/4 short but EFI is not <0 — no entry yet."
    elif setup_l > setup_s:
        trigger, colour = "LONG LEAN (no fire)", "#80CBC4"
        summary = f"Only {setup_l}/4 setup long (need 3). EFI already {'green' if efi_long else 'not green'}."
    elif setup_s > setup_l:
        trigger, colour = "SHORT LEAN (no fire)", "#EF9A9A"
        summary = f"Only {setup_s}/4 setup short (need 3). EFI already {'red' if efi_short else 'not red'}."
    else:
        trigger, colour = "NO DIRECTIONAL TRIGGER", "#FF9800"
        summary = f"Setup split {setup_l}L/{setup_s}S. No fire."

    return {
        "trigger": trigger,
        "colour": colour,
        "summary": summary,
        "long_hits": long_n,
        "short_hits": short_n,
        "checks": checks,
    }


def compute_superhuman_scores(data: dict, df_candles: pd.DataFrame) -> dict:
    """
    Superhuman Decision Engine – v2
    Clear labels: QUIET PIN / GAMMA REVERSION / TREND-BREAKOUT / MILD DIRECTIONAL / NO EDGE
    Incorporates DTE, time-of-day, Flip distance, softened flow, clean realised range.
    """
    import datetime
    import numpy as np
    import pytz
    import pandas as pd

    levels = data.get("levels", {}) or {}
    chain  = data.get("chain_results", []) or []
    spot   = float(data.get("spot_price", 0) or 0)

    if spot <= 0 or not chain:
        return {"error": "Insufficient data for scoring"}

    # ------------------------------------------------------------------
    # Time & DTE
    # ------------------------------------------------------------------
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist)
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    is_weekday = now.weekday() < 5
    is_market_hours = is_weekday and (market_open <= now <= market_close)

    dte = 3.0
    try:
        T = float(data.get("T", 0) or 0)
        if T > 0:
            dte = max(T * 365.0, 0.05)
    except Exception:
        pass
    dte = max(min(dte, 30.0), 0.05)

    minutes_since_open = max((now - market_open).total_seconds() / 60.0, 0)
    tod_factor = 1.0
    if is_market_hours:
        if minutes_since_open > 150:
            tod_factor = max(1.0 - (minutes_since_open - 150) / 240.0, 0.25)
    else:
        tod_factor = 0.3

    # ------------------------------------------------------------------
    # A. Gamma Regime
    # ------------------------------------------------------------------
    total_delta_gex_cr = float(data.get("total_net_gex_oi", 0) or 0) / 1e7
    gamma_raw = total_delta_gex_cr / max(abs(total_delta_gex_cr), 2.5) * 70
    gamma_regime_score = float(np.clip(gamma_raw, -100, 100))

    if dte <= 1.0:
        gamma_regime_score = float(np.clip(gamma_regime_score * 1.25, -100, 100))
    elif dte >= 7.0:
        gamma_regime_score = float(np.clip(gamma_regime_score * 0.85, -100, 100))

    # ------------------------------------------------------------------
    # B. Expected vs Realised Move
    # ------------------------------------------------------------------
    straddle = float(levels.get("Straddle_Cost", 0) or 0)
    expected_move_pct = (straddle / spot * 100.0) if spot > 0 else 0.0

    realised_range_pct = 0.0
    day_high = day_low = day_open = None
    df_session = pd.DataFrame()

    if df_candles is not None and not df_candles.empty:
        df_tmp = df_candles.copy()
        df_tmp["time"] = pd.to_datetime(df_tmp["time"])
        latest_date = df_tmp["time"].dt.date.max()
        df_session = df_tmp[df_tmp["time"].dt.date == latest_date].sort_values("time").reset_index(drop=True)
        if not df_session.empty:
            day_high = float(df_session["high"].max())
            day_low  = float(df_session["low"].min())
            day_open = float(df_session["open"].iloc[0])
            if day_open > 0:
                realised_range_pct = (day_high - day_low) / day_open * 100.0

    if dte > 5:
        expected_move_pct *= 0.85

    if is_market_hours:
        if expected_move_pct > 0.12:
            move_score = float(np.clip(
                (expected_move_pct - realised_range_pct * 1.15) / expected_move_pct * 80, -100, 100))
        else:
            move_score = 0.0
        move_context = "Live session (Expected vs Realised-so-far)"
    else:
        if expected_move_pct > 0.10:
            move_score = float(np.clip(
                (expected_move_pct - realised_range_pct) / max(expected_move_pct, 0.15) * 60, -100, 100))
        else:
            move_score = 0.0
        move_context = "Post-market (Priced Expected vs Full-day Realised)"

    # ------------------------------------------------------------------
    # C. Charm / Vanna Flow (soft scaling)
    # ------------------------------------------------------------------
    total_vex = sum(float(r.get("VEX", 0) or 0) for r in chain)
    total_cex = sum(float(r.get("CEX", 0) or 0) for r in chain)
    vanna_raw = total_vex / 1.2e6
    charm_raw = total_cex / 2.5e6
    flow_raw  = vanna_raw + charm_raw
    flow_score = float(np.clip(np.tanh(flow_raw) * 85, -100, 100))

    # ------------------------------------------------------------------
    # D. Opening Range vs GEX Walls
    # ------------------------------------------------------------------
    or_score = 0.0
    or_high = or_low = None
    gex_sup = float(levels.get("GEX_Support") or spot)
    gex_res = float(levels.get("GEX_Resistance") or spot)

    if not df_session.empty and len(df_session) >= 5:
        n_or = min(9, max(5, len(df_session) // 6))
        or_bars = df_session.head(n_or)
        or_high = float(or_bars["high"].max())
        or_low  = float(or_bars["low"].min())
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
    or_score *= tod_factor

    # ------------------------------------------------------------------
    # E. Distance from Zero-Gamma Flip
    # ------------------------------------------------------------------
    flip = float(levels.get("Zero_Gamma_Flip") or spot)
    dist_pct = abs(spot - flip) / spot * 100.0 if spot > 0 else 0.0
    if gamma_regime_score > 20 and dist_pct < 0.35:
        flip_score = 40.0
    elif gamma_regime_score < -20 and dist_pct > 0.6:
        flip_score = -40.0
    else:
        flip_score = float(np.clip((0.4 - dist_pct) * 50, -30, 30))

    # ------------------------------------------------------------------
    # F. Composite
    # ------------------------------------------------------------------
    composite = (
        0.38 * gamma_regime_score +
        0.22 * move_score +
        0.15 * flow_score +
        0.15 * or_score +
        0.10 * flip_score
    )
    composite = float(np.clip(composite, -100, 100))

    # ------------------------------------------------------------------
    # Final Decision – Super Clear Labels
    # ------------------------------------------------------------------
    strong_long_gamma  = gamma_regime_score >= 55
    strong_short_gamma = gamma_regime_score <= -55
    big_range   = realised_range_pct > expected_move_pct * 1.5 if expected_move_pct > 0.1 else realised_range_pct > 1.2
    quiet_range = realised_range_pct < expected_move_pct * 0.85 if expected_move_pct > 0.1 else realised_range_pct < 0.7

    if composite >= 42 and strong_long_gamma and quiet_range:
        bias   = "QUIET PIN"
        action = "Non-directional → Iron Condor / Short Straddle / Iron Fly (high confidence)"
        colour = "#00E676"
        clarity = "Strong long-gamma + quiet range. Classic pinning day."
    elif composite >= 35 and strong_long_gamma and big_range:
        bias   = "GAMMA REVERSION"
        action = "Fade extended moves → fade breakouts, buy dips / sell rallies into close"
        colour = "#26A69A"
        clarity = "Strong long-gamma but price already travelled far. Expect mean-reversion after the trend."
    elif composite <= -42 or (strong_short_gamma and big_range):
        bias   = "TREND / BREAKOUT"
        action = "Directional → Debit spreads / Futures / Naked options"
        colour = "#FF5252"
        clarity = "Short-gamma or clean wall break. Directional follow-through favoured."
    elif abs(composite) <= 15:
        bias   = "NO EDGE"
        action = "Stay out or very tight range strategies only"
        colour = "#FF9800"
        clarity = "No clear dealer positioning or range edge."
    else:
        bias   = "MILD DIRECTIONAL"
        action = "Mild directional → Credit spreads / Calendars with defined risk"
        colour = "#2196F3"
        clarity = "Moderate bias. Prefer defined-risk structures."

    return {
        "gamma_regime_score": round(gamma_regime_score, 1),
        "move_score":         round(move_score, 1),
        "flow_score":         round(flow_score, 1),
        "or_score":           round(or_score, 1),
        "flip_score":         round(flip_score, 1),
        "composite":          round(composite, 1),
        "bias":   bias,
        "action": action,
        "colour": colour,
        "clarity": clarity,
        "is_market_hours":    is_market_hours,
        "move_context":       move_context,
        "expected_move_pct":  round(expected_move_pct, 2),
        "realised_range_pct": round(realised_range_pct, 2),
        "straddle":           round(straddle, 1),
        "total_delta_gex_cr": round(total_delta_gex_cr, 1),
        "or_high":            or_high,
        "or_low":             or_low,
        "gex_support":        gex_sup,
        "gex_resistance":     gex_res,
        "dte":                round(dte, 2),
        "dist_to_flip_pct":   round(dist_pct, 3),
        "tod_factor":         round(tod_factor, 2),
        "day_open":           day_open,
        "day_high":           day_high,
        "day_low":            day_low,
        "big_range":          big_range,
        "quiet_range":        quiet_range,
        "timestamp_ist":      now.strftime("%d-%b-%Y %H:%M:%S IST"),
        "dir_trigger":        evaluate_directional_trigger(data, df_candles),
    }



def compute_technical_indicators(df_candles: pd.DataFrame) -> pd.DataFrame:
    df = df_candles.copy()
    if df.empty or len(df) < 20:
        return df

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["bb_middle"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["bb_std"] = df["close"].rolling(window=20, min_periods=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (2 * df["bb_std"])
    df["bb_bandwidth"] = np.where(df["bb_middle"] > 0, (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"], 0)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    df["date_group"] = df["time"].dt.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_vol"] = df["tp"] * df["volume"]

    df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
    df["cum_tp_vol"] = df.groupby("date_group")["tp_vol"].cumsum()

    raw_vwap = np.where(df["cum_vol"] > 0, df["cum_tp_vol"] / df["cum_vol"], np.nan)
    df["vwap"] = pd.Series(raw_vwap, index=df.index).ffill().fillna(df["close"])

    return df

@st.cache_data(ttl=3600)
def download_master_scrip():
    scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    res = requests.get(scrip_url)
    if res.status_code == 200:
        df_master = pd.DataFrame(res.json())
        df_master['strike_clean'] = pd.to_numeric(df_master['strike'], errors='coerce') / 100.0
        df_master['expiry_dt'] = pd.to_datetime(df_master['expiry'], format='%d%b%Y', errors='coerce')
        return df_master
    return pd.DataFrame()

def get_smartapi_token(df_exp, index_name, target_dt, strike, opt_type):
    try:
        exp_str = target_dt.strftime("%d%b%y").upper()
        exact_symbol = f"{index_name}{exp_str}{int(strike)}{opt_type}"

        sym_match = df_exp[df_exp["symbol"] == exact_symbol]
        if not sym_match.empty:
            return str(sym_match.iloc[0]["token"])

        fallback_match = df_exp[(df_exp["strike_num"] == int(strike)) & (df_exp["symbol"].str.endswith(opt_type))]
        if not fallback_match.empty:
            return str(fallback_match.iloc[0]["token"])
    except Exception:
        pass
    return ""

# --- HOLIDAY / WEEKEND FALLBACK ENGINE FOR CANDLE CHARTS ---

def _tape_gap_window(cached, now_dt, index_name, api_interval):
    """If cache exists, pull from last bar (minus 1 TF) to now. Full session if empty."""
    sh = session_hours(index_name)
    step = 3
    lab = str(api_interval or "")
    if "FIFTEEN" in lab or "15" in lab:
        step = 15
    elif "TEN" in lab:
        step = 10
    elif "FIVE" in lab or lab.endswith("5"):
        step = 5
    elif "THREE" in lab:
        step = 3
    elif "ONE" in lab and "HOUR" not in lab:
        step = 1
    elif "TWO" in lab:
        step = 2
    open_s = f"{now_dt.strftime('%Y-%m-%d')} {sh[4]}"
    if cached is None or getattr(cached, "empty", True) or "time" not in getattr(cached, "columns", []):
        return open_s, now_dt.strftime(f"%Y-%m-%d %H:%M"), True
    last = pd.to_datetime(cached["time"], errors="coerce").max()
    if pd.isna(last):
        return open_s, now_dt.strftime(f"%Y-%m-%d %H:%M"), True
    if last.tzinfo is None:
        last = pytz.timezone("Asia/Kolkata").localize(last)
    last = last.astimezone(pytz.timezone("Asia/Kolkata"))
    # Never ask SmartAPI for a window that crosses midnight — it returns nothing
    if last.date() != now_dt.date():
        return open_s, now_dt.strftime("%Y-%m-%d %H:%M"), True
    start = last - datetime.timedelta(minutes=max(step, 1))
    open_dt = datetime.datetime.strptime(open_s, "%Y-%m-%d %H:%M")
    open_dt = pytz.timezone("Asia/Kolkata").localize(open_dt)
    if start < open_dt:
        start = open_dt
    return start.strftime("%Y-%m-%d %H:%M"), now_dt.strftime("%Y-%m-%d %H:%M"), False


def fetch_candles_with_holiday_fallback(smart_api, spot_token, exchange, api_interval, lookback_days=15, index_name="IDX", tail_minutes=None):
    ist_tz = pytz.timezone("Asia/Kolkata")
    now_dt = datetime.datetime.now(ist_tz)
    live, today, _, _ = market_session_state(now_dt, index_name)
    cached = load_session_cache("spot", index_name, api_interval, today)
    offsets = [0] if live else list(range(0, 16))
    for offset in offsets:
        target_to = now_dt - datetime.timedelta(days=offset)
        if target_to.weekday() >= 5:
            continue
        if live and target_to.date() != today:
            continue
        sh = session_hours(index_name)
        days_back = 0 if live else max(int(lookback_days or 0), 0)
        target_from = target_to if days_back <= 0 else target_to - datetime.timedelta(days=days_back)
        from_clock = sh[4]
        to_clock = sh[5]
        if live:
            fd, td, _full = _tape_gap_window(cached, now_dt, index_name, api_interval)
            candle_param = {
                "exchange": exchange,
                "symboltoken": spot_token,
                "interval": api_interval,
                "fromdate": fd,
                "todate": td,
            }
        else:
            candle_param = {
                "exchange": exchange,
                "symboltoken": spot_token,
                "interval": api_interval,
                "fromdate": target_from.strftime(f"%Y-%m-%d {from_clock}"),
                "todate": target_to.strftime(f"%Y-%m-%d {to_clock}")
            }
        candle_res = safe_api_call(smart_api.getCandleData, candle_param)
        time.sleep(0.20)
        if candle_res and candle_res.get("status") and candle_res.get("data"):
            df_candles = pd.DataFrame(candle_res["data"], columns=["time", "open", "high", "low", "close", "volume"])
            df_candles[["open", "high", "low", "close", "volume"]] = df_candles[["open", "high", "low", "close", "volume"]].astype(float)
            if not df_candles.empty:
                df_candles["time"] = series_to_ist(df_candles["time"])
                if live:
                    df_candles = merge_candle_frames(cached, df_candles)
                    save_session_cache("spot", index_name, api_interval, df_candles, today)
                return compute_technical_indicators(df_candles), (offset > 0 and not live)
    if live and not cached.empty:
        return compute_technical_indicators(cached.copy()), False
    return pd.DataFrame(), False


MCX_NAME_ALIASES = {
    "GOLDM": ["GOLDM", "GOLD"],
    "CRUDEOIL": ["CRUDEOIL", "CRUDE"],
}


def get_near_month_futures_token(df_scrip_master, index_name, fut_exch):
    """Nearest future. GOLDM: symbol GOLDM*FUT, name often GOLD, FUTCOM/MCX."""
    try:
        ist_now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        df = df_scrip_master.copy()
        df["_name"] = df["name"].astype(str).str.upper()
        df["_seg"] = df["exch_seg"].astype(str).str.upper()
        df["_ityp"] = df["instrumenttype"].astype(str).str.upper()
        df["_sym"] = df["symbol"].astype(str).str.upper() if "symbol" in df.columns else ""
        if index_name in MCX_NAME_ALIASES:
            prefixes = tuple(MCX_NAME_ALIASES[index_name])
            fut_scrips = df[
                df["_seg"].isin(["MCX", "NCO"])
                & df["_ityp"].eq("FUTCOM")
                & df["_sym"].str.startswith(prefixes)
                & df["_sym"].str.contains("FUT")
            ].copy()
            # prefer exact product (CRUDEOIL not CRUDEOILM if both exist)
            exact = fut_scrips[fut_scrips["_sym"].str.startswith(index_name)]
            if not exact.empty:
                fut_scrips = exact
        else:
            segs = [str(fut_exch).upper()] if fut_exch else ["NFO"]
            fut_scrips = df[
                df["_seg"].isin(segs)
                & df["_ityp"].isin(["FUTIDX", "FUTSTK", "FUTCOM"])
                & df["_name"].eq(str(index_name).upper())
            ].copy()
        fut_scrips["expiry_dt"] = pd.to_datetime(fut_scrips["expiry"], format="%d%b%Y", errors="coerce")
        active = fut_scrips[fut_scrips["expiry_dt"].dt.date >= ist_now.date()].sort_values("expiry_dt")
        if not active.empty:
            row = active.iloc[0]
            return str(row["token"]), row["expiry"]
    except Exception:
        pass
    return None, None



def series_to_ist(times) -> pd.Series:
    """Coerce SmartAPI candle times to timezone-aware IST."""
    ts = pd.to_datetime(times, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("Asia/Kolkata")
    # Naive: if median hour is outside NSE cash session, treat as UTC
    med = float(ts.dt.hour.median()) if ts.notna().any() else 12
    if med < 7 or med > 18:
        return ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    return ts.dt.tz_localize("Asia/Kolkata")



SESSION_CACHE_DIR = Path("/home/workdir/artifacts/session_cache")
try:
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    SESSION_CACHE_DIR = Path("session_cache")
    SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _price_delta_path(index_name, day):
    return SESSION_CACHE_DIR / f"price_delta_{index_name}_{day}.json"


def load_price_delta_map(index_name, day):
    try:
        import redis
        r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=0.4)
        raw = r.get(f"pdelta:{index_name}:{day}")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    fp = _price_delta_path(index_name, day)
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            return {}
    return {}


def save_price_delta_map(index_name, day, mp):
    try:
        fp = _price_delta_path(index_name, day)
        fp.write_text(json.dumps(mp))
    except Exception:
        pass
    try:
        import redis
        r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=0.4)
        r.set(f"pdelta:{index_name}:{day}", json.dumps(mp))
        # drop previous session keys
        for k in r.scan_iter(f"pdelta:{index_name}:*"):
            if k.decode().split(":")[-1] != str(day):
                r.delete(k)
    except Exception:
        pass


def update_price_delta_map(index_name, ltp, bid, ask, dvol_lots, day):
    """Accumulate signed lots at last price. + hit ask / - hit bid. Not exchange VAP."""
    if not ltp or dvol_lots is None:
        return load_price_delta_map(index_name, day)
    mp = load_price_delta_map(index_name, day)
    px = str(int(round(float(ltp))))
    signed = 0.0
    try:
        if ask and float(ltp) >= float(ask):
            signed = abs(float(dvol_lots))
        elif bid and float(ltp) <= float(bid):
            signed = -abs(float(dvol_lots))
        else:
            signed = float(dvol_lots)
    except Exception:
        signed = float(dvol_lots or 0)
    mp[px] = float(mp.get(px, 0)) + signed
    save_price_delta_map(index_name, day, mp)
    return mp


def _ist_now():
    return datetime.datetime.now(pytz.timezone("Asia/Kolkata"))


def session_hours(index_name=None):
    """Cash/FO 09:15–15:30; MCX metals/energy 09:00–23:30 IST weekdays."""
    name = str(
        index_name
        or st.session_state.get("_last_index")
        or st.session_state.get("Index_Name")
        or "NIFTY"
    ).upper()
    tok = INDEX_TOKEN_MAP.get(name, ("", "", ""))
    exch = str(tok[2] if len(tok) > 2 else "").upper()
    mcx_names = {
        "GOLDM", "GOLD", "SILVERM", "SILVER",
        "CRUDEOIL", "CRUDEOILM", "CRUDE",
        "NATGASMINI", "NATURALGAS",
    }
    if name in mcx_names or exch == "MCX":
        return 9, 0, 23, 30, "09:00", "23:30"
    return 9, 15, 15, 30, "09:15", "15:30"


def session_axis_labels(tf_label=None, index_name=None):
    """Full-session HH:MM slots so bar width stays constant from the open."""
    oh, om, ch, cm, _, _ = session_hours(index_name)
    lab = str(tf_label or st.session_state.get("selected_timeframe") or "3 min")
    if "15" in lab:
        step = 15
    elif "10" in lab:
        step = 10
    elif lab.startswith("2"):
        step = 2
    elif lab.startswith("1"):
        step = 1
    elif "3" in lab:
        step = 3
    else:
        step = 5
    out = []
    t = oh * 60 + om
    end = ch * 60 + cm
    while t <= end:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step
    return out or ["09:15"]


def market_session_state(now=None, index_name=None):
    now = now or _ist_now()
    oh, om, ch, cm, _, _ = session_hours(index_name)
    open_t = now.replace(hour=oh, minute=om, second=0, microsecond=0)
    close_t = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
    live = now.weekday() < 5 and open_t <= now <= close_t
    return live, now.date(), open_t, close_t


def _cache_path(kind: str, index_name: str, interval: str, day):
    safe = f"{kind}_{index_name}_{interval}_{day}.pkl".replace(" ", "")
    return SESSION_CACHE_DIR / safe


def load_session_cache(kind: str, index_name: str, interval: str, day=None):
    live, today, _, _ = market_session_state(index_name=index_name)
    day = day or today
    path = _cache_path(kind, index_name, interval, day)
    if path.exists():
        try:
            df = pd.read_pickle(path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
    key = f"sess_cache_{kind}_{index_name}_{interval}_{day}"
    df = st.session_state.get(key)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return pd.DataFrame()


def save_session_cache(kind: str, index_name: str, interval: str, df: pd.DataFrame, day=None):
    if df is None or getattr(df, "empty", True):
        return
    live, today, _, _ = market_session_state(index_name=index_name)
    day = day or today
    key = f"sess_cache_{kind}_{index_name}_{interval}_{day}"
    st.session_state[key] = df
    try:
        df.to_pickle(_cache_path(kind, index_name, interval, day))
    except Exception:
        pass


def _redis_client():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or ""
    if not url:
        return None
    try:
        import redis as _redis
        return _redis.from_url(url, decode_responses=True, socket_timeout=2)
    except Exception:
        return None


def _prune_redis_old_days(r, prefix: str, keep_day):
    """Keep current session + previous calendar day keys only."""
    try:
        keys = r.keys(f"{prefix}*")
        keep = {str(keep_day), str(keep_day - datetime.timedelta(days=1)),
                str(keep_day - datetime.timedelta(days=2))}
        for k in keys:
            tail = k.split(":")[-1]
            if tail not in keep:
                r.delete(k)
    except Exception:
        pass


def load_flow_tape(index_name: str, day):
    r = _redis_client()
    if r is not None:
        try:
            raw = r.get(f"flow:{index_name}:{day}")
            if raw:
                tape = json.loads(raw)
                st.session_state[f"flow_tape_{index_name}_{day}"] = tape
                return tape
        except Exception:
            pass
    path = SESSION_CACHE_DIR / f"flow_{index_name}_{day}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return list(st.session_state.get(f"flow_tape_{index_name}_{day}") or [])


def save_flow_tape(index_name: str, day, tape):
    st.session_state[f"flow_tape_{index_name}_{day}"] = tape
    st.session_state["flow_tape"] = tape
    try:
        (SESSION_CACHE_DIR / f"flow_{index_name}_{day}.json").write_text(json.dumps(tape))
    except Exception:
        pass
    r = _redis_client()
    if r is not None:
        try:
            r.setex(f"flow:{index_name}:{day}", 48 * 3600, json.dumps(tape))
            _prune_redis_old_days(r, f"flow:{index_name}:", day if hasattr(day, "year") else _ist_now().date())
        except Exception:
            pass



def cvd_price_stats(df: pd.DataFrame, window: int = 20) -> dict:
    """OLS slopes + p-values on this TF only. Divergence only if both slopes significant.

    Regular bearish: price HH and CVD LH (or price slope+ and CVD slope-).
    Regular bullish: price LL and CVD HL (or price slope- and CVD slope+).
    Swings use 3-bar pivots on the same bars — no other TF mixed in.
    """
    out = {
        "ok": False, "n": 0, "win": window,
        "px_slope": None, "px_p": None, "cvd_slope": None, "cvd_p": None,
        "spearman": None, "spearman_p": None,
        "div": "NONE", "div_note": "Need more bars.",
        "sig": False,
    }
    if df is None or df.empty or "close" not in df.columns or "cvd" not in df.columns:
        return out
    px = pd.to_numeric(df["close"], errors="coerce")
    cv = pd.to_numeric(df["cvd"], errors="coerce")
    mask = px.notna() & cv.notna()
    px, cv = px[mask], cv[mask]
    n = int(len(px))
    out["n"] = n
    if n < max(12, window // 2):
        out["div_note"] = f"Only {n} bars — no test."
        return out
    w = min(window, n)
    y_p = px.iloc[-w:].to_numpy(dtype=float)
    y_c = cv.iloc[-w:].to_numpy(dtype=float)
    t = np.arange(w, dtype=float)
    try:
        from scipy.stats import linregress, spearmanr
        rp = linregress(t, y_p)
        rc = linregress(t, y_c)
        rho, rho_p = spearmanr(y_p, y_c)
    except Exception:
        out["div_note"] = "scipy linregress failed."
        return out
    out.update({
        "ok": True,
        "px_slope": float(rp.slope), "px_p": float(rp.pvalue),
        "cvd_slope": float(rc.slope), "cvd_p": float(rc.pvalue),
        "spearman": float(rho) if pd.notna(rho) else None,
        "spearman_p": float(rho_p) if pd.notna(rho_p) else None,
    })
    t_px = float(rp.slope / rp.stderr) if rp.stderr else 0.0
    t_cv = float(rc.slope / rc.stderr) if rc.stderr else 0.0
    px_sig = rp.pvalue < 0.05 and abs(t_px) >= 2.0
    cv_sig = rc.pvalue < 0.05 and abs(t_cv) >= 2.0
    out["sig"] = bool(px_sig and cv_sig)
    out["px_t"] = t_px
    out["cvd_t"] = t_cv

    def _pivots(s, left=3):
        s = s.reset_index(drop=True)
        hi, lo = [], []
        for i in range(left, len(s) - left):
            wdw = s.iloc[i - left:i + left + 1]
            if s.iloc[i] == wdw.max() and (wdw == s.iloc[i]).sum() == 1:
                hi.append(i)
            if s.iloc[i] == wdw.min() and (wdw == s.iloc[i]).sum() == 1:
                lo.append(i)
        return hi, lo

    swing = "NONE"
    hi_p, lo_p = _pivots(px)
    hi_c, lo_c = _pivots(cv)
    if len(hi_p) >= 2 and len(hi_c) >= 2:
        p1, p2 = hi_p[-2], hi_p[-1]
        # match CVD highs nearest those price highs
        def nearest(idxs, i):
            return min(idxs, key=lambda j: abs(j - i))
        c1, c2 = nearest(hi_c, p1), nearest(hi_c, p2)
        if px.iloc[p2] > px.iloc[p1] * 1.0002 and cv.iloc[c2] < cv.iloc[c1]:
            swing = "BEARISH SWING (price HH, CVD LH)"
        elif px.iloc[p2] < px.iloc[p1] * 0.9998 and cv.iloc[c2] > cv.iloc[c1]:
            swing = "BULLISH SWING (price LL, CVD HL)"
    slope_div = "NONE"
    if px_sig and cv_sig:
        if rp.slope > 0 and rc.slope < 0:
            slope_div = "BEARISH SLOPE"
        elif rp.slope < 0 and rc.slope > 0:
            slope_div = "BULLISH SLOPE"
        else:
            slope_div = "CONFIRMED (same sign)"
    elif not px_sig and not cv_sig:
        slope_div = "NO TREND (both slopes p≥0.05)"
    else:
        slope_div = "ONE-SIDED (only one slope p<0.05)"

    last_px = float(px.iloc[-1])
    last_cv = float(cv.iloc[-1])
    px_pct = float((px.iloc[-w:] <= last_px).mean())
    cv_pct = float((cv.iloc[-w:] <= last_cv).mean())
    stretch = False
    if "vwap" in df.columns:
        vw = pd.to_numeric(df["vwap"], errors="coerce").iloc[-w:]
        last_vw = float(vw.iloc[-1]) if vw.notna().any() else last_px
        sig = float((px.iloc[-w:] - vw).std()) or 1.0
        z_vw = (last_px - last_vw) / sig
    else:
        last_vw, z_vw = last_px, 0.0
    rho_ok = (out["spearman_p"] is not None and out["spearman_p"] < 0.10
              and out["spearman"] is not None)
    # High-prob reversion: ALL of opposing |t|>=2, swing, location stretch, CVD extreme, spearman against
    if slope_div.startswith("BEARISH") and "BEARISH" in swing and z_vw >= 0.6 and px_pct >= 0.70 and cv_pct <= 0.35 and rho_ok and out["spearman"] < 0:
        out["div"] = "HIGH-PROB BEARISH DIV"
        out["div_note"] = (f"Fade long: Px t={t_px:+.2f} p={rp.pvalue:.3f}, CVD t={t_cv:+.2f} p={rc.pvalue:.3f}. "
                           f"zVWAP {z_vw:+.2f}, Px pct {px_pct:.0%}, CVD pct {cv_pct:.0%}, ρ={out['spearman']:+.2f}.")
    elif slope_div.startswith("BULLISH") and "BULLISH" in swing and z_vw <= -0.6 and px_pct <= 0.30 and cv_pct >= 0.65 and rho_ok and out["spearman"] < 0:
        out["div"] = "HIGH-PROB BULLISH DIV"
        out["div_note"] = (f"Fade short: Px t={t_px:+.2f} p={rp.pvalue:.3f}, CVD t={t_cv:+.2f} p={rc.pvalue:.3f}. "
                           f"zVWAP {z_vw:+.2f}, Px pct {px_pct:.0%}, CVD pct {cv_pct:.0%}, ρ={out['spearman']:+.2f}.")
    elif slope_div.startswith("BEARISH") and "BEARISH" in swing:
        out["div"] = "WATCH BEARISH"
        out["div_note"] = "Slope+swing only — missing VWAP stretch / CVD extreme / Spearman. No D line."
    elif slope_div.startswith("BULLISH") and "BULLISH" in swing:
        out["div"] = "WATCH BULLISH"
        out["div_note"] = "Slope+swing only — missing VWAP stretch / CVD extreme / Spearman. No D line."
    else:
        out["div"] = slope_div
        out["div_note"] = swing if swing != "NONE" else slope_div
    return out


def scan_cvd_div_events(df: pd.DataFrame, window: int = 20) -> list:
    """Rising-edge divergence marks on THIS TF. No look-ahead: window ends at bar i."""
    ev = []
    if df is None or df.empty or "cvd" not in df.columns:
        return ev
    n = len(df)
    if n < window + 2:
        return ev
    prev = None
    keep = {"HIGH-PROB BEARISH DIV", "HIGH-PROB BULLISH DIV"}
    last_mark = -10**9
    for i in range(window - 1, n):
        sl = df.iloc[i - window + 1:i + 1]
        stt = cvd_price_stats(sl, window=window)
        lab = stt.get("div") or ""
        if lab in keep and lab != prev and (i - last_mark) >= window:
            last_mark = i
            ts = sl["time_str"].iloc[-1] if "time_str" in sl.columns else str(i)
            ev.append({
                "t": ts,
                "div": lab,
                "px_slope": stt.get("px_slope"),
                "px_p": stt.get("px_p"),
                "cvd_slope": stt.get("cvd_slope"),
                "cvd_p": stt.get("cvd_p"),
                "spearman": stt.get("spearman"),
                "spearman_p": stt.get("spearman_p"),
                "note": stt.get("div_note") or "",
            })
        prev = lab if lab in keep else None
    return ev





def detect_fp_absorptions(o, h, l, cl, vol, z, mids, min_bars=20):
    """Causal proxy bid/offer absorption. z is signed bin volume [price x bar]."""
    n = len(cl)
    out = []
    if n < min_bars + 2 or z.size == 0:
        return out
    absz = np.abs(z)
    # typical |bin| from history up to j-1
    for j in range(min_bars, n):
        rng = float(h[j] - l[j])
        if rng <= 0:
            continue
        loc = (float(cl[j]) - float(l[j])) / rng
        prev_l = l[j - min_bars:j]
        prev_h = h[j - min_bars:j]
        mu_l, sd_l = float(np.mean(prev_l)), float(np.std(prev_l, ddof=0) or 1.0)
        mu_h, sd_h = float(np.mean(prev_h)), float(np.std(prev_h, ddof=0) or 1.0)
        z_sweep_dn = (mu_l - float(l[j])) / sd_l
        z_sweep_up = (float(h[j]) - mu_h) / sd_h
        past = absz[:, :j]
        typ = float(np.nanmean(past[past > 0])) if np.any(past > 0) else 1.0
        sig = float(np.nanstd(past[past > 0], ddof=0) or typ or 1.0)
        lo_cut = float(l[j]) + 0.33 * rng
        hi_cut = float(h[j]) - 0.33 * rng
        sell_low = 0.0
        buy_high = 0.0
        for i, m in enumerate(mids):
            if m <= lo_cut and z[i, j] < 0:
                sell_low += -z[i, j]
            if m >= hi_cut and z[i, j] > 0:
                buy_high += z[i, j]
        z_sell = (sell_low - typ) / sig
        z_buy = (buy_high - typ) / sig
        # Setup 1 bid absorption — stricter so mid-range wicks do not print
        if z_sweep_dn >= 1.35 and loc >= 0.68 and z_sell >= 1.15 and cl[j] >= o[j]:
            out.append({
                "j": j, "kind": "BID ABS", "side": 1,
                "stop": float(l[j]) - 2.0,
                "note": f"sweep z={z_sweep_dn:.2f} shelf z={z_sell:.2f} close {loc:.0%} range",
            })
        # Setup 2 offer absorption
        if z_sweep_up >= 1.35 and loc <= 0.32 and z_buy >= 1.15 and cl[j] < o[j]:
            out.append({
                "j": j, "kind": "OFFER ABS", "side": -1,
                "stop": float(h[j]) + 2.0,
                "note": f"sweep z={z_sweep_up:.2f} shelf z={z_buy:.2f} close {loc:.0%} range",
            })
    # one mark per side per 8 bars
    keep, last = [], {-1: -99, 1: -99}
    for e in out:
        s = e["side"]
        if e["j"] - last[s] < 5:
            continue
        keep.append(e)
        last[s] = e["j"]
    return keep


def build_delta_footprint_figure(dfi: pd.DataFrame, axis_times=None, bin_pts=2.0):
    """Bar-range footprint proxy: volume split by close location in the bar. Not exchange bid/ask VAP."""
    if dfi is None or dfi.empty or "volume" not in dfi.columns:
        return None, []
    d = dfi.copy()
    if "time_str" not in d.columns:
        d["time"] = pd.to_datetime(d["time"])
        d["time_str"] = d["time"].dt.strftime("%H:%M")
    xs = list(d["time_str"].astype(str))
    o = d["open"].astype(float).values
    h = d["high"].astype(float).values
    l = d["low"].astype(float).values
    cl = d["close"].astype(float).values
    vol = d["volume"].astype(float).values
    y0, y1 = float(np.nanmin(l)), float(np.nanmax(h))
    span_px = max(y1 - y0, 1.0)
    step = max(float(bin_pts or 2.0), span_px / 48.0)
    edges = np.arange(np.floor(y0 / step) * step, np.ceil(y1 / step) * step + step, step)
    if len(edges) < 3:
        return None, []
    mids = (edges[:-1] + edges[1:]) / 2.0
    z = np.zeros((len(mids), len(xs)))
    bar_dlt = []
    for j in range(len(xs)):
        v = float(vol[j] or 0)
        signed = v if cl[j] >= o[j] else -v
        bar_dlt.append(signed)
        lo, hi = float(l[j]), float(h[j])
        if hi <= lo:
            hi = lo + step
        span = hi - lo
        buy_v = v * max(0.0, (cl[j] - lo) / span)
        sell_v = v * max(0.0, (hi - cl[j]) / span)
        for i, m in enumerate(mids):
            a, b = edges[i], edges[i + 1]
            ov = max(0.0, min(hi, b) - max(lo, a))
            if ov <= 0:
                continue
            frac = ov / span
            # lower half of bar -> sell, upper half -> buy (close location weights)
            mid_bar = (lo + hi) / 2.0
            if m >= mid_bar:
                z[i, j] += buy_v * frac
            else:
                z[i, j] -= sell_v * frac
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.03)
    fig.add_trace(plt_go.Heatmap(
        x=xs, y=mids, z=z, colorscale=[
            [0.00, "#FF1744"],
            [0.35, "#7F1D1D"],
            [0.48, "#16181D"],
            [0.52, "#16181D"],
            [0.65, "#14532D"],
            [1.00, "#00E676"],
        ],
        zmid=0, colorbar=dict(thickness=10, len=0.6, y=0.72),
        opacity=0.95,
        hovertemplate="%{x} · %{y:.0f}<br>Δvol %{z:.0f}<extra>footprint</extra>",
    ), row=1, col=1)
    fig.add_trace(plt_go.Candlestick(
        x=xs, open=o, high=h, low=l, close=cl, name="Idx",
        increasing_line_color="#B2FF59", decreasing_line_color="#FF8A80",
        increasing_fillcolor="rgba(0,0,0,0)", decreasing_fillcolor="rgba(0,0,0,0)",
        line=dict(width=1.6),
        showlegend=False,
    ), row=1, col=1)
    # delta labels on sparse bars only
    step_l = max(1, len(xs) // 24)
    tx, ty, tt, tc = [], [], [], []
    peak = y1 + (y1 - y0) * 0.04
    for j in range(0, len(xs), step_l):
        dv = bar_dlt[j]
        if abs(dv) < 1:
            continue
        lab = f"Δ{dv/1000:+.1f}K" if abs(dv) >= 1000 else f"Δ{dv:+.0f}"
        tx.append(xs[j]); ty.append(peak); tt.append(lab)
        tc.append("#00E676" if dv >= 0 else "#FF5252")
    if tx:
        fig.add_trace(plt_go.Scatter(
            x=tx, y=ty, mode="text", text=tt,
            textfont=dict(size=9, color="#B0BEC5"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    evs = detect_fp_absorptions(o, h, l, cl, vol, z, mids)
    st.session_state["_fp_abs_evs"] = evs
    if evs:
        bx = [xs[e["j"]] for e in evs if e["side"] > 0]
        by = [h[e["j"]] for e in evs if e["side"] > 0]
        bt = [e["kind"] for e in evs if e["side"] > 0]
        sx = [xs[e["j"]] for e in evs if e["side"] < 0]
        sy = [l[e["j"]] for e in evs if e["side"] < 0]
        stt = [e["kind"] for e in evs if e["side"] < 0]
        if bx:
            fig.add_trace(plt_go.Scatter(
                x=bx, y=by, mode="markers+text", name="Bid abs",
                text=bt, textposition="top center",
                marker=dict(size=11, symbol="triangle-up", color="#00E676",
                            line=dict(width=1, color="#FFF")),
                hovertext=[e["note"] + f"  SL {e['stop']:.1f}" for e in evs if e["side"] > 0],
                hoverinfo="text",
            ), row=1, col=1)
        if sx:
            fig.add_trace(plt_go.Scatter(
                x=sx, y=sy, mode="markers+text", name="Offer abs",
                text=stt, textposition="bottom center",
                marker=dict(size=11, symbol="triangle-down", color="#FF5252",
                            line=dict(width=1, color="#FFF")),
                hovertext=[e["note"] + f"  SL {e['stop']:.1f}" for e in evs if e["side"] < 0],
                hoverinfo="text",
            ), row=1, col=1)
    fig.add_trace(plt_go.Bar(
        x=xs, y=bar_dlt,
        marker_color=["#00E676" if v >= 0 else "#FF5252" for v in bar_dlt],
        showlegend=False, name="Bar Δ",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#FFF", row=2, col=1)
    # Use this tape's bars only — session categoryarray leaves a blank left pad on MCX
    times = xs
    xr = [-0.5, max(len(times) - 0.5, 0.5)]
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=640, margin=dict(l=40, r=8, t=8, b=18),
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=times,
                     range=xr, showticklabels=False, row=1, col=1)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=times,
                     range=xr, nticks=8, row=2, col=1)
    fig.update_yaxes(title_text="Px", row=1, col=1)
    fig.update_yaxes(title_text="Δ", row=2, col=1)
    return fig, evs


def _option_session_figure(df_opt, label, index_name="NIFTY", tf_label="5 min", axis_times=None):
    if df_opt is None or df_opt.empty or len(df_opt) < 3:
        return None, "NO DATA"
    raw = df_opt.copy()
    raw["time"] = pd.to_datetime(raw["time"])
    try:
        sess, day, _ = pick_last_nse_session(
            raw, min_bars=3, prefer_today=True, index_name=index_name,
        )
        d = sess if sess is not None and not sess.empty else raw
    except Exception:
        d = raw
        d["session_date"] = pd.to_datetime(d["time"]).dt.date
        last = sorted(d["session_date"].dropna().unique())[-1]
        d = d[d["session_date"] == last].copy()
    px = pd.to_numeric(d["close"], errors="coerce")
    med = float(px.median() or 0) or 1.0
    if med > 0:
        clipped = d[(px > med * 0.15) & (px < med * 6.0)].copy()
        if clipped is not None and len(clipped) >= 3:
            d = clipped
    if d.empty or len(d) < 3:
        return None, "NO DATA"
    d = attach_bar_flow(d.reset_index(drop=True), rebuild=True)
    d["time"] = pd.to_datetime(d["time"])
    d = d.dropna(subset=["time"]).sort_values("time")
    d = d.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    ts = d["time"].dt.strftime("%H:%M")
    if ts.duplicated().any():
        ts = d["time"].dt.strftime("%H:%M:%S")
    d["time_str"] = ts.astype(str)
    d["tp"] = (d["high"] + d["low"] + d["close"]) / 3.0
    vol_s = pd.to_numeric(d.get("volume", 1.0), errors="coerce").fillna(1.0)
    d["volume"] = vol_s
    cv = vol_s.cumsum().replace(0, np.nan)
    d["vwap"] = (d["tp"] * vol_s).cumsum() / cv
    d["vwap"] = d["vwap"].ffill().fillna(d["close"])
    k_sig = float(st.session_state.get("vwap_sigma_select") or 1.5)
    dev = d["close"].astype(float) - d["vwap"]
    win = 20 if len(d) >= 24 else max(5, min(12, len(d)))
    d["vwap_std"] = dev.rolling(win, min_periods=max(3, win // 3)).std()
    cap = float(d["close"].astype(float).std() or 1.0) * 3.0
    d["vwap_std"] = d["vwap_std"].clip(upper=max(cap, 1.0)).fillna(0.0)
    d["vwap_u"] = d["vwap"] + k_sig * d["vwap_std"]
    d["vwap_l"] = d["vwap"] - k_sig * d["vwap_std"]
    own = d["time_str"].astype(str).tolist()
    full_axis = list(axis_times) if axis_times else session_axis_labels(tf_label, index_name)
    if len(own) < 8 or len(own) < max(12, int(len(full_axis) * 0.35)):
        axis_times = own
    else:
        axis_times = full_axis
    xr = [-0.5, max(len(axis_times) - 0.5, 0.5)]
    y0 = float(min(d["low"].min(), d["vwap"].min()))
    y1 = float(max(d["high"].max(), d["vwap"].max()))
    pad = (y1 - y0) * (0.14 if st.session_state.get("pdec_labels_on") else 0.06) if y1 > y0 else 2.0
    y0, y1 = y0 - pad, y1 + pad
    vp = compute_session_volume_profile(d, bin_step=max(0.5, (y1 - y0) / 40.0), prominence_factor=0.35)
    dp = compute_delta_profile(d, bin_step=max(0.5, (y1 - y0) / 40.0))
    fig = make_subplots(
        rows=4, cols=3,
        column_widths=[0.13, 0.71, 0.16],
        row_heights=[0.46, 0.16, 0.18, 0.20],
        shared_xaxes=False,
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
        specs=[[{}, {}, {}], [None, {}, None], [None, {}, None], [None, {}, None]],
    )
    if dp.get("ok"):
        fig.add_trace(plt_go.Bar(
            x=list(dp["delta"]), y=list(dp["mids"]), orientation="h", showlegend=False, name="ΔP",
            marker=dict(color=["#00E676" if v >= 0 else "#FF5252" for v in dp["delta"]]),
            hovertemplate="Px %{y:.1f}<br>Δ %{x:.0f}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(plt_go.Candlestick(
        x=d["time_str"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name=label, increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
        increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350", showlegend=True,
    ), row=1, col=2)
    if d["vwap_u"].notna().any():
        fig.add_trace(plt_go.Scatter(
            x=d["time_str"], y=d["vwap_u"], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
        ), row=1, col=2)
        fig.add_trace(plt_go.Scatter(
            x=d["time_str"], y=d["vwap_l"], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,152,0,0.08)",
        ), row=1, col=2)
    fig.add_trace(plt_go.Scatter(x=d["time_str"], y=d["vwap"], name="VWAP",
                                line=dict(color="#FF9800", width=1.6)), row=1, col=2)
    day_hi, day_lo = float(d["high"].max()), float(d["low"].min())
    opt_lvls = [
        {"price": day_hi, "name": "Day H", "color": "#FF8A80", "width": 1.1, "dash": "dot"},
        {"price": day_lo, "name": "Day L", "color": "#69F0AE", "width": 1.1, "dash": "dot"},
    ]
    if vp.get("ok"):
        opt_lvls.extend(vp_chart_levels(vp))
    last_y = None
    for lv in opt_lvls:
        if lv.get("price") is None:
            continue
        yv = float(lv["price"])
        fig.add_hline(
            y=yv, line_dash=lv.get("dash", "dot"), line_color=lv["color"],
            line_width=lv.get("width", 1.1), row=1, col=2,
        )
        y_txt = yv
        if last_y is not None and abs(y_txt - last_y) < 2.0:
            y_txt = last_y + 2.0
        last_y = y_txt
        fig.add_annotation(
            x=d["time_str"].iloc[-1], y=y_txt,
            text=f"{lv['name']} {yv:.0f}",
            showarrow=False, xanchor="right",
            font=dict(size=8, color=lv["color"]), row=1, col=2,
        )
    if st.session_state.get("pdec_labels_on"):
        try:
            recs = pdec_session_history(d)
            prev = None
            peak = float(d["high"].max())
            for rec in recs:
                act = rec.get("action") or ""
                if not act or act == prev:
                    continue
                prev = act
                short = act if len(act) <= 22 else act[:20] + "…"
                fig.add_annotation(
                    x=rec.get("t") or "", y=peak,
                    text=short, showarrow=False, textangle=-90,
                    xanchor="center", yanchor="bottom",
                    font=dict(size=8, color="#CFD8DC"),
                    row=1, col=2,
                )
        except Exception:
            pass
    if vp.get("ok"):
        cols = ["#FFD54F" if abs(m - vp["poc"]) < 1e-9 else "rgba(100,181,246,0.75)" for m in vp["mids"]]
        fig.add_trace(plt_go.Bar(
            x=list(vp["vol"]), y=list(vp["mids"]), orientation="h", showlegend=False, name="VP",
            marker=dict(color=cols),
            hovertemplate="Px %{y:.1f}<br>Vol %{x:.0f}<extra></extra>",
        ), row=1, col=3)
    vol = d["volume"].astype(float)
    up = d["close"] >= d["open"]
    fig.add_trace(plt_go.Bar(x=d["time_str"], y=np.where(up, vol, 0), showlegend=False,
                             marker_color="rgba(0,230,118,0.7)"), row=2, col=2)
    fig.add_trace(plt_go.Bar(x=d["time_str"], y=np.where(~up, -vol, 0), showlegend=False,
                             marker_color="rgba(255,82,82,0.7)"), row=2, col=2)
    efi_c = np.where(d["efi13"] >= 0, "#00E676", "#FF5252")
    fig.add_trace(plt_go.Bar(x=d["time_str"], y=d["efi13"], marker_color=efi_c, showlegend=False), row=3, col=2)
    fig.add_trace(plt_go.Scatter(x=d["time_str"], y=d["cvd"], line=dict(color="#B0BEC5", width=1.2),
                                showlegend=False), row=4, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#FFF", row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#FFF", row=3, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#FFF", row=4, col=2)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                      height=560, margin=dict(l=40, r=6, t=8, b=18),
                      hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(range=[y0, y1], tickformat=".1f", row=1, col=1)
    fig.update_yaxes(range=[y0, y1], title_text="LTP", row=1, col=2)
    fig.update_yaxes(range=[y0, y1], showticklabels=False, row=1, col=3)
    fig.update_xaxes(type="linear", showgrid=False, row=1, col=1)
    fig.update_xaxes(type="linear", showticklabels=False, showgrid=False, row=1, col=3)
    for r in (1, 2, 3):
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                         range=xr, showticklabels=False, row=r, col=2)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                     range=xr, nticks=8, row=4, col=2)
    try:
        vm = float(max(abs(float(vol.max())), abs(float(vol.min())), 1.0))
        fig.update_yaxes(range=[-vm * 1.2, vm * 1.2], title_text="Vol", row=2, col=2)
        em = float(max(abs(float(d["efi13"].max())), abs(float(d["efi13"].min())), 1.0))
        fig.update_yaxes(range=[-em * 1.2, em * 1.2], title_text="EFI", row=3, col=2)
        cm = float(max(abs(float(d["cvd"].max())), abs(float(d["cvd"].min())), 1.0))
        fig.update_yaxes(range=[min(0.0, float(d["cvd"].min()) * 1.1), max(0.0, float(d["cvd"].max()) * 1.1)],
                         title_text="CVD", row=4, col=2)
    except Exception:
        fig.update_yaxes(title_text="Vol", row=2, col=2)
        fig.update_yaxes(title_text="EFI", row=3, col=2)
        fig.update_yaxes(title_text="CVD", row=4, col=2)
    act = "NO DATA"
    try:
        recs = pdec_session_history(d)
        if recs:
            act = recs[-1].get("action") or "—"
    except Exception:
        pass
    return fig, act



def attach_bar_flow(df: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """CVD/OBV on THESE bars. Call on native TF only. Resample must use last(CVD), not rebuild."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if not rebuild and "cvd" in out.columns and out["cvd"].notna().any():
        if "obv_ma20" not in out.columns and "obv" in out.columns:
            pass
        return out
    cl = out["close"].astype(float)
    vol = out["volume"].astype(float) if "volume" in out.columns else pd.Series(0.0, index=out.index)
    hi = out["high"].astype(float) if "high" in out.columns else cl
    lo = out["low"].astype(float) if "low" in out.columns else cl
    hl = (hi - lo).replace(0, np.nan)
    loc = ((cl - lo) / hl * 2.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
    out["signed_flow"] = vol * loc
    out["cvd"] = out["signed_flow"].cumsum()
    direction = np.sign(cl.diff().fillna(0.0))
    out["obv"] = (direction * vol).cumsum()
    out["efi13"] = (cl.diff() * vol).ewm(span=13, adjust=False).mean()
    return out


def merge_candle_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    parts = [x for x in (old, new) if x is not None and not getattr(x, "empty", True)]
    if not parts:
        return pd.DataFrame()
    if len(parts) == 2 and "close" in parts[0].columns and "close" in parts[1].columns:
        a = pd.to_numeric(parts[0]["close"], errors="coerce").median()
        b = pd.to_numeric(parts[1]["close"], errors="coerce").median()
        if a and b and a > 0 and b > 0 and max(a, b) / min(a, b) > 1.25:
            return parts[1].copy()
    out = pd.concat(parts, ignore_index=True)
    out["time"] = series_to_ist(out["time"])
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return out


def pick_last_nse_session(df: pd.DataFrame, min_bars: int = 20, prefer_today: bool = True,
                          force_date=None, index_name=None) -> tuple:
    """Return (session_df, session_date, used_prior_session).
    During live hours prefer TODAY even with 1–2 bars (Mon 09:16 problem).
    min_bars only applies when falling back to a completed prior session.
    force_date: pin to that calendar date when present in the frame.
    """
    empty = (pd.DataFrame(), None, True)
    if df is None or df.empty or "time" not in df.columns:
        return empty
    out = df.copy()
    out["time"] = series_to_ist(out["time"])
    out = out.dropna(subset=["time"]).sort_values("time")
    oh, om, ch, cm, _, _ = session_hours(index_name)
    mins = out["time"].dt.hour * 60 + out["time"].dt.minute
    out = out[(mins >= oh * 60 + om) & (mins <= ch * 60 + cm)]
    if out.empty:
        return empty
    out["session_date"] = out["time"].dt.date
    counts = out.groupby("session_date").size().sort_index()
    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.datetime.now(ist).date()
    chosen = None
    used_prior = True
    live_now, _, _, _ = market_session_state(index_name=index_name)
    if force_date is not None:
        try:
            fd = force_date if hasattr(force_date, "year") else pd.to_datetime(force_date).date()
        except Exception:
            fd = None
        if fd is not None and fd in counts.index:
            sess = out[out["session_date"] == fd].copy().reset_index(drop=True)
            sess["time_str"] = sess["time"].dt.strftime("%H:%M")
            return sess, fd, fd != today
        if fd is not None:
            return pd.DataFrame(), fd, True
    if prefer_today and today in counts.index and today.weekday() < 5 and int(counts.loc[today]) >= 1:
        chosen = today
        used_prior = False
    elif live_now and prefer_today:
        # Market open: do not walk back to Friday even if today is thin/missing
        chosen = today if today in counts.index else None
        used_prior = chosen is None
        if chosen is None:
            # Live but today's bars not in frame yet — keep prior session visible
            weekdays = [(d, n) for d, n in counts.items() if d.weekday() < 5]
            if weekdays:
                chosen = max(weekdays, key=lambda x: x[0])[0]
                used_prior = True
            else:
                return empty
    else:
        for d, n in list(counts.items())[::-1]:
            if d.weekday() >= 5:
                continue
            if d == today:
                continue
            if int(n) >= min_bars:
                chosen = d
                break
        if chosen is None:
            weekdays = [(d, n) for d, n in counts.items() if d.weekday() < 5]
            if weekdays:
                chosen = max(weekdays, key=lambda x: x[1])[0]
            else:
                chosen = counts.index[-1]
            used_prior = chosen != today
    sess = out[out["session_date"] == chosen].copy().reset_index(drop=True)
    sess["time_str"] = sess["time"].dt.strftime("%H:%M")
    return sess, chosen, used_prior


def fetch_one_session_ohlcv(smart_api, token, exchange, api_interval, day, index_name="IDX"):
    """Single NSE/FO session of candles for replay. Returns raw OHLCV or empty."""
    empty = pd.DataFrame()
    if not smart_api or not token or day is None:
        return empty
    try:
        day = day if hasattr(day, "year") else pd.to_datetime(day).date()
    except Exception:
        return empty
    _, _, _, _, from_hm, to_hm = session_hours(index_name)
    param = {
        "exchange": exchange,
        "symboltoken": str(token),
        "interval": api_interval,
        "fromdate": f"{day.strftime('%Y-%m-%d')} {from_hm}",
        "todate": f"{day.strftime('%Y-%m-%d')} {to_hm}",
    }
    res = safe_api_call(smart_api.getCandleData, param)
    time.sleep(0.20)
    if not (res and res.get("status") and res.get("data")):
        return empty
    df = pd.DataFrame(res["data"], columns=["time", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["time"] = series_to_ist(df["time"])
    df["session_date"] = df["time"].dt.date
    df = df[df["session_date"] == day].copy()
    if df.empty:
        return empty
    df["time_str"] = df["time"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)


def fetch_india_vix_sessions(smart_api, lookback_days=10):
    """Last 3 NSE sessions of India VIX (5-min) + last-session % change."""
    empty = (pd.DataFrame(), None, None)
    if not smart_api:
        return empty
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist)
    candle_param = {
        "exchange": "NSE",
        "symboltoken": INDIA_VIX_TOKEN,
        "interval": "FIVE_MINUTE",
        "fromdate": (now - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d 09:15"),
        "todate": now.strftime("%Y-%m-%d 15:30"),
    }
    res = safe_api_call(smart_api.getCandleData, candle_param)
    if not (res and res.get("status") and res.get("data")):
        return empty
    df = pd.DataFrame(res["data"], columns=["time", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["time"] = series_to_ist(df["time"])
    df = df.dropna(subset=["time"])
    mins = df["time"].dt.hour * 60 + df["time"].dt.minute
    df = df[(mins >= 9 * 60 + 15) & (mins <= 15 * 60 + 30)]
    df["session"] = df["time"].dt.date
    dates = [d for d in sorted(df["session"].unique()) if d.weekday() < 5][-3:]
    if not dates:
        return empty
    df = df[df["session"].isin(dates)].sort_values("time")
    last_d = dates[-1]
    sess = df[df["session"] == last_d]
    open_px = float(sess["open"].iloc[0])
    last_px = float(sess["close"].iloc[-1])
    pct = ((last_px - open_px) / open_px * 100.0) if open_px else 0.0
    df["time_str"] = df["time"].dt.strftime("%d-%b %H:%M")
    return df, pct, last_d


def fetch_futures_candles_with_vwap(smart_api, index_name, df_scrip_master, api_interval, lookback_days=15, tail_minutes=None):
    """
    Fetch near-month futures candles + real VWAP.
    Outside market hours → falls back to the most recent trading session and flags it.
    Returns: (df, is_fallback, basis_info, fallback_msg)
    """
    spot_token, spot_exch, fut_exch = INDEX_TOKEN_MAP.get(index_name, ("99926000", "NSE", "NFO"))
    fut_token, fut_expiry = get_near_month_futures_token(df_scrip_master, index_name, fut_exch)

    if not fut_token:
        return pd.DataFrame(), False, {}, "No active futures contract found"

    ist_tz = pytz.timezone("Asia/Kolkata")
    now_dt = datetime.datetime.now(ist_tz)

    live_sess, _, market_open, market_close = market_session_state(now_dt, index_name)
    is_weekday = now_dt.weekday() < 5
    currently_closed = not live_sess

    cached = load_session_cache("fut", index_name, api_interval, now_dt.date())
    best_df = pd.DataFrame()
    best_offset = 0
    offsets = [0] if not currently_closed else list(range(0, 16))

    for offset in offsets:
        target_to = now_dt - datetime.timedelta(days=offset)
        if target_to.weekday() >= 5:
            continue
        target_from = target_to if int(lookback_days or 0) <= 0 else target_to - datetime.timedelta(days=lookback_days)
        sh = session_hours(index_name)
        from_clock, to_clock = sh[4], sh[5]
        if not currently_closed:
            fd, td, _full = _tape_gap_window(cached, now_dt, index_name, api_interval)
            candle_param = {
                "exchange": fut_exch,
                "symboltoken": str(fut_token),
                "interval": api_interval,
                "fromdate": fd,
                "todate": td,
            }
        else:
            candle_param = {
                "exchange": fut_exch,
                "symboltoken": str(fut_token),
                "interval": api_interval,
                "fromdate": target_from.strftime(f"%Y-%m-%d {from_clock}"),
                "todate": target_to.strftime(f"%Y-%m-%d {to_clock}")
            }
        candle_res = safe_api_call(smart_api.getCandleData, candle_param)
        time.sleep(0.20)
        if not (candle_res and candle_res.get("status") and candle_res.get("data")):
            continue
        df = pd.DataFrame(candle_res["data"], columns=["time", "open", "high", "low", "close", "volume"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        if df.empty:
            continue
        df["time"] = series_to_ist(df["time"])
        if not currently_closed:
            df = merge_candle_frames(cached, df)
        min_need = 1 if not currently_closed else 5
        if len(df) < min_need:
            continue
        df = compute_technical_indicators(df)
        best_df = df
        best_offset = offset
        break

    if best_df.empty and not cached.empty and not currently_closed:
        best_df = compute_technical_indicators(cached.copy())
        best_offset = 0
    if best_df.empty:
        return pd.DataFrame(), False, {}, "Could not fetch futures candles"

    fut_ltp = float(best_df["close"].iloc[-1])

    spot_ltp = None
    try:
        spot_resp = safe_api_call(smart_api.ltpData, exchange=spot_exch,
                                  tradingsymbol=index_name, symboltoken=spot_token)
        time.sleep(0.25)
        if spot_resp and spot_resp.get("status") and spot_resp.get("data"):
            spot_ltp = float(spot_resp["data"]["ltp"])
    except Exception:
        pass

    basis_info = {
        "fut_token": fut_token,
        "fut_expiry": str(fut_expiry) if fut_expiry is not None else "N/A",
        "spot_ltp": spot_ltp,
        "fut_ltp": fut_ltp,
        "basis": round(fut_ltp - spot_ltp, 2) if (spot_ltp and fut_ltp) else None
    }

    sess_df, sess_date, used_prior = pick_last_nse_session(
        best_df, min_bars=20, prefer_today=True, index_name=index_name,
    )
    if sess_df.empty:
        last_bar_time = pd.to_datetime(best_df["time"].iloc[-1])
        session_date_str = last_bar_time.strftime("%d-%b-%Y")
        used_prior = True
    else:
        # Keep today's live tape even if only a handful of bars (do not require 5)
        if sess_date == now_dt.date() or len(sess_df) >= 5:
            best_df = sess_df
        session_date_str = sess_date.strftime("%d-%b-%Y") if sess_date else pd.to_datetime(best_df["time"].iloc[-1]).strftime("%d-%b-%Y")

    today_sess = sess_date == now_dt.date() if sess_date else False
    live_now = is_weekday and (market_open <= now_dt <= market_close)

    if today_sess or (live_now and not best_df.empty):
        save_session_cache("fut", index_name, api_interval, best_df, now_dt.date())

    if today_sess and live_now:
        fallback_msg = f"Live session • {session_date_str} IST"
        is_fallback = False
    elif live_now and used_prior:
        fallback_msg = (
            f"Today's candles not in yet — showing last session {session_date_str} IST"
        )
        is_fallback = True
    elif currently_closed:
        fallback_msg = (
            f"Market closed. Last session: {session_date_str} IST (fallback)"
        )
        is_fallback = True
    else:
        fallback_msg = f"Session {session_date_str} IST"
        is_fallback = used_prior

    return best_df, is_fallback, basis_info, fallback_msg


# --- METHOD 1: FUTURES VOLUME & PROXY Z-SCORE COMPUTATION ---
def fetch_futures_zscores_method1(smart_api, symbol, days, df_scrip_master, progress_container=None):
    try:
        if progress_container is not None:
            progress_container.caption("⏳ Fetching continuous historical data for Z-score window...")

        ist_now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        spot_token, spot_exch, fut_exch = INDEX_TOKEN_MAP.get(symbol, ("99926000", "NSE", "NFO"))

        window_sz = int(days)
        from_date = (ist_now - datetime.timedelta(days=window_sz * 3 + 30)).strftime("%Y-%m-%d 09:15")
        to_date = ist_now.strftime("%Y-%m-%d 15:30")

        spot_param = {
            "exchange": spot_exch,
            "symboltoken": spot_token,
            "interval": "ONE_DAY",
            "fromdate": from_date,
            "todate": to_date
        }
        s_res = safe_api_call(smart_api.getCandleData, spot_param)
        time.sleep(0.40)
        if not (s_res and s_res.get('status') and s_res.get('data')):
            return pd.DataFrame()

        df_spot = pd.DataFrame(s_res['data'], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df_spot['date'] = df_spot['time'].str.split('T').str[0]
        df_spot['spot_close'] = df_spot['close'].astype(float)
        df_spot = df_spot.set_index('date')

        df_spot['log_ret'] = np.log(df_spot['spot_close'] / df_spot['spot_close'].shift(1))
        df_spot['HV_Lookback'] = df_spot['log_ret'].rolling(window=window_sz).std() * np.sqrt(252)

        fut_scrips = df_scrip_master[
            (df_scrip_master['exch_seg'] == fut_exch) & 
            (df_scrip_master['name'] == symbol) & 
            (df_scrip_master['instrumenttype'].isin(['FUTIDX', 'FUTSTK']))
        ].copy()

        fut_scrips['expiry_dt'] = pd.to_datetime(fut_scrips['expiry'], format='%d%b%Y', errors='coerce')
        active_futs = fut_scrips[fut_scrips['expiry_dt'].dt.date >= ist_now.date()].sort_values('expiry_dt')
        
        if not active_futs.empty:
            near_fut_token = str(active_futs.iloc[0]['token'])
            fut_param = {
                "exchange": fut_exch,
                "symboltoken": near_fut_token,
                "interval": "ONE_DAY",
                "fromdate": from_date,
                "todate": to_date
            }
            f_res = safe_api_call(smart_api.getCandleData, fut_param)
            time.sleep(0.40)
            if f_res and f_res.get('status') and f_res.get('data'):
                df_fut = pd.DataFrame(f_res['data'], columns=['time', 'open', 'high', 'low', 'close', 'fut_volume'])
                df_fut['date'] = df_fut['time'].str.split('T').str[0]
                df_fut['fut_volume'] = df_fut['fut_volume'].astype(float)
                df_fut = df_fut.set_index('date')
                df_spot = df_spot.join(df_fut['fut_volume'], how='left').fillna(0.0)
            else:
                df_spot['fut_volume'] = 0.0
        else:
            df_spot['fut_volume'] = 0.0

        min_p = max(5, window_sz // 2)
        
        df_spot['Vol_Mean'] = df_spot['fut_volume'].rolling(window=window_sz, min_periods=min_p).mean()
        df_spot['Vol_Std'] = df_spot['fut_volume'].rolling(window=window_sz, min_periods=min_p).std()
        df_spot['Futures_Volume_Z'] = (df_spot['fut_volume'] - df_spot['Vol_Mean']) / df_spot['Vol_Std'].replace(0, np.nan)

        df_spot['HV_Mean'] = df_spot['HV_Lookback'].rolling(window=window_sz, min_periods=min_p).mean()
        df_spot['HV_Std'] = df_spot['HV_Lookback'].rolling(window=window_sz, min_periods=min_p).std()
        df_spot['Volatility_Proxy_Z'] = (df_spot['HV_Lookback'] - df_spot['HV_Mean']) / df_spot['HV_Std'].replace(0, np.nan)

        df_res = pd.DataFrame({
            'Futures_Close': df_spot['spot_close'],
            'Futures_Volume': df_spot['fut_volume'],
            'Vol_Mean': df_spot['Vol_Mean'],
            'Vol_Std': df_spot['Vol_Std'],
            'Futures_Volume_Z': df_spot['Futures_Volume_Z'],
            'Volatility_Proxy': df_spot['HV_Lookback'],
            'HV_Mean': df_spot['HV_Mean'],
            'HV_Std': df_spot['HV_Std'],
            'Volatility_Proxy_Z': df_spot['Volatility_Proxy_Z']
        })

        if progress_container is not None:
            progress_container.empty()

        return df_res.dropna(subset=['Futures_Volume_Z', 'Volatility_Proxy_Z'])

    except Exception:
        if progress_container is not None:
            progress_container.empty()
        return pd.DataFrame()

# --- SIDEBAR SETUP ---
st.sidebar.markdown("### ⚙️ Parameters & Strategy Builder")

df_master = download_master_scrip()

_view_labels = {
    "default": "1. Default mode",
    "multi": "2. Multi Index mode",
    "scalper": "3. Scalper mode",
}
st.session_state["app_view"] = st.sidebar.radio(
    "View mode",
    ["default", "multi", "scalper"],
    format_func=lambda x: _view_labels.get(x, x),
    key="app_view_radio",
)
st.session_state["multi_index_mode"] = st.session_state["app_view"] == "multi"

with st.sidebar.expander("4. Market Parameters", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        Index_Name = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "GOLDM", "CRUDEOIL"])
    if st.session_state.get("_last_index") and st.session_state["_last_index"] != Index_Name:
        st.session_state["flow_tape"] = []
        st.session_state.pop("liq_delta_history", None)
    st.session_state["_last_index"] = Index_Name
    
    default_token, spot_exchange, Exchange = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))
    if not default_token:
        tok, _ = get_near_month_futures_token(df_master, Index_Name, Exchange)
        default_token = tok or ""
    
    with c2:
        st.text_input("Exchange", value=Exchange, disabled=True)

    rate_param = st.number_input("Risk Free Rate (r)", min_value=0.0, max_value=0.15, value=0.07, step=0.01)

    if Index_Name in MCX_NAME_ALIASES:
        aliases = [a.upper() for a in MCX_NAME_ALIASES[Index_Name]]
        df_options = df_master[
            (df_master["exch_seg"].isin(["MCX", "NCO"]))
            & (df_master["instrumenttype"].isin(["OPTFUT", "OPTCOM"]))
            & (df_master["name"].astype(str).str.upper().isin(aliases + [Index_Name]))
        ].copy()
        if "symbol" in df_options.columns:
            pref = df_options[df_options["symbol"].astype(str).str.upper().str.startswith(Index_Name)]
            df_options = pref if not pref.empty else df_options
    else:
        df_options = df_master[
            (df_master["exch_seg"] == Exchange)
            & (df_master["name"] == Index_Name)
            & (df_master["instrumenttype"].isin(["OPTIDX", "OPTSTK", "OPTFUT"]))
        ].copy()

    df_options["expiry_dt"] = pd.to_datetime(df_options["expiry"], format="%d%b%Y", errors='coerce')
    ist_tz = pytz.timezone("Asia/Kolkata")
    today_dt = pd.to_datetime(datetime.datetime.now(ist_tz).date())

    valid_expiries = sorted(df_options[df_options["expiry_dt"] >= today_dt]["expiry_dt"].unique())
    expiry_options_str = [pd.to_datetime(exp).strftime("%d%b%Y").upper() for exp in valid_expiries]

    selected_expiry_str = st.selectbox("Expiry Date", expiry_options_str if expiry_options_str else ["N/A"])

    c3, c4 = st.columns(2)
    with c3:
        strikes_below = st.number_input("Below ATM", min_value=1, max_value=50, value=10, step=1)
    with c4:
        strikes_above = st.number_input("Above ATM", min_value=1, max_value=50, value=10, step=1)

    hv_days = st.slider("HV Lookback (Days)", min_value=10, max_value=90, value=30, step=5)
    heatmap_timeframe = st.selectbox(
        "GEX Heatmap Timeframe",
        ["3 min", "5 min", "10 min"],
        index=["3 min", "5 min", "10 min"].index(st.session_state["heatmap_timeframe"])
        if st.session_state.get("heatmap_timeframe") in ["3 min", "5 min", "10 min"] else 1
    )
    st.session_state["heatmap_timeframe"] = heatmap_timeframe

target_expiry_dt = pd.to_datetime(selected_expiry_str, format="%d%b%Y") if selected_expiry_str != "N/A" else today_dt
df_expiry = df_options[df_options["expiry_dt"] == target_expiry_dt].copy()

df_expiry["strike_num"] = pd.to_numeric(df_expiry["strike"], errors="coerce") / (100.0 if Exchange == "NFO" else 1.0)
if df_expiry["strike_num"].max() > 1000000:
    df_expiry["strike_num"] = df_expiry["strike_num"] / 100.0

all_expiry_strikes = sorted([int(s) for s in df_expiry["strike_num"].dropna().unique()])

with st.sidebar.expander("5. Build Strategy Basket", expanded=False):
    selected_strike = st.selectbox("Option Strike Price", all_expiry_strikes if all_expiry_strikes else [24500], format_func=lambda x: f"{int(x)}")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        opt_type = st.selectbox("Option Type", ["CE", "PE"])
    with b_col2:
        trade_action = st.selectbox("Trade Action", ["BUY", "SELL"])

    entry_price_input = st.number_input("Entry Price (₹) [0 for LTP]", min_value=0.0, value=0.0, step=0.5)
    default_qty = LOT_SIZES.get(Index_Name, 65)
    qty_lots = st.number_input("Quantity / Units", min_value=1, value=default_qty, step=1)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("+ Add Leg", use_container_width=True):
            st.session_state["basket_legs"].append({
                "strike": int(selected_strike),
                "type": opt_type,
                "action": trade_action,
                "entry_price": entry_price_input,
                "qty": qty_lots
            })
            sync_local_storage()
            st.success("Leg Added!")
            st.rerun()
    with c_btn2:
        if st.button("Clear Basket", use_container_width=True):
            st.session_state["basket_legs"] = []
            sync_local_storage()
            st.rerun()

    # Cache Control Panel
    st.markdown("---")
    st.caption("💾 **Cache Strategy Storage**")
    cache_c1, cache_c2 = st.columns(2)
    with cache_c1:
        if st.button("Save to Cache", use_container_width=True):
            save_basket_to_cache(st.session_state["basket_legs"])
            st.success("Basket Cached!")
    with cache_c2:
        if st.button("Load Cache", use_container_width=True):
            cached_legs = get_cached_basket()
            if cached_legs:
                st.session_state["basket_legs"] = cached_legs
                sync_local_storage()
                st.success("Basket Restored!")
                st.rerun()
            else:
                st.info("Cache is empty.")

if st.session_state.get("app_view") == "multi":
    with st.sidebar.expander("Multi Index picks", expanded=True):
        st.caption("Enable only the indices you watch. Hidden/disabled skip API calls.")
        en = dict(st.session_state.get("multi_enabled") or {})
        for _idx in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "GOLDM", "CRUDEOIL"]:
            en[_idx] = st.checkbox(_idx, value=bool(en.get(_idx, False)), key=f"multi_en_{_idx}")
        st.session_state["multi_enabled"] = en
        n_on = sum(1 for v in en.values() if v)
        st.caption(f"{n_on} live · tapes 5s · Net GEX 5 min")
elif st.session_state.get("app_view") == "scalper":
    st.sidebar.caption("Scalper uses the index + expiry from Market Parameters. ATM PE | Spot | ATM CE.")
    with st.sidebar.expander("6. Gemini analysis", expanded=True):
        st.checkbox(
            "Gemini setups",
            key="gemini_enabled",
            help="Reads Spot + ATM CE/PE tapes and writes high-probability option setups. Needs GEMINI_API_KEY in secrets.",
        )
        st.session_state["gemini_interval_min"] = st.selectbox(
            "Gemini interval",
            options=[3, 5, 10, 15],
            index=1,
            format_func=lambda m: f"every {m} min",
            key="gemini_interval_sel",
        )
        nxt = int(st.session_state.get("gemini_regular_wait") or 0)
        st.caption(f"Next call in {nxt}s")
        if st.button("Generate setups now", use_container_width=True, key="gemini_force_btn"):
            st.session_state["gemini_regular_ts"] = 0
            maybe_gemini_scalper_setups()
        if not _gemini_key():
            st.warning("Add GEMINI_API_KEY to Streamlit secrets.")
else:
    st.session_state["gemini_enabled"] = False

run_btn = st.sidebar.button("🚀 Fetch Chain & Greeks", use_container_width=True)

interval_mapping = {
    "1 min": ("ONE_MINUTE", 0),
    "2 min": ("ONE_MINUTE", 0),
    "3 min": ("THREE_MINUTE", 10),
    "5 min": ("FIVE_MINUTE", 15),
    "10 min": ("TEN_MINUTE", 20),
    "15 min": ("FIFTEEN_MINUTE", 30),
}

# --- SECURE SESSION HANDLER (do not cache failed logins) ---
def get_smart_api_client():
    cached = st.session_state.get("_smart_api_obj")
    cached_ts = float(st.session_state.get("_smart_api_ts") or 0)
    if cached is not None and (time.time() - cached_ts) < 3000:
        return cached

    missing = [n for n, v in [
        ("API_KEY", API_KEY),
        ("CLIENT_CODE", CLIENT_CODE),
        ("PIN", PIN),
        ("TOTP_SECRET", TOTP_SECRET),
    ] if not v]
    if missing:
        st.session_state["_smart_api_err"] = (
            "Secrets not visible to the app: " + ", ".join(missing)
            + ". In Streamlit Cloud use flat TOML keys with these exact names."
        )
        return None

    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp_token = pyotp.TOTP(TOTP_SECRET).now()
        session = safe_api_call(smart_api.generateSession, CLIENT_CODE, PIN, totp_token)

        if session and session.get("status"):
            st.session_state["_smart_api_obj"] = smart_api
            st.session_state["_smart_api_ts"] = time.time()
            st.session_state["_smart_api_err"] = ""
            return smart_api
        st.session_state["_smart_api_err"] = (
            "generateSession failed: "
            + str((session or {}).get("message") if isinstance(session, dict) else session or "empty response")
        )
    except Exception as e:
        st.session_state["_smart_api_err"] = f"SmartAPI exception: {e}"
    return None

main_top_progress_holder = st.container()

# --- DATA FETCHING ENGINE ---
def fetch_live_data(selected_interval_label="5 min", progress_container=None):
    p_bar = None
    p_status = None
    if progress_container is not None:
        p_bar = progress_container.progress(0.0)
        p_status = progress_container.empty()

    def update_p(pct, msg):
        if p_bar is not None:
            p_bar.progress(min(1.0, max(0.0, pct)))
        if p_status is not None:
            p_status.caption(f"⏳ {msg}")
        try:
            update_load_status(msg)
        except Exception:
            pass

    update_p(0.10, "Authenticating SmartAPI Session...")

    smart_api = get_smart_api_client()
    if not smart_api:
        if p_bar: p_bar.empty()
        if p_status: p_status.empty()
        detail = st.session_state.get("_smart_api_err") or "unknown"
        st.error(f"Missing credentials or failed to generate SmartAPI session! {detail}")
        return None

    try:
        update_p(0.20, f"Fetching Live Spot & Volatility for {Index_Name}...")
        spot_token, spot_exch, opt_exch = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))
        fut_tok, fut_exp = get_near_month_futures_token(df_master, Index_Name, opt_exch)
        if (not spot_token) and fut_tok:
            spot_token = fut_tok
        ltp_sym = Index_Name
        try:
            if fut_tok is not None and df_master is not None:
                hit = df_master[df_master["token"].astype(str) == str(fut_tok)]
                if not hit.empty and "symbol" in hit.columns:
                    ltp_sym = str(hit.iloc[0]["symbol"])
        except Exception:
            pass
        spot_resp = None
        if spot_token:
            spot_resp = safe_api_call(smart_api.ltpData, exchange=spot_exch, tradingsymbol=ltp_sym, symboltoken=spot_token)
        time.sleep(0.40)
        if spot_resp and spot_resp.get("status") and spot_resp.get("data"):
            spot_price = float(spot_resp["data"]["ltp"])
        elif Index_Name == "SENSEX":
            spot_price = 80000.0
        else:
            spot_price = 0.0

        hv_key = f"hv_{Index_Name}_{hv_days}"
        hv_ts = st.session_state.get("_hv_ts") or 0
        now_s = datetime.datetime.now().timestamp()
        if st.session_state.get(hv_key) and (now_s - hv_ts) < 900:
            index_hv = float(st.session_state[hv_key])
        else:
            index_hv = VolatilityEngine.calculate_hv(smart_api, spot_token, spot_exch, days=hv_days)
            st.session_state[hv_key] = float(index_hv or 0)
            st.session_state["_hv_ts"] = now_s

        update_p(0.35, "Fetching Historical Underlying Candles (with Holiday Fallback)...")
        api_interval, lookback_days = interval_mapping.get(selected_interval_label, ("FIVE_MINUTE", 15))

        df_candles, is_holiday_fallback = fetch_candles_with_holiday_fallback(
            smart_api, spot_token, spot_exch, api_interval, lookback_days, Index_Name
        )

        # ----- Near-month Futures candles + real VWAP -----
        update_p(0.42, "Fetching Near-Month Futures candles for VWAP...")
        df_futures, fut_is_fallback, basis_info, fut_fallback_msg = fetch_futures_candles_with_vwap(
            smart_api, Index_Name, df_master, api_interval, lookback_days
        )
        if (not spot_price) or float(spot_price) <= 0:
            for src in (df_futures, df_candles):
                if src is not None and not getattr(src, "empty", True) and "close" in src.columns:
                    spot_price = float(src["close"].iloc[-1])
                    break

        now_dt = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        expiry_datetime = target_expiry_dt.replace(hour=15, minute=30, second=0).tz_localize("Asia/Kolkata")
        time_diff_seconds = (expiry_datetime - now_dt).total_seconds()
        T = max(time_diff_seconds / (365.0 * 24 * 3600), 1e-5)

        atm_strike = min(all_expiry_strikes, key=lambda x: abs(x - spot_price)) if all_expiry_strikes else spot_price
        atm_idx = all_expiry_strikes.index(atm_strike) if all_expiry_strikes else 0
        filtered_strikes = all_expiry_strikes[max(0, atm_idx - strikes_below): min(len(all_expiry_strikes), atm_idx + strikes_above + 1)] if all_expiry_strikes else []

        tokens_to_fetch = set()
        strike_mapping = []

        update_p(0.50, "Mapping Option Chain Tokens...")
        for strike in filtered_strikes:
            strike_int = int(strike)
            c_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, strike_int, "CE")
            p_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, strike_int, "PE")

            if c_tok: tokens_to_fetch.add(c_tok)
            if p_tok: tokens_to_fetch.add(p_tok)

            strike_mapping.append({
                "strike": strike_int,
                "call_tok": c_tok,
                "put_tok": p_tok
            })

        basket_tokens_info = {}
        for leg in st.session_state.get("basket_legs", []):
            k = int(leg["strike"])
            t = leg["type"]
            tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, k, t)
            if tok:
                tokens_to_fetch.add(tok)
                basket_tokens_info[f"{k}_{t}"] = tok

        tokens_to_fetch_list = [t for t in tokens_to_fetch if t and t != "nan"]

        update_p(0.65, "Fetching Live Market Depth & Prices...")
        market_data = {}
        chunk_size = 40
        for i in range(0, len(tokens_to_fetch_list), chunk_size):
            chunk = tokens_to_fetch_list[i:i + chunk_size]
            res = safe_api_call(smart_api.getMarketData, "FULL", {opt_exch: chunk})
            if res and res.get("status") and res.get("data") and res["data"].get("fetched"):
                for item in res["data"]["fetched"]:
                    vol_val = (
                        item.get("tradeVolume")
                        or item.get("volume")
                        or item.get("v")
                        or item.get("vol")
                        or item.get("volumeTraded")
                        or 0
                    )

                    market_data[str(item["symbolToken"])] = {
                        "ltp": float(item.get("ltp", 0.0)),
                        "oi": int(item.get("opnInterest", 0)),
                        "volume": int(vol_val),
                        "best_bid": float(item.get("bestBidPrice", item.get("ltp", 0.0))),
                        "best_ask": float(item.get("bestAskPrice", item.get("ltp", 0.0)))
                    }
            time.sleep(0.40)

        update_p(0.85, "Calculating Option Greeks & Gamma Exposure Profile...")
        atm_c_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, int(atm_strike), "CE")
        atm_p_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, int(atm_strike), "PE")

        atm_c_ltp = market_data.get(atm_c_tok, {}).get("ltp", 0.0)
        atm_p_ltp = market_data.get(atm_p_tok, {}).get("ltp", 0.0)

        F = atm_strike + math.exp(rate_param * T) * (atm_c_ltp - atm_p_ltp) if (atm_c_ltp > 0 and atm_p_ltp > 0) else spot_price

        lot_size = LOT_SIZES.get(Index_Name, 65)
        chain_results = []
        total_call_oi = total_put_oi = 0
        total_net_gex_oi = 0.0
        total_net_gex_vol = 0.0
        all_ivs = []

        for row in strike_mapping:
            K = row["strike"]
            c_info = market_data.get(row["call_tok"], {"ltp": 0.0, "oi": 0, "volume": 0, "best_bid": 0.0, "best_ask": 0.0})
            p_info = market_data.get(row["put_tok"], {"ltp": 0.0, "oi": 0, "volume": 0, "best_bid": 0.0, "best_ask": 0.0})

            # Solve CE and PE IV independently. Do NOT substitute HV into deep ITM
            # options — that invents gamma on almost-intrinsic contracts and paints
            # low strikes (ITM calls) as large positive / green GEX.
            c_iv = VolatilityEngine.calculate_iv(c_info["ltp"], F, K, T, rate_param, "c")
            p_iv = VolatilityEngine.calculate_iv(p_info["ltp"], F, K, T, rate_param, "p")

            strike_step = 50.0
            if len(all_expiry_strikes) >= 2:
                strike_step = float(abs(all_expiry_strikes[1] - all_expiry_strikes[0]))
            near_atm = abs(K - F) <= (2.0 * strike_step)

            c_sigma = c_iv if c_iv > 0 else (index_hv if near_atm and c_info["ltp"] > 0 else 0.0)
            p_sigma = p_iv if p_iv > 0 else (index_hv if near_atm and p_info["ltp"] > 0 else 0.0)

            c_greeks = VolatilityEngine.calculate_greeks(F, K, T, rate_param, c_sigma, "c")
            p_greeks = VolatilityEngine.calculate_greeks(F, K, T, rate_param, p_sigma, "p")

            gex_scale = lot_size * (spot_price ** 2) * 0.01
            call_gex_oi = c_greeks["gamma"] * c_info["oi"] * gex_scale
            put_gex_oi = p_greeks["gamma"] * p_info["oi"] * gex_scale
            net_gex_oi = call_gex_oi - put_gex_oi
            total_net_gex_oi += net_gex_oi

            # Delta-adjusted GEX: weight each wing by its own delta, keep put sign
            # negative. After the ITM-IV fix this matches published GEX maps
            # (red / short-gamma below spot, green / long-gamma above spot).
            call_delta_gex_oi = call_gex_oi * max(c_greeks["delta"], 0.0)
            put_delta_gex_oi = put_gex_oi * abs(p_greeks["delta"])
            net_delta_gex_oi = call_delta_gex_oi - put_delta_gex_oi

            call_gex_vol = c_greeks["gamma"] * c_info["volume"] * lot_size * (spot_price ** 2) * 0.01
            put_gex_vol = p_greeks["gamma"] * p_info["volume"] * lot_size * (spot_price ** 2) * 0.01
            net_gex_vol = call_gex_vol - put_gex_vol
            total_net_gex_vol += net_gex_vol

            vex_val = (c_greeks["vega"] * c_info["oi"] - p_greeks["vega"] * p_info["oi"]) * lot_size * 0.01
            cex_val = (c_greeks["charm"] * c_info["oi"] - p_greeks["charm"] * p_info["oi"]) * lot_size * spot_price * 0.01

            if c_iv > 0:
                all_ivs.append(c_iv)
            if p_iv > 0:
                all_ivs.append(p_iv)
            total_call_oi += c_info["oi"]
            total_put_oi += p_info["oi"]

            c_iv_disp = c_iv if c_iv > 0 else c_sigma
            p_iv_disp = p_iv if p_iv > 0 else p_sigma

            chain_results.append({
                "C_Vol": c_info["volume"], "C_OI": c_info["oi"],
                "C_Δ": round(c_greeks["delta"], 2), "C_γ": round(c_greeks["gamma"], 4),
                "C_θ": round(c_greeks["theta"], 2), "C_ν": round(c_greeks["vega"], 2),
                "C_IV_val": c_iv_disp, "C_IV": f"{c_iv_disp * 100:.1f}%", "C_LTP": c_info["ltp"],
                "C_Bid": c_info.get("best_bid", 0.0), "C_Ask": c_info.get("best_ask", 0.0),
                "Strike": K,
                "Net_GEX_OI": round(net_gex_oi, 2),
                "Net_Delta_GEX_OI": round(net_delta_gex_oi, 2),
                "Net_GEX_Vol": round(net_gex_vol, 2),
                "VEX": round(vex_val, 2),
                "CEX": round(cex_val, 2),
                "P_LTP": p_info["ltp"], "P_IV_val": p_iv_disp, "P_IV": f"{p_iv_disp * 100:.1f}%",
                "P_Bid": p_info.get("best_bid", 0.0), "P_Ask": p_info.get("best_ask", 0.0),
                "P_Δ": round(p_greeks["delta"], 2), "P_γ": round(p_greeks["gamma"], 4),
                "P_θ": round(p_greeks["theta"], 2), "P_ν": round(p_greeks["vega"], 2),
                "P_OI": p_info["oi"], "P_Vol": p_info["volume"]
            })

        update_p(0.95, "Finalizing Level Calculations & Dashboard View...")
        pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0
        iv_percentile = 0.0
        if all_ivs:
            min_iv, max_iv = min(all_ivs), max(all_ivs)
            atm_c_iv = VolatilityEngine.calculate_iv(atm_c_ltp, F, atm_strike, T, rate_param, "c")
            if max_iv > min_iv and atm_c_iv > 0:
                iv_percentile = ((atm_c_iv - min_iv) / (max_iv - min_iv)) * 100.0

        max_pain_strike = VolatilityEngine.calculate_max_pain(chain_results)
        levels = calculate_support_resistance_targets(chain_results, spot_price, max_pain_strike)

        try:
            live_now, today, _, _ = market_session_state(now_dt)
            sess_day = today
            if df_futures is not None and not df_futures.empty and "time" in df_futures.columns:
                ft = series_to_ist(df_futures["time"])
                sess_day = pd.to_datetime(ft.iloc[-1]).date()
            lot = LOT_SIZES.get(Index_Name, 65)
            dex = prem_c = prem_p = 0.0
            for r in chain_results:
                dex += float(r.get("C_Δ") or 0) * float(r.get("C_Vol") or 0) * lot
                dex -= abs(float(r.get("P_Δ") or 0)) * float(r.get("P_Vol") or 0) * lot
                prem_c += float(r.get("C_LTP") or 0) * float(r.get("C_Vol") or 0) * lot
                prem_p += float(r.get("P_LTP") or 0) * float(r.get("P_Vol") or 0) * lot
            tape = load_flow_tape(Index_Name, sess_day)
            if live_now and sess_day == today:
                tape.append({
                    "ts": now_dt.strftime("%H:%M"),
                    "dex": dex, "prem_c": prem_c, "prem_p": prem_p,
                    "day": str(sess_day),
                })
                tape = tape[-300:]
                save_flow_tape(Index_Name, sess_day, tape)
            else:
                st.session_state["flow_tape"] = tape
                st.session_state["flow_tape_day"] = str(sess_day)
        except Exception:
            pass

        update_p(1.0, "Done!")
        if p_bar: p_bar.empty()
        if p_status: p_status.empty()

        return {
            "spot_price": spot_price, "F": F, "T": T, "index_hv": index_hv, "iv_percentile": iv_percentile,
            "pcr": pcr, "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "total_net_gex_oi": total_net_gex_oi, "total_net_gex_vol": total_net_gex_vol,
            "max_pain_strike": max_pain_strike, "levels": levels, "df_candles": df_candles,
            "df_futures": df_futures,
            "basis_info": basis_info,
            "fut_is_fallback": fut_is_fallback,
            "fut_fallback_msg": fut_fallback_msg,
            "chain_results": chain_results, "market_data": market_data, "basket_tokens_info": basket_tokens_info,
            "is_holiday_fallback": is_holiday_fallback,
            "timestamp": now_dt.strftime("%d-%b-%Y %H:%M:%S IST"),
            "bar_tf": selected_interval_label,
            "fut_bars": 0 if df_futures is None else int(len(df_futures)),
            "atm_strike": atm_strike,
            "atm_ce_token": atm_c_tok,
            "atm_pe_token": atm_p_tok,
            "opt_exchange": opt_exch,
        }

    except Exception:
        if p_bar: p_bar.empty()
        if p_status: p_status.empty()
        st.warning("API Rate limit / sync notice: Retrying on next cycle...")
        return None

# Full-chain fetch only in Default, and only on Fetch click.
# Missing data_store is filled inside live_dashboard_fragment so we do not
# double-call SmartAPI on every script run (that trips AB1004 / rate limits).
if run_btn and st.session_state.get("app_view", "default") == "default":
    new_data = fetch_live_data(st.session_state["selected_timeframe"], progress_container=main_top_progress_holder)
    if new_data:
        new_data["selected_expiry"] = selected_expiry_str
        st.session_state["data_store"] = new_data

# --- Z-SCORE ANALYSIS FRAGMENT ---
@st.fragment(run_every=300 if st.session_state.get("enable_zscore_refresh", False) else None)
def zscore_analysis_fragment(mode="full"):
    """mode: 'highlights' = compact table only | 'raw' = inspect expander only | 'full' = both"""
    def _style_z(val):
        if val >= 1.5 or val <= -1.5:
            return 'background-color: #ff4d4d; color: white; font-weight: bold;'
        elif 0.5 <= val < 1.5 or -1.5 < val <= -0.5:
            return 'background-color: #ffea80; color: black;'
        else:
            return 'background-color: #b3ffb3; color: black;'

    if mode in ("highlights", "full"):
        heading_ribbon(f"📊 Z-Scores ({Index_Name})",
            "z = (x − μ) / σ over the lookback window. Used for unusual OI/volume and index z.")
        st.caption("HV + Futures Volume Z")
        enable_z_ref = st.checkbox("Auto-Refresh 5 min", value=st.session_state["enable_zscore_refresh"], key="cb_zscore_refresh")
        if enable_z_ref != st.session_state["enable_zscore_refresh"]:
            st.session_state["enable_zscore_refresh"] = enable_z_ref
            st.rerun()
        calc_z_btn = st.button("🔄 Compute / Refresh Z-Scores", use_container_width=True, key="btn_zscore_compute")
        progress_holder = st.container()
        smart_api = get_smart_api_client()
        if calc_z_btn or st.session_state["zscore_data_store"].empty:
            if smart_api:
                z_df = fetch_futures_zscores_method1(smart_api, Index_Name, hv_days, df_master, progress_container=progress_holder)
                st.session_state["zscore_data_store"] = z_df
            else:
                st.error("SmartAPI Session uninitialized.")
        z_df = st.session_state["zscore_data_store"]
        if not z_df.empty:
            st.caption("Recent Highlights (Last 5 Days)")
            last_5 = z_df.tail(5).copy()
            summary_cols = ["Futures_Volume_Z", "Volatility_Proxy_Z"]
            styled_z_df = last_5[summary_cols].sort_index(ascending=False).style.map(
                _style_z, subset=summary_cols
            ).format({col: "{:.2f}" for col in summary_cols})
            st.dataframe(styled_z_df, use_container_width=True)
        else:
            st.info("Click Compute to load Z-Scores.")

    if mode in ("raw", "full"):
        z_df = st.session_state.get("zscore_data_store", pd.DataFrame())
        if not z_df.empty:
            with st.expander("🔍 Inspect Raw Calculated Daily Totals & HV Details (All Evaluated Dates)", expanded=False):
                st.info(f"Showing all **{len(z_df)}** trading days used in evaluating rolling averages and std dev.")
                display_cols = [
                    "Futures_Close", "Futures_Volume", "Vol_Mean", "Vol_Std", "Futures_Volume_Z",
                    "Volatility_Proxy", "HV_Mean", "HV_Std", "Volatility_Proxy_Z",
                ]
                st.dataframe(
                    z_df[display_cols].sort_index(ascending=False).style.format({
                        "Futures_Close": "{:,.2f}",
                        "Futures_Volume": "{:,.0f}",
                        "Vol_Mean": "{:,.0f}",
                        "Vol_Std": "{:,.0f}",
                        "Futures_Volume_Z": "{:.2f}",
                        "Volatility_Proxy": "{:.4f}",
                        "HV_Mean": "{:.4f}",
                        "HV_Std": "{:.4f}",
                        "Volatility_Proxy_Z": "{:.2f}",
                    }),
                    use_container_width=True,
                )
        elif mode == "raw":
            st.info("Compute Z-Scores first (from the panel above) to inspect raw daily totals.")

# --- INSTITUTIONAL ORDER FLOW SCANNER MODULE ---
def institutional_order_flow_scanner_fragment():
    st.markdown("---")
    hdr_l, hdr_r = st.columns([0.72, 0.28])
    with hdr_l:
        heading_ribbon(
            f"🏛️ Institutional Order Flow ({Index_Name})",
            "Flags strikes where premium (LTP×OI×lot) ≥ ₹10L or volume ≥ 500 lots.<br>"
            "Unusual vol = session volume well above that strike’s typical print.<br>"
            "Use as flow tape, not a standalone entry.",
        )
        st.caption("High-conviction flow · Premium ≥ ₹10L or Vol ≥ 500 lots")
    with hdr_r:
        show_all_flow = st.checkbox("Show all rows", value=False, key="iof_show_all",
                                    help="Off = only High Vol flagged rows")

    if "data_store" not in st.session_state or not st.session_state["data_store"].get("chain_results"):
        st.info("No option chain data available. Run the main fetch process to enable scanning.")
        return

    chain_data = st.session_state["data_store"]["chain_results"]
    spot_price = st.session_state["data_store"].get("spot_price", 0.0)
    lot_size = LOT_SIZES.get(Index_Name, 65)

    scanned_trades = []
    all_volumes = [r["C_Vol"] for r in chain_data] + [r["P_Vol"] for r in chain_data]
    avg_daily_vol = np.mean(all_volumes) if all_volumes and np.mean(all_volumes) > 0 else 1.0

    for row in chain_data:
        strike = row["Strike"]

        c_ltp = row["C_LTP"]
        c_vol = row["C_Vol"]
        c_premium = c_ltp * c_vol * lot_size
        c_lots = c_vol
        if c_premium >= 1000000 or c_lots >= 500:
            vol_ratio = c_vol / avg_daily_vol if avg_daily_vol > 0 else 0.0
            unusual_flag = vol_ratio > 3.0
            strike_role = "Resistance Level" if strike >= spot_price else "Support / In-The-Money Level"
            scanned_trades.append({
                "Strike": strike,
                "Option_Type": "CE",
                "LTP (₹)": c_ltp,
                "Volume (Lots)": c_lots,
                "Total Premium (₹)": round(c_premium, 2),
                "Key Level Classification": strike_role,
                "Volume Ratio": round(vol_ratio, 2),
                "Unusual Vol Flag": "🚨 High Vol" if unusual_flag else "Normal"
            })

        p_ltp = row["P_LTP"]
        p_vol = row["P_Vol"]
        p_premium = p_ltp * p_vol * lot_size
        p_lots = p_vol
        if p_premium >= 1000000 or p_lots >= 500:
            vol_ratio = p_vol / avg_daily_vol if avg_daily_vol > 0 else 0.0
            unusual_flag = vol_ratio > 3.0
            strike_role = "Support Level" if strike <= spot_price else "Resistance / In-The-Money Level"
            scanned_trades.append({
                "Strike": strike,
                "Option_Type": "PE",
                "LTP (₹)": p_ltp,
                "Volume (Lots)": p_lots,
                "Total Premium (₹)": round(p_premium, 2),
                "Key Level Classification": strike_role,
                "Volume Ratio": round(vol_ratio, 2),
                "Unusual Vol Flag": "🚨 High Vol" if unusual_flag else "Normal"
            })

    if scanned_trades:
        df_scanner = pd.DataFrame(scanned_trades)
        if not show_all_flow:
            df_scanner = df_scanner[df_scanner["Unusual Vol Flag"].astype(str).str.contains("High Vol", na=False)]
            if df_scanner.empty:
                st.info("No High Vol rows right now. Check **Show all rows** to see full institutional flow.")
                return

        def style_classification(val):
            if "Resistance" in str(val):
                return 'background-color: rgba(255, 82, 82, 0.2); color: #FF5252; font-weight: bold;'
            elif "Support" in str(val):
                return 'background-color: rgba(0, 230, 118, 0.2); color: #00E676; font-weight: bold;'
            else:
                return 'color: #FAFAFA;'

        def style_unusual(val):
            if "High Vol" in str(val):
                return 'background-color: rgba(255, 152, 0, 0.25); color: #FF9800; font-weight: bold;'
            return 'color: #FAFAFA;'

        styled_df = df_scanner.style.map(
            style_classification, subset=['Key Level Classification']
        ).map(
            style_unusual, subset=['Unusual Vol Flag']
        ).format({
            "LTP (₹)": "{:,.2f}",
            "Volume (Lots)": "{:,.0f}",
            "Total Premium (₹)": "₹{:,.2f}",
            "Volume Ratio": "{:.2f}x"
        })

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No trades currently meeting the institutional threshold criteria (Premium ≥ ₹10 Lakhs or Volume ≥ 500 lots).")

# --- DELTA-ADJUSTED GEX HEATMAP (session-time vs strike) ---
def _prepare_session_candles(df_candles: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest trading session between 09:15–15:30 IST."""
    if df_candles is None or df_candles.empty:
        return pd.DataFrame()

    df = df_candles.copy()
    time_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
    if time_col is None:
        return pd.DataFrame()

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    df = df.loc[parsed.notna()].copy()
    parsed = parsed.dropna()
    if parsed.empty:
        return pd.DataFrame()

    ist_tz = pytz.timezone("Asia/Kolkata")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(ist_tz)
    else:
        parsed = parsed.dt.tz_convert(ist_tz)
    df[time_col] = parsed
    _, _, _, _, os_, cs_ = session_hours()
    session_start = datetime.datetime.strptime(os_, "%H:%M").time()
    session_end = datetime.datetime.strptime(cs_, "%H:%M").time()
    df = df[(df[time_col].dt.time >= session_start) & (df[time_col].dt.time <= session_end)]
    if df.empty:
        return pd.DataFrame()

    latest_date = df[time_col].dt.date.max()
    df = df[df[time_col].dt.date == latest_date].sort_values(time_col).reset_index(drop=True)
    df.rename(columns={time_col: "time"}, inplace=True)
    return df
# ============================================================
# GEX Heatmap History – Disk Persistence
# ============================================================

# ============================================================
# Superhuman Decision Log – Disk Persistence
# ============================================================
DECISION_LOG_FILE = "superhuman_decision_log.json"

def _load_decision_log(index_name: str) -> list:
    if not os.path.exists(DECISION_LOG_FILE):
        return []
    try:
        with open(DECISION_LOG_FILE, "r") as f:
            raw = json.load(f)
        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.datetime.now(ist).date()
        log = []
        for e in raw:
            if e.get("index") != index_name:
                continue
            ts = pd.to_datetime(e["ts"])
            if ts.tzinfo is None:
                ts = ist.localize(ts)
            if ts.date() != today:
                continue
            log.append({
                "ts": ts,
                "bias": e["bias"],
                "composite": e["composite"],
                "clarity": e.get("clarity", ""),
            })
        log.sort(key=lambda x: x["ts"])
        return log
    except Exception:
        return []


def _save_decision_log(log: list, index_name: str):
    try:
        existing = []
        if os.path.exists(DECISION_LOG_FILE):
            with open(DECISION_LOG_FILE, "r") as f:
                existing = json.load(f)
        existing = [e for e in existing if e.get("index") != index_name]
        for e in log:
            existing.append({
                "index": index_name,
                "ts": e["ts"].isoformat(),
                "bias": e["bias"],
                "composite": e["composite"],
                "clarity": e.get("clarity", ""),
            })
        ist = pytz.timezone("Asia/Kolkata")
        cutoff = datetime.datetime.now(ist) - datetime.timedelta(days=2)
        cleaned = []
        for e in existing:
            ts = pd.to_datetime(e["ts"])
            if ts.tzinfo is None:
                ts = ist.localize(ts)
            if ts >= cutoff:
                cleaned.append(e)
        with open(DECISION_LOG_FILE, "w") as f:
            json.dump(cleaned, f)
    except Exception:
        pass


GEX_HISTORY_FILE = "gex_heatmap_history.json"

def _load_gex_history(index_name: str, expiry_str: str) -> list:
    """Load today's GEX history from disk. Returns list of dicts."""
    if not os.path.exists(GEX_HISTORY_FILE):
        return []
    try:
        with open(GEX_HISTORY_FILE, "r") as f:
            raw = json.load(f)
        # Keep only matching index + expiry + today's date
        ist = pytz.timezone("Asia/Kolkata")
        today = datetime.datetime.now(ist).date()
        history = []
        for h in raw:
            if h.get("index") != index_name or h.get("expiry") != expiry_str:
                continue
            ts = pd.to_datetime(h["ts"])
            if ts.tzinfo is None:
                ts = ist.localize(ts)
            if ts.date() != today:
                continue
            history.append({
                "ts": ts,
                "strikes": h["strikes"],
                "gex_cr": h["gex_cr"],
            })
        # Sort by time
        history.sort(key=lambda x: x["ts"])
        return history
    except Exception:
        return []


def _save_gex_history(history: list, index_name: str, expiry_str: str):
    """Save history to disk (only today's data for this index/expiry)."""
    try:
        # Load existing file (may contain other indices/expiries)
        existing = []
        if os.path.exists(GEX_HISTORY_FILE):
            with open(GEX_HISTORY_FILE, "r") as f:
                existing = json.load(f)

        # Remove old entries for this index+expiry
        existing = [
            e for e in existing
            if not (e.get("index") == index_name and e.get("expiry") == expiry_str)
        ]

        # Add current history
        for h in history:
            existing.append({
                "index": index_name,
                "expiry": expiry_str,
                "ts": h["ts"].isoformat(),
                "strikes": h["strikes"],
                "gex_cr": h["gex_cr"],
            })

        # Keep only last 2 calendar days to avoid unbounded growth
        ist = pytz.timezone("Asia/Kolkata")
        cutoff = datetime.datetime.now(ist) - datetime.timedelta(days=2)
        cleaned = []
        for e in existing:
            ts = pd.to_datetime(e["ts"])
            if ts.tzinfo is None:
                ts = ist.localize(ts)
            if ts >= cutoff:
                cleaned.append(e)

        with open(GEX_HISTORY_FILE, "w") as f:
            json.dump(cleaned, f)
    except Exception:
        pass


@st.cache_data(ttl=120, show_spinner=False)
def compute_multi_expiry_delta_gex(_api, index_name: str, exchange: str, spot: float, rate: float,
                                   lot_size: int, strike_lo: float, strike_hi: float, months: int = 6):
    """Aggregate Delta-Adjusted GEX (OI) across all expiries within `months` for the index."""
    try:
        df_m = download_master_scrip()
        if df_m is None or df_m.empty or spot <= 0:
            return pd.DataFrame()

        today = datetime.datetime.now().date()
        cutoff = today + datetime.timedelta(days=int(months * 31))
        df_opt = df_m[
            (df_m["name"] == index_name) &
            (df_m["exch_seg"] == exchange) &
            (df_m["instrumenttype"].isin(["OPTIDX", "OPTSTK"]))
        ].copy()
        if df_opt.empty:
            return pd.DataFrame()

        if "expiry_dt" not in df_opt.columns:
            df_opt["expiry_dt"] = pd.to_datetime(df_opt["expiry"], format="%d%b%Y", errors="coerce")
        df_opt = df_opt[df_opt["expiry_dt"].notna()]
        df_opt = df_opt[(df_opt["expiry_dt"].dt.date >= today) & (df_opt["expiry_dt"].dt.date <= cutoff)]

        if "strike_clean" not in df_opt.columns:
            df_opt["strike_clean"] = pd.to_numeric(df_opt["strike"], errors="coerce") / 100.0
        # NFO strikes are in paise; if still huge, divide again
        if df_opt["strike_clean"].median() > spot * 5:
            df_opt["strike_clean"] = df_opt["strike_clean"] / 100.0

        df_opt = df_opt[(df_opt["strike_clean"] >= strike_lo) & (df_opt["strike_clean"] <= strike_hi)]
        if df_opt.empty:
            return pd.DataFrame()

        tokens = [str(t) for t in df_opt["token"].dropna().astype(str).unique()]
        market = {}
        for i in range(0, len(tokens), 50):
            chunk = tokens[i:i + 50]
            res = safe_api_call(_api.getMarketData, "FULL", {exchange: chunk})
            if res and res.get("status") and res.get("data") and res["data"].get("fetched"):
                for item in res["data"]["fetched"]:
                    # Angel SmartAPI uses opnInterest for OI
                    oi_val = item.get("opnInterest", item.get("opInterest", item.get("oi", 0)))
                    market[str(item["symbolToken"])] = {
                        "ltp": float(item.get("ltp", 0) or 0),
                        "oi": float(oi_val or 0),
                    }
            time.sleep(0.25)

        rows = []
        for _, r in df_opt.iterrows():
            tok = str(r["token"])
            md = market.get(tok)
            if not md:
                continue
            oi = float(md.get("oi", 0) or 0)
            ltp = float(md.get("ltp", 0) or 0)
            if oi <= 0 or ltp <= 0:
                continue
            sym = str(r.get("symbol", "") or "")
            opt = "CE" if sym.upper().endswith("CE") else "PE"
            exp_dt = r["expiry_dt"].date() if hasattr(r["expiry_dt"], "date") else pd.to_datetime(r["expiry_dt"]).date()
            dte = max((exp_dt - today).days, 0.05)
            T = dte / 365.0
            K = float(r["strike_clean"])
            flag = "c" if opt == "CE" else "p"
            iv = VolatilityEngine.calculate_iv(ltp, spot, K, T, rate, flag)
            if iv is None or iv <= 0:
                continue
            greeks = VolatilityEngine.calculate_greeks(spot, K, T, rate, iv, flag)
            gex_scale = lot_size * (spot ** 2) * 0.01
            raw_gex = greeks["gamma"] * oi * gex_scale
            if opt == "CE":
                delta_gex = raw_gex * max(float(greeks["delta"]), 0.0)
            else:
                delta_gex = -raw_gex * abs(float(greeks["delta"]))
            rows.append({"Strike": K, "Net_Delta_GEX_OI": delta_gex})

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = df.groupby("Strike", as_index=False)["Net_Delta_GEX_OI"].sum()
        return agg.sort_values("Strike").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()



def render_delta_gex_heatmap(data: dict, index_name: str, expiry_str: str, heatmap_tf_label: str):
    """Render strike × session-time Delta-Adjusted GEX heatmap with persistent history."""
    chain_results = data.get("chain_results") or []
    if not chain_results:
        st.info("No option chain data available for GEX heatmap.")
        return

    df_chain = pd.DataFrame(chain_results)
    if df_chain.empty or "Net_Delta_GEX_OI" not in df_chain.columns:
        st.info("Delta-adjusted GEX is not available to build the heatmap.")
        return

    df_chain = df_chain.sort_values("Strike").reset_index(drop=True)
    strikes = df_chain["Strike"].astype(float).tolist()
    gex_cr = (df_chain["Net_Delta_GEX_OI"].astype(float) / 1e7).tolist()
    spot = float(data.get("spot_price", 0.0))
    flip = float(data.get("levels", {}).get("Zero_Gamma_Flip", spot) or spot)

    # ---- derive metrics needed by the two alert modules ----
    pos_gex_above = float(df_chain.loc[
        (df_chain["Strike"] >= spot) & (df_chain["Net_Delta_GEX_OI"] > 0), "Net_Delta_GEX_OI"
    ].sum() / 1e7)
    neg_gex_above = float(df_chain.loc[
        (df_chain["Strike"] >= spot) & (df_chain["Net_Delta_GEX_OI"] < 0), "Net_Delta_GEX_OI"
    ].sum() / 1e7)

    otm_calls = df_chain[(df_chain["Strike"] >= spot)].copy()
    otm_puts = df_chain[(df_chain["Strike"] < spot)].copy()
    otm_call_iv = 0.0
    otm_put_iv = 0.0
    if not otm_calls.empty and "C_IV_val" in otm_calls.columns:
        nearest_c = otm_calls.iloc[(otm_calls["Strike"] - spot).abs().argsort()[:1]]
        otm_call_iv = float(nearest_c["C_IV_val"].iloc[0]) * 100.0 if not nearest_c.empty else 0.0
    if not otm_puts.empty and "P_IV_val" in otm_puts.columns:
        nearest_p = otm_puts.iloc[(otm_puts["Strike"] - spot).abs().argsort()[:1]]
        otm_put_iv = float(nearest_p["P_IV_val"].iloc[0]) * 100.0 if not nearest_p.empty else 0.0

    df_candles = data.get("df_candles", pd.DataFrame())
    vol_ratio = 0.0
    if not df_candles.empty and "volume" in df_candles.columns and len(df_candles) >= 21:
        recent_vol = float(df_candles["volume"].iloc[-1])
        ma20 = float(df_candles["volume"].iloc[-21:-1].mean())
        vol_ratio = recent_vol / ma20 if ma20 > 0 else 0.0

    # ---- Load persistent history from disk ----
    now_ist = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    gex_hist = _load_gex_history(index_name, expiry_str)
    alert_hist = st.session_state.get("alert_metrics_history", [])

    # Reset if strike window changed significantly
    if gex_hist and (len(gex_hist[-1]["strikes"]) != len(strikes) or
                     abs(gex_hist[-1]["strikes"][0] - strikes[0]) > 1 or
                     abs(gex_hist[-1]["strikes"][-1] - strikes[-1]) > 1):
        gex_hist = []
        alert_hist = []

    # ---- Append new snapshot ----
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_hours = (now_ist.weekday() < 5) and (market_open <= now_ist <= market_close)

    # Always define cutoff so it is in scope
    cutoff = now_ist - datetime.timedelta(hours=8)

    should_append = False
    if is_market_hours:
        if not gex_hist:
            should_append = True
        else:
            last_ts = gex_hist[-1]["ts"]
            if (now_ist - last_ts).total_seconds() >= 150:
                should_append = True
    else:
        if not gex_hist:
            should_append = True

    if should_append:
        gex_hist.append({"ts": now_ist, "strikes": list(strikes), "gex_cr": list(gex_cr)})
        gex_hist = [h for h in gex_hist if h["ts"] >= cutoff]
        _save_gex_history(gex_hist, index_name, expiry_str)

        alert_hist.append({
            "ts": now_ist,
            "spot": spot,
            "flip": flip,
            "pos_gex_above": pos_gex_above,
            "neg_gex_above": neg_gex_above,
            "otm_call_iv": otm_call_iv,
            "otm_put_iv": otm_put_iv,
            "vol_ratio": vol_ratio,
        })
        alert_hist = [h for h in alert_hist if h["ts"] >= cutoff]
        st.session_state["alert_metrics_history"] = alert_hist

    # Also keep a copy in session_state for the rest of the app
    st.session_state["gex_heatmap_history"] = gex_hist

    # ---- session candles ----
    heatmap_tf_label = heatmap_tf_label or st.session_state.get("heatmap_timeframe", "5 min")
    chart_tf = st.session_state.get("selected_timeframe", "5 min")
    df_session = pd.DataFrame()

    if heatmap_tf_label == chart_tf:
        df_session = _prepare_session_candles(data.get("df_candles", pd.DataFrame()))

    if df_session.empty:
        smart_api = get_smart_api_client()
        spot_token, spot_exch, _ = INDEX_TOKEN_MAP.get(index_name, ("99926000", "NSE", "NFO"))
        api_interval, lookback_days = interval_mapping.get(heatmap_tf_label, ("FIVE_MINUTE", 15))
        if smart_api:
            df_hm, _ = fetch_candles_with_holiday_fallback(
                smart_api, spot_token, spot_exch, api_interval, lookback_days, index_name
            )
            df_session = _prepare_session_candles(df_hm)

    if df_session.empty:
        st.info("Unable to load session candles for the GEX heatmap.")
        return

    time_labels = pd.to_datetime(df_session["time"]).dt.strftime("%H:%M").tolist()
    spot_path = df_session["close"].astype(float).tolist()
    session_date_str = pd.to_datetime(df_session["time"].iloc[-1]).strftime("%d %B %Y")
    session_times = pd.to_datetime(df_session["time"]).tolist()

    n_strikes = len(strikes)
    n_times = len(time_labels)
    if n_strikes == 0 or n_times == 0:
        st.info("Insufficient strike/time points for the GEX heatmap.")
        return

    # ---- Build evolving GEX matrix (no longer paints latest GEX on whole day) ----
    gex_matrix = np.full((n_strikes, n_times), np.nan, dtype=float)
    current_vec = np.array(gex_cr, dtype=float)

    if len(gex_hist) <= 1:
        # Only one (or zero) snapshot → paint it uniformly across the whole session
        # so the user always sees a proper coloured band.
        paint_vec = current_vec if not gex_hist else np.array(gex_hist[-1]["gex_cr"], dtype=float)
        # Align length just in case
        if len(paint_vec) != n_strikes:
            paint_vec = current_vec
        gex_matrix[:] = paint_vec[:, None]

    else:
        # Multiple snapshots → build true time evolution
        hist_times = []
        hist_vecs = []
        for h in gex_hist:
            ht = h["ts"]
            if getattr(ht, "tzinfo", None) is None:
                ht = pytz.timezone("Asia/Kolkata").localize(ht)

            src_strikes = h["strikes"]
            src_gex = np.array(h["gex_cr"], dtype=float)

            if len(src_strikes) == n_strikes and abs(src_strikes[0] - strikes[0]) < 1:
                vec = src_gex
            else:
                src_map = dict(zip(src_strikes, src_gex))
                vec = np.array([
                    src_map.get(min(src_strikes, key=lambda x: abs(x - s)), 0.0)
                    for s in strikes
                ])
            hist_times.append(ht)
            hist_vecs.append(vec)

        for t_idx, t in enumerate(session_times):
            if getattr(t, "tzinfo", None) is None:
                t = pytz.timezone("Asia/Kolkata").localize(t)

            # Find the latest snapshot that is <= this candle time
            chosen_idx = None
            for i in range(len(hist_times) - 1, -1, -1):
                if hist_times[i] <= t:
                    chosen_idx = i
                    break

            if chosen_idx is not None:
                gex_matrix[:, t_idx] = hist_vecs[chosen_idx]
            else:
                # Before the first snapshot → use the first snapshot
                gex_matrix[:, t_idx] = hist_vecs[0]

        # Forward-fill any remaining gaps
        last_valid = gex_matrix[:, 0].copy()
        for t_idx in range(n_times):
            if not np.isnan(gex_matrix[0, t_idx]):
                last_valid = gex_matrix[:, t_idx].copy()
            else:
                gex_matrix[:, t_idx] = last_valid
    # Colour scale
    finite_vals = gex_matrix[np.isfinite(gex_matrix)]
    if len(finite_vals) > 0:
        p95 = float(np.nanpercentile(np.abs(finite_vals), 95))
        max_abs_gex = max(p95, 0.05)
    else:
        max_abs_gex = 0.05

    gex_colorscale = [
        [0.0, "#b71c1c"],
        [0.25, "#e53935"],
        [0.45, "#ffee58"],
        [0.55, "#ffee58"],
        [0.75, "#43a047"],
        [1.0, "#1b5e20"],
    ]

    hm_c1, hm_c2 = st.columns([0.65, 0.35])
    with hm_c1:
        heading_ribbon(
            f"🔥 Delta-Adjusted GEX Heatmap ({index_name})",
            "Strike × session time. Each snapshot stores delta-adjusted GEX (OI) per strike.<br>"
            "GEX_i × |Δ_i|. Green = long-gamma; red = short-gamma. Cyan = spot path.<br>"
            "Enable Auto-Refresh so the band evolves instead of one snapshot painted all day.",
        )
    with hm_c2:
        st.markdown(
            f"<div class='update-timestamp'>Expiry: {expiry_str} | Session: {session_date_str} | "
            f"{heatmap_tf_label} | Snapshots: {len(gex_hist)}</div>",
            unsafe_allow_html=True,
        )

    n_snap = len(gex_hist)
    if n_snap <= 1:
        st.caption(
            "Strike vs IST session time. Currently showing the **latest live GEX snapshot**. "
            "Enable **Auto-Refresh** and leave the dashboard open – new snapshots will be saved to disk "
            "and the heatmap will start showing true evolution of GEX through the day. "
            "Green = long-gamma; red = short-gamma. Cyan = spot path."
        )
    else:
        first_ts = gex_hist[0]["ts"].strftime("%H:%M")
        last_ts = gex_hist[-1]["ts"].strftime("%H:%M")
        st.caption(
            f"Strike vs IST session time. Colour = live Delta-Adjusted Net GEX (₹ Cr) from "
            f"**{n_snap} snapshots** ({first_ts} → {last_ts}). "
            "History is persisted to disk and survives browser refresh. "
            "Green = long-gamma / pinning; red = short-gamma / acceleration. Cyan line = spot path."
        )

    fig_hm = plt_go.Figure()
    fig_hm.add_trace(
        plt_go.Heatmap(
            z=gex_matrix,
            x=time_labels,
            y=strikes,
            zmin=-max_abs_gex,
            zmax=max_abs_gex,
            zmid=0,
            colorscale=gex_colorscale,
            colorbar=dict(title="Net Δ-GEX (₹ Cr)"),
            hovertemplate="Time: %{x}<br>Strike: %{y}<br>Net Δ-GEX: %{z:.2f} Cr<extra></extra>",
        )
    )
    fig_hm.add_trace(
        plt_go.Scatter(
            x=time_labels,
            y=spot_path,
            mode="lines+markers",
            name=f"{index_name} Spot",
            line=dict(color="cyan", width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x}<br>Spot: %{y:,.2f}<extra></extra>",
        )
    )
    fig_hm.update_layout(
        title=dict(
            text=f"<b>{index_name} Delta-Adjusted GEX Heatmap</b> ({heatmap_tf_label}) | "
                 f"Session: {session_date_str} | Expiry: {expiry_str}",
            x=0.01,
        ),
        xaxis_title="IST Time",
        yaxis_title="Strike Price",
        height=520,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_hm.update_xaxes(type="category")
    fig_hm.update_yaxes(tickformat="d")
    st.plotly_chart(fig_hm, use_container_width=True)
    # =====================================================================
    #  ALERT MODULES – Situation 1 (Exit / Risk-Off) & Situation 2 (Long Entry)
    # =====================================================================
    def _pct_drop(old, new):
        if old is None or old <= 0:
            return 0.0
        return max(0.0, (old - new) / old * 100.0)

    def _iv_change_bps(old_iv, new_iv):
        if old_iv is None:
            return 0.0
        return (new_iv - old_iv)  # already in percentage points → ×100 for bps later if needed

    # Look back ~10 min and ~5 min in the alert history
    t_10 = now_ist - datetime.timedelta(minutes=10)
    t_5 = now_ist - datetime.timedelta(minutes=5)

    snap_10 = None
    snap_5 = None
    for h in alert_hist:
        if h["ts"] <= t_10:
            snap_10 = h
        if h["ts"] <= t_5:
            snap_5 = h

    cur = alert_hist[-1] if alert_hist else {
        "pos_gex_above": pos_gex_above,
        "neg_gex_above": neg_gex_above,
        "otm_call_iv": otm_call_iv,
        "otm_put_iv": otm_put_iv,
        "vol_ratio": vol_ratio,
        "spot": spot,
        "flip": flip,
    }

    # ---------- Situation 1: Non-Directional Exit / Risk-Off ----------
    # A – Wall Collapse: cumulative positive GEX above spot drops ≥ 25% in 10 min
    wall_drop = _pct_drop(snap_10["pos_gex_above"] if snap_10 else None, cur["pos_gex_above"])
    crit_a_exit = wall_drop >= 25.0

    # B – Vol Expansion: OTM Call IV or Put IV rises ≥ +0.8% (80 bps) in 5 min
    call_iv_chg = _iv_change_bps(snap_5["otm_call_iv"] if snap_5 else None, cur["otm_call_iv"])
    put_iv_chg = _iv_change_bps(snap_5["otm_put_iv"] if snap_5 else None, cur["otm_put_iv"])
    crit_b_exit = (call_iv_chg >= 0.8) or (put_iv_chg >= 0.8)

    # C – GEX Flip: spot within 0.3% of flip level OR already inside a negative-GEX strike zone
    dist_to_flip_pct = abs(cur["spot"] - cur["flip"]) / cur["spot"] * 100.0 if cur["spot"] > 0 else 999.0
    in_neg_zone = cur["neg_gex_above"] < 0 and abs(cur["neg_gex_above"]) > 0.01
    crit_c_exit = (dist_to_flip_pct <= 0.3) or in_neg_zone

    exit_alert = crit_a_exit or crit_b_exit or crit_c_exit

    # ---------- Situation 2: Long Position Entry (Directional Breakout) ----------
    # A – Gamma Fuel: negative GEX present directly above spot OR positive GEX collapsed ≥ 40% in 10 min
    has_neg_above = cur["neg_gex_above"] < -0.01
    pos_collapse = _pct_drop(snap_10["pos_gex_above"] if snap_10 else None, cur["pos_gex_above"])
    crit_a_long = has_neg_above or (pos_collapse >= 40.0)

    # B – Aggressive Demand: OTM Call IV spikes ≥ +1.5% (150 bps) in 5 min
    crit_b_long = call_iv_chg >= 1.5

    # C – Volume Confirmation: current 5-min volume ≥ 1.5× its 20-period MA
    crit_c_long = cur["vol_ratio"] >= 1.5

    long_alert = crit_a_long and crit_b_long and crit_c_long   # high-confirmation: all three required

    # Alert criteria computed above; store for full-width 3-col ribbon rendered outside this column
    st.session_state["_live_alert_snapshot"] = {
        "exit_alert": exit_alert, "long_alert": long_alert,
        "crit_a_exit": crit_a_exit, "crit_b_exit": crit_b_exit, "crit_c_exit": crit_c_exit,
        "crit_a_long": crit_a_long, "crit_b_long": crit_b_long, "crit_c_long": crit_c_long,
        "wall_drop": wall_drop, "call_iv_chg": call_iv_chg, "put_iv_chg": put_iv_chg,
        "dist_to_flip_pct": dist_to_flip_pct, "pos_collapse": pos_collapse, "cur": cur,
    }


def _compute_basket_live_totals(data: dict):
    """Return (totals_dict, legs_list) for current basket."""
    if not st.session_state.get("basket_legs"):
        return None, []
    market_data_store = data.get("market_data", {})
    basket_tokens_info = data.get("basket_tokens_info", {})
    F_val = data.get("F", data["spot_price"])
    T_val = data.get("T", 1e-5)
    hv_val = data.get("index_hv", 0.15)
    calculated_legs = []
    tot_pnl = tot_delta = tot_gamma = tot_theta = tot_vega = 0.0
    for idx, leg in enumerate(st.session_state["basket_legs"]):
        k, t, act, qty = leg["strike"], leg["type"], leg["action"], leg["qty"]
        tok = basket_tokens_info.get(f"{k}_{t}", "")
        ltp = market_data_store.get(tok, {}).get("ltp", 0.0) if tok else 0.0
        entry_p = leg["entry_price"] if leg["entry_price"] > 0 else ltp
        pricing_p = ltp if ltp > 0 else entry_p
        leg_iv = VolatilityEngine.calculate_iv(pricing_p, F_val, k, T_val, rate_param, "c" if t == "CE" else "p")
        if leg_iv == 0.0:
            leg_iv = hv_val
        greeks = VolatilityEngine.calculate_greeks(F_val, k, T_val, rate_param, leg_iv, "c" if t == "CE" else "p")
        mult = 1.0 if act == "BUY" else -1.0
        pnl_per_unit = (ltp - entry_p) if act == "BUY" else (entry_p - ltp)
        leg_pnl = pnl_per_unit * qty if ltp > 0 else 0.0
        pos_delta = greeks["delta"] * qty * mult
        pos_gamma = greeks["gamma"] * qty * mult
        pos_theta = greeks["theta"] * qty * mult
        pos_vega = greeks["vega"] * qty * mult
        tot_pnl += leg_pnl
        tot_delta += pos_delta
        tot_gamma += pos_gamma
        tot_theta += pos_theta
        tot_vega += pos_vega
        calculated_legs.append({
            "Leg": idx + 1, "Action": act, "Strike": k, "Type": t, "Qty": qty,
            "Entry (₹)": round(entry_p, 2), "LTP (₹)": round(ltp, 2), "P&L (₹)": round(leg_pnl, 2),
            "Delta (Δ)": round(pos_delta, 2), "Gamma (γ)": round(pos_gamma, 4),
            "Theta (θ/Day)": round(pos_theta, 2), "Vega (ν)": round(pos_vega, 2),
        })
    totals = {"pnl": tot_pnl, "delta": tot_delta, "gamma": tot_gamma, "theta": tot_theta, "vega": tot_vega}
    return totals, calculated_legs


def render_basket_metrics_block(data: dict):
    """Compact P&L / Greeks metrics only (for left of Z-Scores under alerts)."""
    heading_ribbon("🧺 Basket Greeks",
            "Sum of Δ, Γ, Θ, Vega across strategy legs × qty × lot. Net exposure of the basket.")
    totals, _ = _compute_basket_live_totals(data)
    if totals is None:
        st.caption("No legs – add from sidebar")
        return
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P&L", f"₹{totals['pnl']:,.0f}")
    m2.metric("Δ", f"{totals['delta']:,.1f}")
    m3.metric("γ", f"{totals['gamma']:.4f}")
    m4.metric("θ/d", f"{totals['theta']:,.1f}")
    m5.metric("ν", f"{totals['vega']:,.1f}")


def render_basket_table_fullwidth(data: dict):
    """Full-width legs table + remove buttons + fixed historical analytics."""
    totals, calculated_legs = _compute_basket_live_totals(data)
    if totals is None:
        return

    st.markdown("---")
    heading_ribbon("🧺 Strategy Basket – Legs",
            "Each listed option leg with strike, type, side, qty. Greeks from BS on live LTP / IV.")
    df_b = pd.DataFrame(calculated_legs)
    tbl_c, del_c = st.columns([0.92, 0.08])
    with tbl_c:
        st.dataframe(df_b, use_container_width=True, hide_index=True)
    with del_c:
        st.caption("Del")
        for idx in range(len(st.session_state["basket_legs"])):
            if st.button("✕", key=f"del_leg_full_{idx}", help=f"Remove leg #{idx+1}"):
                st.session_state["basket_legs"].pop(idx)
                sync_local_storage()
                st.rerun()

    with st.expander("📊 Basket Historical Analytics (Cum. Θ Decay %, Daily Θ, Vanna, Charm)", expanded=False):
        st.caption(
            "Cum. Θ Decay % = time-integrated |θ| as % of initial extrinsic (smooth accrual). "
            "Daily Θ = signed position theta (long options are negative by design). "
            "Deep-ITM bars use HV fallback to avoid IV-solver spikes."
        )
        smart_api = get_smart_api_client()
        if not (smart_api and not df_master.empty and st.session_state["basket_legs"]):
            st.info("API session needed for historical basket charts.")
            return

        hv_val = data.get("index_hv", 0.15)
        b_hist_dfs = []
        for leg in st.session_state["basket_legs"]:
            k, t, act, qty = leg["strike"], leg["type"], leg["action"], leg["qty"]
            mult = 1.0 if act == "BUY" else -1.0
            tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, k, t)
            if not tok:
                continue
            leg_df = fetch_history(smart_api, tok, days=30, spot_token=default_token, spot_exchange=spot_exchange)
            if leg_df.empty or len(leg_df) < 3:
                continue

            leg_df = leg_df.copy()
            leg_df["Expiry_Date"] = target_expiry_dt.date()
            expiry = pd.to_datetime(leg_df["Expiry_Date"])
            timestamp = pd.to_datetime(leg_df["Raw_Timestamp"])
            try:
                if timestamp.dt.tz is not None:
                    timestamp = timestamp.dt.tz_localize(None)
            except Exception:
                pass
            try:
                if getattr(expiry.dt, "tz", None) is not None:
                    expiry = expiry.dt.tz_localize(None)
            except Exception:
                pass
            leg_df["DTE"] = ((expiry - timestamp).dt.total_seconds() / 86400.0).clip(lower=0.001)
            leg_df["T"] = leg_df["DTE"] / 365.0

            greeks_df = leg_df.apply(compute_greeks, axis=1, K=k, r=rate_param, option_type=t)
            greeks_df.columns = ["IV_%", "Theta", "Theta_Decay_Pct", "Gamma", "Vanna", "Charm", "Extrinsic_Val", "Is_Pure_Intrinsic"]

            # Fallback pure-intrinsic / failed IV → HV-based greeks
            for i in range(len(greeks_df)):
                if bool(greeks_df.iloc[i]["Is_Pure_Intrinsic"]) or float(greeks_df.iloc[i]["IV_%"]) <= 0.5:
                    S = float(leg_df.iloc[i]["Spot_Price"])
                    T = float(leg_df.iloc[i]["T"])
                    if T > 1e-5 and S > 0:
                        g = VolatilityEngine.calculate_greeks(S, k, T, rate_param, hv_val, "c" if t == "CE" else "p")
                        greeks_df.iat[i, 1] = g["theta"]
                        greeks_df.iat[i, 3] = g["gamma"]
                        # keep vanna/charm from BS if available; else 0
                        if "vanna" in g:
                            greeks_df.iat[i, 4] = g["vanna"]
                        if "charm" in g:
                            greeks_df.iat[i, 5] = g["charm"]

            leg_df["Theta_Scaled"] = greeks_df["Theta"].values * qty * mult
            leg_df["Vanna_Scaled"] = greeks_df["Vanna"].values * qty * mult
            leg_df["Charm_Scaled"] = greeks_df["Charm"].values * qty * mult
            leg_df["Extrinsic"] = greeks_df["Extrinsic_Val"].values * qty

            ts = pd.to_datetime(leg_df["Raw_Timestamp"])
            try:
                if ts.dt.tz is not None:
                    ts = ts.dt.tz_localize(None)
            except Exception:
                pass
            dt_days = ts.diff().dt.total_seconds().fillna(0.0) / 86400.0
            dt_days = dt_days.clip(lower=0.0, upper=3.0)

            # Midpoint integration of signed theta over each bar
            theta_mid = (leg_df["Theta_Scaled"] + leg_df["Theta_Scaled"].shift(1).fillna(leg_df["Theta_Scaled"].iloc[0])) / 2.0
            bar_theta_pnl = theta_mid * dt_days
            # Decay cost: positive when time hurts the position (longs)
            bar_decay_cost = (-bar_theta_pnl).clip(lower=0.0)
            leg_df["Bar_Decay_Cost"] = bar_decay_cost
            leg_df["Cum_Decay_Cost"] = leg_df["Bar_Decay_Cost"].cumsum()

            init_extrinsic = max(
                float(leg_df["Extrinsic"].iloc[0]),
                abs(float(leg_df["Theta_Scaled"].iloc[0])) * max(float(leg_df["DTE"].iloc[0]), 1.0),
                1.0,
            )
            leg_df["Init_Extrinsic"] = init_extrinsic

            b_hist_dfs.append(leg_df[["Date", "Raw_Timestamp", "Theta_Scaled", "Vanna_Scaled", "Charm_Scaled", "Cum_Decay_Cost", "Init_Extrinsic"]])

        if not b_hist_dfs:
            st.info("Unable to fetch historical data for strategy basket contracts.")
            return

        combined = b_hist_dfs[0].copy()
        for nxt in b_hist_dfs[1:]:
            combined = pd.merge(combined, nxt, on=["Date", "Raw_Timestamp"], how="inner", suffixes=("", "_sub"))
            for c_col in ["Theta_Scaled", "Vanna_Scaled", "Charm_Scaled", "Cum_Decay_Cost", "Init_Extrinsic"]:
                combined[c_col] = combined[c_col] + combined[f"{c_col}_sub"]
                combined.drop(columns=[f"{c_col}_sub"], inplace=True)
        combined = combined.sort_values("Raw_Timestamp", ascending=True).reset_index(drop=True)

        init_ext = max(float(combined["Init_Extrinsic"].iloc[0]), 1.0)
        combined["Cum_Theta_Decay_Pct"] = (combined["Cum_Decay_Cost"] / init_ext * 100.0).clip(0, 100)

        # Clip single-bar IV-solver spikes on higher-order greeks
        for col in ["Vanna_Scaled", "Charm_Scaled", "Theta_Scaled"]:
            s = combined[col]
            mu, sd = float(s.median()), float(s.std()) if len(s) > 2 else 0.0
            if sd and sd > 0:
                combined[col] = s.clip(mu - 4 * sd, mu + 4 * sd)

        def _plot_bm(df, y, title, color, is_pct=False):
            fig = px.line(df, x="Date", y=y, title=title, markers=True)
            fig.update_traces(line_color=color, marker=dict(size=4))
            fig.update_xaxes(type="category", title="", nticks=10)
            if is_pct:
                ymax = max(5.0, float(df[y].max()) * 1.15)
                fig.update_yaxes(range=[0, ymax], ticksuffix="%")
            fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=30, b=10))
            return fig

        bc1, bc2 = st.columns(2)
        with bc1:
            st.plotly_chart(_plot_bm(combined, "Cum_Theta_Decay_Pct", "Cum. Θ Decay % (time-integrated)", "#00E676", True), use_container_width=True)
        with bc2:
            st.plotly_chart(_plot_bm(combined, "Theta_Scaled", "Daily Θ P&L (₹/Day, signed)", "#FF9800"), use_container_width=True)
        bc3, bc4 = st.columns(2)
        with bc3:
            st.plotly_chart(_plot_bm(combined, "Vanna_Scaled", "Vanna (scaled)", "#00BFFF"), use_container_width=True)
        with bc4:
            st.plotly_chart(_plot_bm(combined, "Charm_Scaled", "Charm (scaled)", "#E040FB"), use_container_width=True)



def render_futures_cvd_chart(data: dict):
    """Cumulative Volume Delta proxy from near-month futures candles.
    SmartAPI does not provide buy/sell volume for index futures, so we use a
    standard directional proxy: +volume when close >= open (up bar), -volume otherwise.
    """
    df_fut = data.get("df_futures", pd.DataFrame())
    if df_fut is None or df_fut.empty or len(df_fut) < 5:
        st.info("Futures CVD not available – need futures candles with volume.")
        return

    df = df_fut.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["session_date"] = df["time"].dt.date
    # Latest session only for a clean CVD
    latest = sorted(df["session_date"].unique())[-1]
    df = df[df["session_date"] == latest].sort_values("time").reset_index(drop=True)
    if df.empty or "volume" not in df.columns:
        st.info("No volume on futures candles for CVD.")
        return

    # Directional volume proxy
    df["signed_vol"] = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    df["cvd"] = df["signed_vol"].cumsum()
    df["time_str"] = df["time"].dt.strftime("%H:%M")

    latest_cvd = float(df["cvd"].iloc[-1])
    colour = "#00E676" if latest_cvd >= 0 else "#FF5252"

    st.markdown(
        f"<span style='font-weight:700;color:#00E676;font-size:14px;'>"
        f"📉 Futures CVD (proxy) — latest session {latest}</span>",
        unsafe_allow_html=True
    )
    st.caption(
        "Cumulative Volume Delta from near-month futures. "
        "SmartAPI does not supply buy/sell side volume for index futures, "
        "so each bar’s volume is signed + if close≥open, − otherwise. "
        "This is a widely used institutional proxy."
    )

    fig = plt_go.Figure()
    fig.add_trace(plt_go.Scatter(
        x=df["time_str"], y=df["cvd"],
        mode="lines", name="CVD",
        line=dict(color=colour, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.08)" if latest_cvd >= 0 else "rgba(255,82,82,0.08)",
    ))
    fig.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"CVD: {latest_cvd:,.0f}", x=0.01, font=dict(size=12)),
        xaxis=dict(type="category", nticks=10),
        yaxis=dict(title="Cumulative signed volume"),
        showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)



def _remember_book5(snap: dict):
    if not snap or not snap.get("ok"):
        return
    hist = st.session_state.get("book5_hist") or []
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")
    hist.append({
        "t": now,
        "ltp": snap.get("ltp"),
        "bids": snap.get("bids5") or [],
        "asks": snap.get("asks5") or [],
    })
    st.session_state["book5_hist"] = hist[-480:]


def fetch_futures_book_snapshot(smart_api, index_name: str, fut_token: str) -> dict:
    """One FULL snapshot of near-month futures book. SmartAPI fields vary by version."""
    empty = {"bid_qty": 0.0, "ask_qty": 0.0, "bid_qty_lots": 0.0, "ask_qty_lots": 0.0,
             "best_bid": 0.0, "best_ask": 0.0, "ok": False}
    if not smart_api or not fut_token:
        return empty
    try:
        _, _, fut_exch = INDEX_TOKEN_MAP.get(index_name, ("99926000", "NSE", "NFO"))
        lot = float(LOT_SIZES.get(index_name, 65) or 65)
        res = safe_api_call(smart_api.getMarketData, "FULL", {fut_exch: [str(fut_token)]})
        if not (res and res.get("status") and res.get("data") and res["data"].get("fetched")):
            return empty
        item = res["data"]["fetched"][0]
        depth = item.get("depth") or {}
        buy_lvls = depth.get("buy") or depth.get("Buy") or []
        sell_lvls = depth.get("sell") or depth.get("Sell") or []

        def _sum_qty(levels):
            tot = 0.0
            for lv in levels or []:
                tot += float(lv.get("quantity", lv.get("qty", 0)) or 0)
            return tot

        bid_qty = float(item.get("totBuyQuan", item.get("totalBuyQuantity", 0)) or 0)
        ask_qty = float(item.get("totSellQuan", item.get("totalSellQuantity", 0)) or 0)
        if bid_qty <= 0:
            bid_qty = _sum_qty(buy_lvls)
        if ask_qty <= 0:
            ask_qty = _sum_qty(sell_lvls)
        if bid_qty <= 0:
            bid_qty = float(item.get("bestBidQty", item.get("bidQty", 0)) or 0)
        if ask_qty <= 0:
            ask_qty = float(item.get("bestAskQty", item.get("askQty", 0)) or 0)

        best_bid = float(item.get("bestBidPrice", item.get("bidPrice", 0)) or 0)
        best_ask = float(item.get("bestAskPrice", item.get("askPrice", 0)) or 0)

        # Angel often reports quantity in units. Convert to lots for alert thresholds.
        bid_lots = bid_qty / lot if lot > 0 else bid_qty
        ask_lots = ask_qty / lot if lot > 0 else ask_qty
        # If already looks like lots (small vs units), keep raw
        if bid_qty > 0 and bid_qty < 20000:
            bid_lots, ask_lots = bid_qty, ask_qty

        ltp = float(item.get("ltp", item.get("lastPrice", 0)) or 0)
        traded_vol = float(
            item.get("tradeVolume") or item.get("volume") or item.get("vol") or 0
        )
        def _lvls(levels, n=5):
            out = []
            for lv in (levels or [])[:n]:
                px = float(lv.get("price", lv.get("pr", 0)) or 0)
                qty = float(lv.get("quantity", lv.get("qty", 0)) or 0)
                if px > 0:
                    out.append({"px": px, "qty": qty})
            return out
        bids5 = _lvls(buy_lvls, 5)
        asks5 = _lvls(sell_lvls, 5)
        return {
            "bid_qty": bid_qty, "ask_qty": ask_qty,
            "bid_qty_lots": bid_lots, "ask_qty_lots": ask_lots,
            "best_bid": best_bid, "best_ask": best_ask,
            "ltp": ltp, "traded_vol": traded_vol,
            "bids5": bids5, "asks5": asks5,
            "ok": (bid_qty > 0 or ask_qty > 0 or ltp > 0),
        }
    except Exception:
        return empty


def update_liq_delta_history(snap: dict, index_name: str) -> list:
    """Append book snapshot, bid/ask change, and tick-rule incremental CVD."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(ist)
    hist = list(st.session_state.get("liq_delta_history") or [])
    if not hist:
        r = _redis_client()
        if r is not None:
            try:
                raw = r.get(f"liq:{index_name}:{_ist_now().date()}")
                if raw:
                    hist = json.loads(raw)
            except Exception:
                pass
    prev = hist[-1] if hist else None
    bid = float(snap.get("bid_qty_lots") or 0)
    ask = float(snap.get("ask_qty_lots") or 0)
    bid_chg = bid - float(prev["bid_qty_lots"]) if prev else 0.0
    ask_chg = ask - float(prev["ask_qty_lots"]) if prev else 0.0
    net = bid_chg - ask_chg

    ltp = float(snap.get("ltp") or 0)
    vol = float(snap.get("traded_vol") or 0)
    best_bid = float(snap.get("best_bid") or 0)
    best_ask = float(snap.get("best_ask") or 0)
    mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else ltp

    # Tick rule: incremental printed volume since last snapshot
    dvol = 0.0
    if prev and vol > 0:
        prev_vol = float(prev.get("traded_vol") or 0)
        dvol = max(vol - prev_vol, 0.0)
        if dvol == 0 and vol < prev_vol:
            dvol = 0.0  # session volume reset
    signed = 0.0
    if dvol > 0 and ltp > 0:
        prev_ltp = float(prev.get("ltp") or 0) if prev else 0.0
        if best_ask > 0 and ltp >= best_ask:
            signed = dvol          # lift the offer
        elif best_bid > 0 and ltp <= best_bid:
            signed = -dvol         # hit the bid
        elif prev_ltp > 0:
            signed = dvol if ltp >= prev_ltp else -dvol
        elif mid > 0:
            signed = dvol if ltp >= mid else -dvol
    prev_cvd = float(prev.get("tick_cvd") or 0) if prev else 0.0
    tick_cvd = prev_cvd + signed

    rec = {
        "ts": now,
        "index": index_name,
        "bid_qty_lots": bid,
        "ask_qty_lots": ask,
        "bid_change": bid_chg,
        "ask_change": ask_chg,
        "net_liq": net,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "ltp": ltp,
        "traded_vol": vol,
        "dvol": dvol,
        "signed_vol": signed,
        "tick_cvd": tick_cvd,
    }
    hist.append(rec)
    cutoff = now - datetime.timedelta(minutes=90)
    cleaned = []
    for h in hist:
        ts = h["ts"]
        if getattr(ts, "tzinfo", None) is None:
            ts = ist.localize(ts) if hasattr(ist, "localize") else ts
        if ts >= cutoff and h.get("index", index_name) == index_name:
            cleaned.append(h)
    st.session_state["liq_delta_history"] = cleaned[-180:]
    r = _redis_client()
    if r is not None:
        try:
            day = _ist_now().date()
            r.setex(f"liq:{index_name}:{day}", 48 * 3600, json.dumps(cleaned[-180:], default=str))
            _prune_redis_old_days(r, f"liq:{index_name}:", day)
        except Exception:
            pass
    return st.session_state["liq_delta_history"]



def _prominence_nodes(vol_at: np.ndarray, mids: np.ndarray, prominence_factor: float = 0.35):
    """HVN = peaks on volume. LVN = peaks on inverted volume, kept if valley depth
    vs nearest surrounding HVNs is >= prominence_factor * surrounding peak volume."""
    n = len(vol_at)
    if n < 3:
        return [], []
    vmax = float(np.max(vol_at)) or 1.0
    hvn_idx = []
    for i in range(1, n - 1):
        if vol_at[i] >= vol_at[i - 1] and vol_at[i] >= vol_at[i + 1] and vol_at[i] >= 0.20 * vmax:
            hvn_idx.append(i)
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
        left_p = vol_at[left[-1]]
        right_p = vol_at[right[0]]
        surround = min(float(left_p), float(right_p))
        depth = surround - float(vol_at[i])
        if surround > 0 and depth >= prominence_factor * surround:
            lvn_idx.append(i)
    hvn = [float(mids[i]) for i in hvn_idx]
    lvn = [float(mids[i]) for i in lvn_idx]
    return hvn, lvn



def compute_delta_profile(df: pd.DataFrame, bin_step: float = 2.0) -> dict:
    """Signed futures volume at index-mapped price. + close>=open, - otherwise."""
    empty = {"ok": False}
    if df is None or df.empty or "volume" not in df.columns:
        return empty
    lo = float(min(df["low"].min(), df["close"].min()))
    hi = float(max(df["high"].max(), df["close"].max()))
    if hi <= lo:
        hi = lo + bin_step
    bin_step = float(bin_step) if bin_step and bin_step > 0 else 2.0
    lo = np.floor(lo / bin_step) * bin_step
    hi = np.ceil(hi / bin_step) * bin_step
    edges = np.arange(lo, hi + bin_step * 0.5, bin_step)
    if len(edges) < 3:
        return empty
    n_bins = len(edges) - 1
    dlt = np.zeros(n_bins, dtype=float)
    has_open = "open" in df.columns
    for _, r in df.iterrows():
        v = float(r.get("volume") or 0)
        if v <= 0:
            continue
        signed = v if (not has_open or float(r["close"]) >= float(r["open"])) else -v
        l = float(r["low"]); h = float(r["high"])
        if h <= l:
            h = l + 1e-6
        span = h - l
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            overlap = max(0.0, min(h, b) - max(l, a))
            if overlap > 0:
                dlt[i] += signed * (overlap / span)
    mids = (edges[:-1] + edges[1:]) / 2.0
    return {"ok": True, "mids": mids, "delta": dlt}

def compute_session_volume_profile(df: pd.DataFrame, bin_step: float = 5.0, prominence_factor: float = 0.35) -> dict:
    """Volume-at-price with tick-size bins, POC, ±1σ/±1.5σ VA, prominence HVN/LVN."""
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
    hvn, lvn = _prominence_nodes(vol_at, mids, prominence_factor=prominence_factor)

    def _expand(seed_i, frac, mask=None):
        w = vol_at * mask if mask is not None else vol_at
        t = float(w.sum())
        if t <= 0:
            return None
        tgt = frac * t
        lo_i = hi_i = int(np.clip(seed_i, 0, n_bins - 1))
        acc = float(w[lo_i])
        guard = 0
        while acc < tgt and guard < n_bins + 2:
            guard += 1
            left = float(w[lo_i - 1]) if lo_i > 0 else -1.0
            right = float(w[hi_i + 1]) if hi_i < n_bins - 1 else -1.0
            if right < 0 and left < 0:
                break
            if right >= left:
                hi_i = min(hi_i + 1, n_bins - 1)
                acc += float(w[hi_i])
            else:
                lo_i = max(lo_i - 1, 0)
                acc += float(w[lo_i])
        return {
            "vah": float(mids[hi_i]),
            "val": float(mids[lo_i]),
            "poc": float(mids[int(np.clip(seed_i, 0, n_bins - 1))]),
            "frac": frac,
        }

    va70 = _expand(poc_i, 0.70)
    va80 = _expand(poc_i, 0.80)
    vas = []
    if va70:
        vas.append({**va70, "tag": "VA70", "primary": True})
    if va80:
        vas.append({**va80, "tag": "VA80", "primary": False})

    # Local value areas around secondary HVNs (basins split by LVNs / midpoints)
    hvn_idx = []
    for px in hvn:
        j = int(np.argmin(np.abs(mids - float(px))))
        if j not in hvn_idx:
            hvn_idx.append(j)
    hvn_idx = sorted(hvn_idx)
    lvn_idx = sorted({int(np.argmin(np.abs(mids - float(px)))) for px in (lvn or [])})
    poc_vol = float(vol_at[poc_i]) or 1.0
    node_n = 1
    for hi in hvn_idx:
        if abs(hi - poc_i) < max(3, int(8 / max(bin_step, 1.0))):
            continue
        if float(vol_at[hi]) < 0.38 * poc_vol:
            continue
        left_cut = 0
        right_cut = n_bins - 1
        lefts = [x for x in lvn_idx if x < hi]
        rights = [x for x in lvn_idx if x > hi]
        others = [x for x in hvn_idx if x != hi]
        if lefts:
            left_cut = max(left_cut, lefts[-1])
        elif others:
            lo_o = [x for x in others if x < hi]
            if lo_o:
                left_cut = max(left_cut, (lo_o[-1] + hi) // 2)
        if rights:
            right_cut = min(right_cut, rights[0])
        elif others:
            hi_o = [x for x in others if x > hi]
            if hi_o:
                right_cut = min(right_cut, (hi_o[0] + hi) // 2)
        mask = np.zeros(n_bins, dtype=float)
        mask[left_cut:right_cut + 1] = 1.0
        loc = _expand(hi, 0.70, mask=mask)
        if not loc:
            continue
        node_n += 1
        loc["tag"] = f"VA{node_n}"
        loc["primary"] = False
        # drop if almost identical to primary
        if va70 and abs(loc["vah"] - va70["vah"]) < bin_step * 2 and abs(loc["val"] - va70["val"]) < bin_step * 2:
            continue
        vas.append(loc)

    vah = float(va70["vah"]) if va70 else float(mids[min(n_bins - 1, poc_i)])
    val = float(va70["val"]) if va70 else float(mids[max(0, poc_i)])
    nodes = _cluster_value_areas(vol_at, mids, bin_step=bin_step, max_nodes=5)
    # Prefer cluster body that contains session POC as the drawn primary VA
    for nd in nodes:
        if nd.get("primary"):
            # keep classic 70% as model VAH/VAL; drawing uses node body
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
        "vah": vah,
        "val": val,
        "vah80": float(va80["vah"]) if va80 else vah,
        "val80": float(va80["val"]) if va80 else val,
        "vas": vas,
        "nodes": nodes,
        "hvn": hvn,
        "lvn": lvn,
        "bin_step": bin_step,
    }


def _cluster_value_areas(vol_at, mids, bin_step=2.0, max_nodes=5):
    """Practical VA nodes: body of each significant volume peak, not σ-bands."""
    vol_at = np.asarray(vol_at, dtype=float)
    mids = np.asarray(mids, dtype=float)
    n = len(vol_at)
    if n < 5:
        return []
    # 3-bin smooth for peak finding only
    sm = vol_at.copy()
    if n >= 3:
        sm[1:-1] = 0.25 * vol_at[:-2] + 0.50 * vol_at[1:-1] + 0.25 * vol_at[2:]
    poc_i = int(np.argmax(vol_at))
    poc_vol = float(vol_at[poc_i]) or 1.0
    min_sep = max(4, int(round(12.0 / max(float(bin_step), 1.0))))
    peaks = []
    for i in range(1, n - 1):
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and vol_at[i] >= 0.22 * poc_vol:
            peaks.append(i)
    # keep highest in each min_sep window
    peaks = sorted(peaks, key=lambda i: -vol_at[i])
    kept = []
    for i in peaks:
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
        if len(kept) >= max_nodes:
            break
    if poc_i not in kept:
        kept = [poc_i] + [i for i in kept if abs(i - poc_i) >= min_sep]
        kept = kept[:max_nodes]
    nodes = []
    for i in kept:
        peak_v = float(vol_at[i])
        floor = max(0.42 * peak_v, 0.10 * poc_vol)
        lo = hi = i
        while lo > 0 and vol_at[lo - 1] >= floor:
            lo -= 1
        while hi < n - 1 and vol_at[hi + 1] >= floor:
            hi += 1
        # snap to first clear trough beyond the body
        while lo > 0 and vol_at[lo] > vol_at[lo - 1] and vol_at[lo - 1] >= 0.55 * floor:
            lo -= 1
        while hi < n - 1 and vol_at[hi] > vol_at[hi + 1] and vol_at[hi + 1] >= 0.55 * floor:
            hi += 1
        nodes.append({
            "i": i, "poc": float(mids[i]), "vah": float(mids[hi]), "val": float(mids[lo]),
            "vol": peak_v, "primary": i == poc_i,
        })
    # merge heavy overlap
    nodes = sorted(nodes, key=lambda r: -r["vol"])
    merged = []
    for nd in nodes:
        hit = None
        for m in merged:
            ov = min(nd["vah"], m["vah"]) - max(nd["val"], m["val"])
            span = max(nd["vah"] - nd["val"], m["vah"] - m["val"], 1.0)
            if ov > 0.55 * span:
                hit = m
                break
        if hit:
            if nd["vol"] > hit["vol"]:
                hit.update({k: nd[k] for k in ("poc", "vah", "val", "vol", "i")})
            hit["primary"] = hit["primary"] or nd["primary"]
        else:
            merged.append(dict(nd))
    merged = sorted(merged, key=lambda r: -r["poc"])
    return merged[:max_nodes]


def vp_chart_levels(vp: dict) -> list:
    """Only practical node VAH/VAL + session POC. No σ / 80% / HVN scatter."""
    out = []
    if not isinstance(vp, dict) or not vp.get("ok"):
        return out
    step = float(vp.get("bin_step") or 2.0)
    nodes = list(vp.get("nodes") or [])
    if not nodes:
        nodes = [{
            "vah": vp.get("vah"), "val": vp.get("val"), "poc": vp.get("poc"),
            "primary": True, "vol": vp.get("poc_vol") or 0,
        }]
    specs = [(vp.get("poc"), "POC", "#FFD54F", 1.7, "solid")]
    prim = next((n for n in nodes if n.get("primary")), nodes[0] if nodes else None)
    others = [n for n in nodes if n is not prim]
    # rank other nodes high-to-low price
    others = sorted(others, key=lambda n: -float(n.get("poc") or 0))
    if prim:
        specs.append((prim.get("vah"), "VAH", "#F48FB1", 1.55, "dash"))
        specs.append((prim.get("val"), "VAL", "#F48FB1", 1.55, "dash"))
    for k, n in enumerate(others, start=2):
        specs.append((n.get("vah"), f"VAH{k}", "#F8BBD0", 1.2, "dash"))
        specs.append((n.get("val"), f"VAL{k}", "#F8BBD0", 1.2, "dash"))
    used = []
    for px, name, col, w, dash in specs:
        if px is None:
            continue
        try:
            px = float(px)
        except Exception:
            continue
        if any(abs(px - u) < max(step * 1.5, 3.0) for u, _ in used):
            continue
        used.append((px, name))
        out.append({"price": px, "name": name, "color": col, "width": w, "dash": dash})
    out.sort(key=lambda r: -r["price"])
    return out


def book_change_sigmas(hist: list, min_n: int = 8):
    """Std of bid/ask lot changes from the live tape. Returns (bid_sigma, ask_sigma)."""
    bids = [float(h.get("bid_change") or 0) for h in hist]
    asks = [float(h.get("ask_change") or 0) for h in hist]
    if len(bids) < min_n:
        return None, None
    bid_s = float(np.std(bids, ddof=1)) if len(bids) > 1 else 0.0
    ask_s = float(np.std(asks, ddof=1)) if len(asks) > 1 else 0.0
    bid_s = max(bid_s, 1.0)
    ask_s = max(ask_s, 1.0)
    return bid_s, ask_s


def classify_liq_alert(bid_change: float, ask_change: float, bid_sigma=None, ask_sigma=None, k=1.5) -> dict:
    """Flag pull/stack when |Δ| exceeds k × rolling σ of that side's book changes."""
    if bid_sigma is None or ask_sigma is None:
        return {"code": "NONE", "label": "NO SIGNAL", "color": "#888888",
                "hint": "Need more snapshots to estimate σ. Keep Auto-Refresh on.",
                "bid_thr": None, "ask_thr": None, "k": k}
    bid_thr = k * float(bid_sigma)
    ask_thr = k * float(ask_sigma)
    if bid_change <= -bid_thr:
        return {"code": "BIDS_PULLED", "label": "BIDS PULLED", "color": "#FF5252",
                "hint": f"Bid Δ {bid_change:+.0f} ≤ −{k:g}σ ({-bid_thr:.0f} lots). Liquidity vacuum.",
                "bid_thr": bid_thr, "ask_thr": ask_thr, "k": k}
    if ask_change <= -ask_thr:
        return {"code": "ASKS_PULLED", "label": "ASKS PULLED", "color": "#00E676",
                "hint": f"Ask Δ {ask_change:+.0f} ≤ −{k:g}σ ({-ask_thr:.0f} lots). Offers lifted.",
                "bid_thr": bid_thr, "ask_thr": ask_thr, "k": k}
    if bid_change >= bid_thr:
        return {"code": "BIDS_STACKED", "label": "BIDS STACKED", "color": "#2196F3",
                "hint": f"Bid Δ {bid_change:+.0f} ≥ +{k:g}σ ({bid_thr:.0f} lots). Passive absorption.",
                "bid_thr": bid_thr, "ask_thr": ask_thr, "k": k}
    return {"code": "NONE", "label": "NO SIGNAL", "color": "#888888",
            "hint": f"Inside ±{k:g}σ  (bid {bid_thr:.0f} / ask {ask_thr:.0f} lots).",
            "bid_thr": bid_thr, "ask_thr": ask_thr, "k": k}



def render_block_tape(data: dict, index_name: str):
    """Credible tape: only Δsession-volume between two chain snapshots.

    Not exchange TBT. Side is inferred only when LTP is outside the same-snap bid/ask.
    Cumulative volume is never shown as 'lots'.
    """
    heading_ribbon(
        "📜 Block tape (Δ volume)",
        "REST chain differencing — <b>not</b> NSE tick tape.<br>"
        "Δlots = (Vol_now − Vol_prev) / lot. Ignore ΔVol ≤ 0.<br>"
        "Side only if LTP ≥ ask (LIFT) or LTP ≤ bid (HIT) on <b>that</b> snapshot; else —.<br>"
        "Premium = Δlots × lot × LTP. Time = snapshot clock, not exchange match time.<br>"
        "Cutoff = max(8 lots, median Δlots, mean+1.5σ of positive Δlots). Cap 40 rows.",
    )
    chain = data.get("chain_results") or []
    lot = max(int(LOT_SIZES.get(index_name, 65)), 1)
    ts = str(data.get("timestamp") or "")
    live, today, _, _ = market_session_state()
    sess_day = today
    df_fut = data.get("df_futures")
    if isinstance(df_fut, pd.DataFrame) and not df_fut.empty and "time" in df_fut.columns:
        try:
            sess_day = pd.to_datetime(series_to_ist(df_fut["time"]).iloc[-1]).date()
        except Exception:
            pass
    prev_key = f"chain_vol_{index_name}_{sess_day}"
    tape_key = f"block_events_{index_name}_{sess_day}"
    prev = dict(st.session_state.get(prev_key) or {})
    events = list(st.session_state.get(tape_key) or [])
    now_map = {}
    new_deltas = []
    for r in chain:
        K = int(r.get("Strike") or 0)
        for typ, vol_k, ltp_k, d_k, bid_k, ask_k in (
            ("CE", "C_Vol", "C_LTP", "C_Δ", "C_Bid", "C_Ask"),
            ("PE", "P_Vol", "P_LTP", "P_Δ", "P_Bid", "P_Ask"),
        ):
            key = f"{K}{typ}"
            vol = float(r.get(vol_k) or 0)
            now_map[key] = vol
            if not live or sess_day != today:
                continue
            if key not in prev:
                continue
            dvol = vol - float(prev[key])
            if dvol <= 0:
                continue
            dlots = dvol / lot
            ltp = float(r.get(ltp_k) or 0)
            bid = float(r.get(bid_k) or 0)
            ask = float(r.get(ask_k) or 0)
            side = "—"
            if ltp > 0 and ask > 0 and ltp >= ask:
                side = "LIFT"
            elif ltp > 0 and bid > 0 and ltp <= bid:
                side = "HIT"
            new_deltas.append(dlots)
            events.append({
                "Time": ts[-12:] if ts else "",
                "Contract": f"{K} {typ}",
                "Side": side,
                "ΔLots": round(dlots, 1),
                "LTP": round(ltp, 2),
                "Δ": round(float(r.get(d_k) or 0), 2),
                "Prem ₹L": round(dlots * lot * ltp / 1e5, 2),
            })
    st.session_state[prev_key] = now_map
    if new_deltas:
        s = pd.Series(new_deltas, dtype=float)
        thr = max(8.0, float(s.median()), float(s.mean() + 1.5 * (s.std(ddof=0) or 0)))
    else:
        thr = 8.0
    events = [e for e in events if float(e.get("ΔLots") or 0) >= thr]
    events = events[-40:]
    st.session_state[tape_key] = events
    if not live:
        st.caption(f"Replay {sess_day} · {len(events)} stored Δ prints (no new diffs off-hours).")
    elif not prev:
        st.caption("Baseline stored. Next refresh will emit Δlots. Keep Auto-Refresh on.")
        return
    if not events:
        st.caption(f"No Δlots ≥ {thr:.0f} since baseline.")
        return
    df = pd.DataFrame(events[::-1])
    st.dataframe(df, use_container_width=True, hide_index=True, height=220)


def render_liquidity_delta_panel(data: dict, df_fchart: pd.DataFrame, index_name: str, compact: bool = False):
    """Middle panel: futures limit-book liquidity delta + imbalance alerts."""
    basis = data.get("basis_info") or {}
    fut_token = basis.get("fut_token")
    smart_api = get_smart_api_client()
    snap = fetch_futures_book_snapshot(smart_api, index_name, fut_token) if smart_api else {"ok": False}
    if snap.get("ok"):
        _remember_book5(snap)
    hist = update_liq_delta_history(snap, index_name) if snap.get("ok") else list(st.session_state.get("liq_delta_history") or [])
    hist = [h for h in hist if h.get("index", index_name) == index_name]

    if not hist:
        h1, h2 = st.columns([0.78, 0.22])
        with h1:
            heading_ribbon(
                "📘 Limit Book Liquidity Δ",
                "<b>Resting book size only</b> — not executed prints. ΔBid=Bid_t-Bid_t-1.",
            )
            st.caption("NO SIGNAL — need snapshots. Enable Auto-Refresh.")
        with h2:
            st.session_state["liq_sigma_k"] = st.selectbox(
                "σ flag", options=[1.5, 2.0], index=0,
                format_func=lambda x: f"±{x}σ", key="liq_sigma_select",
                label_visibility="collapsed",
            )
        return

    last = hist[-1]
    liq_k = float(st.session_state.get("liq_sigma_k", 1.5))
    bid_s, ask_s = book_change_sigmas(hist)
    alert = classify_liq_alert(
        float(last.get("bid_change") or 0),
        float(last.get("ask_change") or 0),
        bid_s, ask_s, k=liq_k,
    )
    h1, h2, h3, h4, h5 = st.columns([0.26, 0.24, 0.16, 0.20, 0.14])
    with h1:
        heading_ribbon(
            "📘 Limit Book Liquidity Δ",
            "<b>Resting book size only</b> — not executed prints.<br>"
            "ΔBid = BidQty_t − BidQty_t-1 · Net = ΔBid − ΔAsk<br>"
            "<b>Imbalance</b> = (BidLots − AskLots) / (BidLots + AskLots)<br>"
            "Range −1…+1. + → bid-heavy (support). − → ask-heavy (supply).<br>"
            "A size drop can be a pull or a hit; the feed does not tag which.",
        )
    with h2:
        bq = float(last.get('bid_qty_lots') or 0)
        aq = float(last.get('ask_qty_lots') or 0)
        imb = (bq - aq) / (bq + aq) if (bq + aq) else 0.0
        st.caption(
            f"Book Bid {bq:,.0f} / Ask {aq:,.0f} · Imb {imb:+.2f}"
        )
    with h3:
        st.caption(f"ΔBid {last.get('bid_change', 0):+.0f} · ΔAsk {last.get('ask_change', 0):+.0f}")
    with h4:
        st.markdown(
            f"<div style='border:1px solid {alert['color']};border-radius:6px;padding:3px 8px;'>"
            f"<span style='color:{alert['color']};font-weight:800;font-size:12px;'>{alert['label']}</span></div>",
            unsafe_allow_html=True
        )
    with h5:
        st.session_state["liq_sigma_k"] = st.selectbox(
            "σ flag", options=[1.5, 2.0], index=0,
            format_func=lambda x: f"±{x}σ", key="liq_sigma_select",
            label_visibility="collapsed",
        )

    times = [h["ts"].strftime("%H:%M:%S") if hasattr(h["ts"], "strftime") else str(h["ts"]) for h in hist]
    nets = [float(h.get("net_liq") or 0) for h in hist]
    bid_d = [float(h.get("bid_change") or 0) for h in hist]
    ask_d = [float(h.get("ask_change") or 0) for h in hist]
    bar_colors = ["#00E676" if v >= 0 else "#FF5252" for v in nets]

    fig = plt_go.Figure()
    fig.add_trace(plt_go.Bar(x=times, y=nets, name="Net Δ", marker_color=bar_colors, opacity=0.7, showlegend=True))
    fig.add_trace(plt_go.Scatter(x=times, y=bid_d, name="Bid Δ", line=dict(color="#2196F3", width=1.6), showlegend=True))
    fig.add_trace(plt_go.Scatter(x=times, y=ask_d, name="Ask Δ", line=dict(color="#FF9800", width=1.6), showlegend=True))
    fig.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot")
    if alert.get("bid_thr"):
        fig.add_hline(y=alert["bid_thr"], line_width=1, line_color="#2196F3", line_dash="dash")
        fig.add_hline(y=-alert["bid_thr"], line_width=1, line_color="#FF5252", line_dash="dash")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=170 if compact else 240, margin=dict(l=4, r=4, t=4, b=28),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.0, xanchor="left",
                    font=dict(size=9), bgcolor="rgba(14,17,23,0.85)"),
        hovermode="x unified", showlegend=True,
        title=None,
    )
    fig.update_xaxes(type="category", nticks=5, tickfont=dict(size=8))
    fig.update_yaxes(title="", tickfont=dict(size=9))
    st.plotly_chart(fig, use_container_width=True)
    if bid_s and ask_s:
        st.caption(f"Flag at ±{liq_k:g}σ · bid σ={bid_s:.0f} ask σ={ask_s:.0f} lots")
    else:
        st.caption("σ estimated after ~8 snapshots. Keep Auto-Refresh on.")


def render_live_alert_ribbon(data: dict = None):
    """Compact layout: left = Basket Greeks | right = Z-Scores. (Live alerts removed)"""
    st.markdown("---")
    left_blk, right_blk = st.columns([0.40, 0.60])

    with left_blk:
        if data is not None:
            render_basket_metrics_block(data)
        else:
            st.caption("Basket Greeks – no data")

    with right_blk:
        zscore_analysis_fragment(mode="highlights")



# --- LIVE DASHBOARD FRAGMENT ---

def refresh_index_tapes(data, want_tf):
    """5s path: only spot LTP + index/futures candles. Keep last chain/GEX."""
    if not data:
        return data
    try:
        smart_api = get_smart_api_client()
        if not smart_api:
            return data
        api_interval, lookback_days = interval_mapping.get(want_tf, ("THREE_MINUTE", 10))
        live_now, _, _, _ = market_session_state(_ist_now(), Index_Name)
        # 5s path: today only + merge into frames already in data_store
        lookback_days = 0 if live_now else min(int(lookback_days or 5), 2)
        spot_token, spot_exch, opt_exch = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))
        fut_tok, _ = get_near_month_futures_token(df_master, Index_Name, opt_exch)
        if not spot_token and fut_tok:
            spot_token = fut_tok
        ltp_sym = Index_Name
        try:
            if fut_tok is not None:
                hit = df_master[df_master["token"].astype(str) == str(fut_tok)]
                if not hit.empty and "symbol" in hit.columns:
                    ltp_sym = str(hit.iloc[0]["symbol"])
        except Exception:
            pass
        if spot_token:
            spot_resp = safe_api_call(smart_api.ltpData, exchange=spot_exch, tradingsymbol=ltp_sym, symboltoken=spot_token)
            if spot_resp and spot_resp.get("status") and spot_resp.get("data"):
                data["spot_price"] = float(spot_resp["data"]["ltp"])
        have_sess = data.get("df_candles")
        tail = 25 if (have_sess is not None and not getattr(have_sess, "empty", True) and len(have_sess) >= 8) else None
        df_candles, is_fb = fetch_candles_with_holiday_fallback(
            smart_api, spot_token, spot_exch, api_interval, lookback_days, Index_Name,
            tail_minutes=tail,
        )
        if df_candles is not None and not df_candles.empty:
            data["df_candles"] = merge_candle_frames(data.get("df_candles"), df_candles)
            data["is_holiday_fallback"] = is_fb
        have_f = data.get("df_futures")
        tail_f = 25 if (have_f is not None and not getattr(have_f, "empty", True) and len(have_f) >= 8) else None
        df_futures, fut_fb, basis_info, fut_msg = fetch_futures_candles_with_vwap(
            smart_api, Index_Name, df_master, api_interval, lookback_days, tail_minutes=tail_f,
        )
        if df_futures is not None and not df_futures.empty:
            data["df_futures"] = merge_candle_frames(data.get("df_futures"), df_futures)
            data["fut_is_fallback"] = fut_fb
            data["basis_info"] = basis_info
            data["fut_fallback_msg"] = fut_msg
        if not data.get("spot_price"):
            src = data.get("df_futures")
            if src is None or getattr(src, "empty", True):
                src = data.get("df_candles")
            if src is not None and not getattr(src, "empty", True):
                data["spot_price"] = float(src["close"].iloc[-1])
        data["bar_tf"] = want_tf
        data["timestamp"] = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%d-%b-%Y %H:%M:%S IST")
        try:
            tok = (data.get("basis_info") or {}).get("fut_token")
            if tok:
                sn = fetch_futures_book_snapshot(smart_api, Index_Name, tok)
                _remember_book5(sn)
        except Exception:
            pass
    except Exception:
        pass
    return data


def _multi_option_frame(index_name):
    _, _, exch = INDEX_TOKEN_MAP.get(index_name, ("99926000", "NSE", "NFO"))
    if index_name in MCX_NAME_ALIASES:
        aliases = [a.upper() for a in MCX_NAME_ALIASES[index_name]]
        df_opt = df_master[
            (df_master["exch_seg"].isin(["MCX", "NCO"]))
            & (df_master["instrumenttype"].isin(["OPTFUT", "OPTCOM"]))
            & (df_master["name"].astype(str).str.upper().isin(aliases + [index_name]))
        ].copy()
    else:
        df_opt = df_master[
            (df_master["exch_seg"] == exch)
            & (df_master["name"] == index_name)
            & (df_master["instrumenttype"].isin(["OPTIDX", "OPTSTK", "OPTFUT"]))
        ].copy()
    if df_opt.empty:
        return df_opt, exch, None
    df_opt["expiry_dt"] = pd.to_datetime(df_opt["expiry"], format="%d%b%Y", errors="coerce")
    today_dt = pd.to_datetime(datetime.datetime.now(pytz.timezone("Asia/Kolkata")).date())
    valid = sorted(df_opt[df_opt["expiry_dt"] >= today_dt]["expiry_dt"].dropna().unique())
    exp = valid[0] if len(valid) else None
    return df_opt, exch, exp


def _multi_prepare_session(df_fut, df_spot, spot_px, want_tf, index_name=None):
    if df_fut is None or getattr(df_fut, "empty", True) or "time" not in df_fut.columns:
        return pd.DataFrame(), None
    df_fut = attach_bar_flow(df_fut.copy())
    tf_min = {"3 min": 3, "5 min": 5, "15 min": 15}.get(want_tf, 5)
    try:
        t = series_to_ist(df_fut["time"])
        dt = t.diff().dt.total_seconds().median()
        native = float(dt) / 60.0 if pd.notna(dt) and dt else tf_min
        if native > 0 and tf_min > native + 0.6:
            g = df_fut.copy()
            g["time"] = t
            g = g.set_index("time").sort_index()
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ohlc = g.resample(f"{int(tf_min)}min", label="right", closed="right").agg(agg).dropna(subset=["close"])
            if len(ohlc) >= 5:
                df_fut = ohlc.reset_index()
    except Exception:
        pass
    dfc, latest, _ = pick_last_nse_session(df_fut, min_bars=12, prefer_today=True, index_name=index_name)
    if dfc is None or getattr(dfc, "empty", True):
        df_fut = df_fut.copy()
        df_fut["time"] = series_to_ist(df_fut["time"])
        df_fut["session_date"] = df_fut["time"].dt.date
        if df_fut["session_date"].notna().any():
            latest = sorted(df_fut["session_date"].dropna().unique())[-1]
            dfc = df_fut[df_fut["session_date"] == latest].copy().reset_index(drop=True)
        else:
            return pd.DataFrame(), None
    dfc = dfc.copy().reset_index(drop=True)
    dfc["time"] = series_to_ist(dfc["time"])
    dfc["time_str"] = dfc["time"].dt.strftime("%H:%M")
    dfc["tp"] = (dfc["high"] + dfc["low"] + dfc["close"]) / 3.0
    cum_vol = cum_tp = cum_sq = 0.0
    vwaps, stds = [], []
    for i in range(len(dfc)):
        vol = float(dfc.loc[i, "volume"]) if "volume" in dfc.columns else 1.0
        tp = float(dfc.loc[i, "tp"])
        cum_vol += vol
        cum_tp += tp * vol
        vwap = cum_tp / cum_vol if cum_vol > 0 else tp
        cum_sq += vol * (tp - vwap) ** 2
        vwaps.append(vwap)
        stds.append(math.sqrt(max(cum_sq / cum_vol if cum_vol > 0 else 0.0, 0.0)))
    dfc["vwap"] = vwaps
    dfc["vwap_upper"] = [v + 1.5 * s for v, s in zip(vwaps, stds)]
    dfc["vwap_lower"] = [v - 1.5 * s for v, s in zip(vwaps, stds)]
    dfc["spot_px"] = np.nan
    if df_spot is not None and not getattr(df_spot, "empty", True) and "close" in df_spot.columns:
        try:
            sp = df_spot.copy()
            sp["time"] = series_to_ist(sp["time"])
            sp = sp.dropna(subset=["time"]).sort_values("time")
            if latest is not None:
                sp = sp[sp["time"].dt.date == latest]
            mapped = pd.merge_asof(
                dfc[["time"]].sort_values("time"),
                sp[["time", "close"]].rename(columns={"close": "spot_px"}).sort_values("time"),
                on="time", direction="nearest", tolerance=pd.Timedelta("6min"),
            )
            dfc["spot_px"] = pd.to_numeric(mapped["spot_px"], errors="coerce").values
        except Exception:
            pass
    if int(pd.to_numeric(dfc["spot_px"], errors="coerce").notna().sum()) < max(4, len(dfc) // 4):
        last_f = float(dfc["close"].iloc[-1])
        basis_q = last_f - float(spot_px or 0) if spot_px else 0.0
        dfc["spot_px"] = dfc["close"].astype(float) - basis_q
    else:
        dfc["spot_px"] = pd.to_numeric(dfc["spot_px"], errors="coerce").ffill().bfill()
    dfc["basis"] = (dfc["close"].astype(float) - dfc["spot_px"].astype(float)).ewm(span=5, min_periods=1, adjust=False).mean()
    dfc["vwap_idx"] = dfc["vwap"].astype(float) - dfc["basis"]
    dfc["vwap_upper_idx"] = dfc["vwap_upper"].astype(float) - dfc["basis"]
    dfc["vwap_lower_idx"] = dfc["vwap_lower"].astype(float) - dfc["basis"]
    return dfc, latest


def _multi_va_labels(dfi, key):
    out = []
    if dfi is None or dfi.empty or len(dfi) < 16:
        return out
    n = len(dfi)
    store_key = f"_multi_va_{key}"
    prev = list(st.session_state.get(store_key) or [])
    start = max(15, n - 48)
    if prev and prev[-1].get("n", 0) <= n:
        start = max(start, int(prev[-1].get("i", start)))
        out = [r for r in prev if r.get("i", 0) < n]
    last_act = out[-1]["act"] if out else ""
    for i in range(start, n):
        sl = dfi.iloc[: i + 1]
        try:
            rec = classify_microstructure(sl)
        except Exception:
            continue
        act = str(rec.get("action") or "")
        up = act.upper()
        if (not act) or ("NO ENTRY" in up) or ("INSIDE" in up) or ("CHOP" in up):
            continue
        short = _va_short_label(act)
        if not short or short == last_act:
            continue
        tlab = sl["time_str"].iloc[-1] if "time_str" in sl.columns else str(i)
        out.append({"t": tlab, "act": short, "i": i, "n": n})
        last_act = short
    st.session_state[store_key] = out[-16:]
    return st.session_state[store_key]


def _va_short_label(act: str) -> str:
    s = str(act or "").upper()
    if "WATCH SHORT" in s:
        return "WATCH-S"
    if "WATCH LONG" in s:
        return "WATCH-L"
    if "ADD SHORT" in s:
        return "ADD-S"
    if "ADD LONG" in s:
        return "ADD-L"
    if "SHORT" in s and ("BREAK" in s or s.startswith("SHORT B")):
        return "SHORT-B"
    if "LONG" in s and ("BREAK" in s or s.startswith("LONG B")):
        return "LONG-B"
    if "SHORT" in s:
        return "SHORT-ME"
    if "LONG" in s:
        return "LONG-ME"
    return str(act or "").strip()[:12]


def build_multi_index_figure(index_name, dfi, vp):
    if dfi is None or dfi.empty:
        return None
    axis_times = session_axis_labels(st.session_state.get("multi_tf"), index_name)
    if not axis_times:
        axis_times = list(dfi["time_str"])
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.82, 0.18],
        shared_yaxes=True, horizontal_spacing=0.012,
        specs=[[{}, {}]],
    )
    idx_o = dfi["spot_px"].astype(float) + (dfi["open"].astype(float) - dfi["close"].astype(float))
    idx_h = dfi["spot_px"].astype(float) + (dfi["high"].astype(float) - dfi["close"].astype(float))
    idx_l = dfi["spot_px"].astype(float) + (dfi["low"].astype(float) - dfi["close"].astype(float))
    idx_c = dfi["spot_px"].astype(float)
    if "vwap_upper_idx" in dfi.columns:
        fig.add_trace(plt_go.Scatter(
            x=dfi["time_str"], y=dfi["vwap_upper_idx"], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(plt_go.Scatter(
            x=dfi["time_str"], y=dfi["vwap_lower_idx"], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,152,0,0.08)"), row=1, col=1)
    fig.add_trace(plt_go.Scatter(
        x=dfi["time_str"], y=dfi["vwap_idx"], mode="lines", name="VWAP",
        line=dict(color="#FF9800", width=2), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(plt_go.Candlestick(
        x=dfi["time_str"], open=idx_o, high=idx_h, low=idx_l, close=idx_c,
        name=str(index_name), increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
        increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350", showlegend=False,
    ), row=1, col=1)
    candle_hi = float(np.nanmax(idx_h))
    candle_lo = float(np.nanmin(idx_l))
    smin = float(np.nanmin([candle_lo, dfi["vwap_lower_idx"].min() if "vwap_lower_idx" in dfi.columns else candle_lo]))
    smax = float(np.nanmax([candle_hi, dfi["vwap_upper_idx"].max() if "vwap_upper_idx" in dfi.columns else candle_hi]))
    pad = (smax - smin) * 0.04 if smax > smin else 12
    y0, y1 = smin - pad, max(smax, candle_hi) + max(pad * 0.8, (smax - smin) * 0.06 if smax > smin else 8)
    last_basis = float(dfi["basis"].iloc[-1]) if "basis" in dfi.columns else 0.0
    vp2 = dict(vp) if isinstance(vp, dict) else {"ok": False}
    if vp2.get("ok"):
        try:
            if vp2.get("mids") is not None:
                vp2["mids"] = [float(m) - last_basis for m in vp2["mids"]]
            for k in ("poc", "vah", "val", "vah1", "val1", "vah15", "val15"):
                if vp2.get(k) is not None and not isinstance(vp2.get(k), (list, np.ndarray)):
                    vp2[k] = float(vp2[k]) - last_basis
            nodes = []
            for n in (vp2.get("nodes") or []):
                nn = dict(n)
                for kk in ("poc", "vah", "val"):
                    if nn.get(kk) is not None:
                        nn[kk] = float(nn[kk]) - last_basis
                nodes.append(nn)
            if nodes:
                vp2["nodes"] = nodes
        except Exception:
            vp2 = dict(vp)
        def _as_list(val):
            if val is None:
                return []
            if isinstance(val, np.ndarray):
                return val.tolist()
            try:
                return list(val)
            except TypeError:
                return []
        raw_mids = _as_list(vp2.get("mids"))
        raw_vols = _as_list(vp2.get("vol") if vp2.get("vol") is not None else vp2.get("vols"))
        mids, vols, colors = [], [], []
        poc = vp2.get("poc")
        for m, v in zip(raw_mids, raw_vols):
            if not (y0 <= float(m) <= (smax + pad)):
                continue
            mids.append(float(m))
            vols.append(float(v))
            colors.append("#FFD54F" if poc is not None and abs(float(m) - float(poc)) < 1e-6 else "rgba(100,181,246,0.75)")
        if mids and vols:
            fig.add_trace(plt_go.Bar(
                x=vols, y=mids, orientation="h", showlegend=False, name="VP",
                marker=dict(color=colors),
                hovertemplate="Px %{y:.0f}<br>Vol %{x:.0f}<extra>VP</extra>",
            ), row=1, col=2)
        x_lab = axis_times[-1] if axis_times else dfi["time_str"].iloc[-1]
        for lv in vp_chart_levels(vp2):
            yv = float(lv["price"])
            if not (smin - pad <= yv <= smax + pad):
                continue
            fig.add_hline(y=yv, line_color=lv["color"], line_width=lv["width"], line_dash=lv["dash"], row=1, col=1)
            fig.add_annotation(
                x=x_lab, y=yv, text=f"{lv['name']} {yv:.0f}",
                showarrow=False, xanchor="right",
                font=dict(size=9, color=lv["color"]),
                bgcolor="rgba(14,17,23,0.45)", row=1, col=1,
            )
    labels = _multi_va_labels(dfi, index_name)
    y_lab = float(candle_hi) + max((y1 - y0) * 0.012, 1.0)
    for rec in labels[-12:]:
        fig.add_annotation(
            x=str(rec.get("t", "")),
            y=y_lab,
            text=str(rec.get("act", "")),
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=12, color="#FFE082"),
            bgcolor="rgba(8,10,16,0.88)",
            bordercolor="#FFE082",
            borderwidth=1,
            row=1,
            col=1,
        )
    xr = None
    try:
        if axis_times and len(dfi):
            last = str(dfi["time_str"].iloc[-1])
            if last in axis_times:
                i = axis_times.index(last)
                xr = [max(0, i - 80), min(len(axis_times) - 1, i + 2)]
    except Exception:
        xr = None
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=400, margin=dict(l=36, r=6, t=52, b=22),
        xaxis_rangeslider_visible=False, showlegend=False, hovermode="x unified",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                     range=xr, nticks=7, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(range=[y0, y1], tickfont=dict(size=8), row=1, col=1)
    fig.update_yaxes(range=[y0, y1], showticklabels=False, showgrid=False, row=1, col=2)
    return fig


def refresh_multi_index_tapes(want_tf):
    if st.session_state.get("app_view") != "multi":
        return st.session_state.get("multi_store") or {}
    api = get_smart_api_client()
    if not api:
        return
    enabled = [
        k for k, v in (st.session_state.get("multi_enabled") or {}).items()
        if v and not st.session_state.get(f"multi_hide_{k}")
    ]
    api_int, lb = interval_mapping.get(want_tf, ("FIVE_MINUTE", 15))
    store = dict(st.session_state.get("multi_store") or {})
    for name in enabled:
        try:
            live_now, _, _, _ = market_session_state(_ist_now(), name)
            lookback = 0 if live_now else min(int(lb or 5), 2)
            pack = dict(store.get(name) or {})
            have_f = pack.get("df_futures")
            tail_f = 25 if have_f is not None and not getattr(have_f, "empty", True) and len(have_f) >= 8 else None
            df_fut, fut_fb, basis_info, fut_msg = fetch_futures_candles_with_vwap(
                api, name, df_master, api_int, lookback, tail_minutes=tail_f,
            )
            if df_fut is not None and not df_fut.empty:
                pack["df_futures"] = merge_candle_frames(pack.get("df_futures"), df_fut)
                pack["basis_info"] = basis_info
                pack["fut_msg"] = fut_msg
                pack["fut_fb"] = fut_fb
            spot_token, spot_exch, _ = INDEX_TOKEN_MAP.get(name, ("99926000", "NSE", "NFO"))
            if not spot_token:
                tok, _ = get_near_month_futures_token(df_master, name, INDEX_TOKEN_MAP.get(name, ("", "", "NFO"))[2])
                spot_token = tok or ""
            have_s = pack.get("df_candles")
            tail_s = 25 if have_s is not None and not getattr(have_s, "empty", True) and len(have_s) >= 8 else None
            if spot_token:
                df_sp, _ = fetch_candles_with_holiday_fallback(
                    api, spot_token, spot_exch, api_int, lookback, name, tail_minutes=tail_s,
                )
                if df_sp is not None and not df_sp.empty:
                    pack["df_candles"] = merge_candle_frames(pack.get("df_candles"), df_sp)
                    pack["spot_price"] = float(df_sp["close"].iloc[-1])
            if not pack.get("spot_price"):
                src = pack.get("df_futures")
                if src is not None and not getattr(src, "empty", True):
                    pack["spot_price"] = float(src["close"].iloc[-1])
            pack["bar_tf"] = want_tf
            pack["ts"] = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")
            store[name] = pack
        except Exception:
            continue
    st.session_state["multi_store"] = store


def refresh_multi_index_gex():
    if st.session_state.get("app_view") != "multi":
        return
    api = get_smart_api_client()
    if not api:
        return
    enabled = [
        k for k, v in (st.session_state.get("multi_enabled") or {}).items()
        if v and not st.session_state.get(f"multi_hide_{k}")
    ]
    store = dict(st.session_state.get("multi_store") or {})
    r = float(rate_param) if "rate_param" in dir() else 0.07
    for name in enabled:
        try:
            pack = dict(store.get(name) or {})
            spot = float(pack.get("spot_price") or 0)
            df_opt, exch, exp = _multi_option_frame(name)
            if df_opt.empty or exp is None or spot <= 0:
                continue
            sub = df_opt[df_opt["expiry_dt"] == exp].copy()
            sub["strike_num"] = pd.to_numeric(sub["strike"], errors="coerce") / (100.0 if exch == "NFO" else 1.0)
            if sub["strike_num"].max() > 1000000:
                sub["strike_num"] = sub["strike_num"] / 100.0
            strikes = sorted(int(s) for s in sub["strike_num"].dropna().unique())
            if not strikes:
                continue
            atm = min(strikes, key=lambda x: abs(x - spot))
            ai = strikes.index(atm)
            window = strikes[max(0, ai - 6): min(len(strikes), ai + 7)]
            tokens = []
            mapping = []
            for k in window:
                c_tok = get_smartapi_token(sub, name, exp, int(k), "CE")
                p_tok = get_smartapi_token(sub, name, exp, int(k), "PE")
                if c_tok:
                    tokens.append(c_tok)
                if p_tok:
                    tokens.append(p_tok)
                mapping.append((int(k), c_tok, p_tok))
            md = {}
            for i in range(0, len(tokens), 40):
                chunk = [str(t) for t in tokens[i:i + 40] if t]
                res = safe_api_call(api.getMarketData, "FULL", {exch: chunk})
                if res and res.get("status") and res.get("data") and res["data"].get("fetched"):
                    for item in res["data"]["fetched"]:
                        md[str(item["symbolToken"])] = {
                            "ltp": float(item.get("ltp", 0.0)),
                            "oi": int(item.get("opnInterest", 0)),
                        }
                time.sleep(0.15)
            now_dt = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
            T = max((pd.Timestamp(exp).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30) - now_dt).total_seconds() / (365.0 * 24 * 3600), 1e-5)
            lot = float(LOT_SIZES.get(name, 65))
            net = 0.0
            rows = []
            for K, ct, pt in mapping:
                ci = md.get(str(ct), {"ltp": 0.0, "oi": 0})
                pi = md.get(str(pt), {"ltp": 0.0, "oi": 0})
                c_iv = VolatilityEngine.calculate_iv(ci["ltp"], spot, K, T, r, "c")
                p_iv = VolatilityEngine.calculate_iv(pi["ltp"], spot, K, T, r, "p")
                cg = VolatilityEngine.calculate_greeks(spot, K, T, r, c_iv or 0.0, "c")
                pg = VolatilityEngine.calculate_greeks(spot, K, T, r, p_iv or 0.0, "p")
                gex_scale = lot * (spot ** 2) * 0.01
                net_g = cg["gamma"] * ci["oi"] * gex_scale - pg["gamma"] * pi["oi"] * gex_scale
                net += net_g
                rows.append({"Strike": K, "Net_GEX_OI": net_g})
            pack["net_gex_oi"] = net
            pack["gex_rows"] = rows
            pack["gex_ts"] = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")
            store[name] = pack
        except Exception:
            continue
    st.session_state["multi_store"] = store
    st.session_state["multi_gex_ts"] = time.time()


def render_multi_index_mode():
    if st.session_state.get("app_view") not in ("multi",) and not st.session_state.get("multi_index_mode"):
        return
    want_tf = st.session_state.get("multi_tf", "15 min")
    auto = bool(st.session_state.get("enable_main_refresh", False))
    enabled = [k for k, v in (st.session_state.get("multi_enabled") or {}).items() if v]
    live_now, _, _, _ = market_session_state(_ist_now(), enabled[0] if enabled else "NIFTY")
    have = st.session_state.get("multi_store") or {}
    now_s = time.time()
    if run_btn or (not have) or any(k not in have for k in enabled):
        refresh_multi_index_tapes(want_tf)
        refresh_multi_index_gex()
    elif auto and live_now:
        refresh_multi_index_tapes(want_tf)
        last_g = float(st.session_state.get("multi_gex_ts") or 0)
        if (now_s - last_g) >= 300:
            refresh_multi_index_gex()
    store = st.session_state.get("multi_store") or {}

    st.markdown("<div class='sticky-summary'>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns([0.42, 0.38, 0.20])
    with h1:
        st.markdown(
            "<h1 class='custom-heading' style='margin:0;font-size:17px;'>📊 Multi Index · Futures & Session flow</h1>",
            unsafe_allow_html=True,
        )
        st.caption("Index + VA triggers + right VP only · VEX/CEX/IV/Z-score/basket off · Net GEX every 5 min")
    with h2:
        tf = st.radio("Bar", ["3 min", "5 min", "15 min"], horizontal=True, key="multi_tf_radio",
                      index=["3 min", "5 min", "15 min"].index(want_tf) if want_tf in ("3 min", "5 min", "15 min") else 2)
        if tf != st.session_state.get("multi_tf"):
            st.session_state["multi_tf"] = tf
            st.session_state["selected_timeframe"] = tf
            st.rerun()
    with h3:
        cb_main = st.checkbox("Auto-Refresh 5s", value=st.session_state["enable_main_refresh"], key="cb_main_refresh_multi")
        if cb_main != st.session_state["enable_main_refresh"]:
            st.session_state["enable_main_refresh"] = cb_main
            st.rerun()
        gts = datetime.datetime.fromtimestamp(float(st.session_state.get("multi_gex_ts") or 0)).strftime("%H:%M:%S") if st.session_state.get("multi_gex_ts") else "—"
        st.caption(f"GEX {gts}")
    st.markdown("</div>", unsafe_allow_html=True)

    if not enabled:
        st.info("Enable at least one index in sidebar → 3. Multi Index Mode.")
        return
    if not get_smart_api_client():
        detail = st.session_state.get("_smart_api_err") or "unknown"
        st.error(f"Missing credentials or failed to generate SmartAPI session! {detail}")
        return


    def _render_one(name):
        pack = store.get(name) or {}
        hide_key = f"multi_hide_{name}"
        head, hide_c, gex_c = st.columns([0.50, 0.18, 0.32])
        with hide_c:
            hidden = st.checkbox("Hide", value=bool(st.session_state.get(hide_key, False)), key=hide_key)
        last = pack.get("spot_price")
        gex_txt = fmt_compact_num(pack.get("net_gex_oi")) if pack.get("net_gex_oi") is not None else "—"
        with head:
            st.markdown(
                f"<div style='font-size:13px;font-weight:800;color:#00E676;'>{name}"
                f"{' · ' + f'{last:,.0f}' if last else ''} · {pack.get('ts') or ''}</div>",
                unsafe_allow_html=True,
            )
        with gex_c:
            st.markdown(
                f"<div style='text-align:right;font-size:12px;color:#8FA4B8;'>Net GEX "
                f"<span style='color:#00E676;font-weight:800;'>{gex_txt}</span></div>",
                unsafe_allow_html=True,
            )
        if hidden:
            st.caption("Hidden — tape skipped.")
            return
        dfi, _sess = _multi_prepare_session(
            pack.get("df_futures"), pack.get("df_candles"), pack.get("spot_price"), want_tf,
            index_name=name,
        )
        if dfi is None or dfi.empty:
            st.caption(f"{name}: no session tape yet.")
            return
        vp_src = pd.DataFrame({
            "open": dfi["open"].astype(float),
            "high": dfi["high"].astype(float),
            "low": dfi["low"].astype(float),
            "close": dfi["close"].astype(float),
            "volume": dfi["volume"].astype(float) if "volume" in dfi.columns else 1.0,
        })
        vp = compute_session_volume_profile(vp_src, bin_step=2.0, prominence_factor=0.35)
        try:
            micro = classify_microstructure(dfi)
            act = _va_short_label(micro.get("action") or "")
            raw = str(micro.get("action") or "")
            up = raw.upper()
            if act and ("NO ENTRY" not in up) and ("CHOP" not in up) and ("INSIDE" not in up):
                col = "#FF5252" if "SHORT" in up else "#00E676"
                st.markdown(
                    f"<div style='margin:2px 0 4px 0;padding:6px 10px;border:1px solid {col};"
                    f"border-radius:6px;background:rgba(20,24,32,0.9);color:{col};"
                    f"font-weight:800;font-size:13px;letter-spacing:0.04em;'>"
                    f"VA TRIGGER · {micro.get('regime','')} · {raw}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"VA · {micro.get('regime','')} · {raw or '—'}")
        except Exception:
            pass
        fig = build_multi_index_figure(name, dfi, vp)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"multi_fig_{name}")

    visible = [n for n in enabled if not st.session_state.get(f"multi_hide_{n}")]
    hidden_only = [n for n in enabled if st.session_state.get(f"multi_hide_{n}")]
    for i in range(0, len(visible), 2):
        pair = visible[i:i+2]
        cols = st.columns(2, gap="small")
        for c, name in zip(cols, pair):
            with c:
                _render_one(name)
        if len(pair) == 1:
            with cols[1]:
                st.empty()
    if hidden_only:
        st.caption("Hidden: " + ", ".join(hidden_only))
        for name in hidden_only:
            st.checkbox("Hide", value=True, key=f"multi_hide_{name}")

def _scalper_resolve_atm_token(data, lab):
    lab = str(lab).upper()
    tok = data.get("atm_ce_token") if lab == "CE" else data.get("atm_pe_token")
    if tok and str(tok) not in ("", "nan", "None"):
        return str(tok)
    try:
        strike = int(data.get("atm_strike") or 0)
        if not strike:
            return ""
        return str(get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, strike, lab) or "")
    except Exception:
        return ""


def _scalper_fetch_opt(api, tok, lab, want_tf):
    cache_key = f"_scalp_df_{Index_Name}_{lab}_{want_tf}"
    exch_opt = (st.session_state.get("data_store") or {}).get("opt_exchange") or Exchange
    intervals = []
    primary, _ = interval_mapping.get(want_tf, ("THREE_MINUTE", 10))
    intervals.append((want_tf, primary))
    for lab_tf, api_int in (("3 min", "THREE_MINUTE"), ("5 min", "FIVE_MINUTE"), ("1 min", "ONE_MINUTE")):
        if api_int not in [x[1] for x in intervals]:
            intervals.append((lab_tf, api_int))
    if not api or not tok:
        prev = st.session_state.get(cache_key)
        return (prev if isinstance(prev, pd.DataFrame) else pd.DataFrame()), want_tf
    last_df = pd.DataFrame()
    used = want_tf
    for tf_lab, api_int in intervals:
        for attempt in range(2):
            try:
                dfo, _ = fetch_candles_with_holiday_fallback(
                    api, str(tok), exch_opt, api_int, 1, f"{Index_Name}_{lab}"
                )
            except Exception:
                dfo = pd.DataFrame()
            need = 40 if api_int == "ONE_MINUTE" else 8
            if dfo is not None and not dfo.empty and len(dfo) >= need:
                st.session_state[cache_key] = dfo
                return dfo, tf_lab
            if dfo is not None and not dfo.empty:
                last_df = dfo
                used = tf_lab
            time.sleep(0.25 + 0.2 * attempt)
    if last_df is not None and not last_df.empty:
        st.session_state[cache_key] = last_df
        return last_df, used
    prev = st.session_state.get(cache_key)
    if isinstance(prev, pd.DataFrame) and not prev.empty:
        return prev, want_tf
    return pd.DataFrame(), want_tf


def render_scalper_mode():
    if st.session_state.get("app_view") != "scalper":
        return
    st.session_state["pdec_labels_on"] = True
    st.session_state["atm_live_ok"] = True
    want_tf = st.session_state.get("selected_timeframe", "3 min")
    if "data_store" not in st.session_state or run_btn:
        refreshed = fetch_live_data(want_tf)
        if refreshed:
            refreshed["selected_expiry"] = selected_expiry_str
            refreshed["bar_tf"] = want_tf
            st.session_state["data_store"] = refreshed
    data = st.session_state.get("data_store")
    if not data:
        st.info("Click 🚀 Fetch Chain & Greeks to load ATM tokens and spot.")
        return
    auto = bool(st.session_state.get("enable_main_refresh", False))
    live_now, _, _, _ = market_session_state(_ist_now(), Index_Name)
    if auto and live_now:
        st.session_state["data_store"] = refresh_index_tapes(data, want_tf)
        data = st.session_state["data_store"]

    st.markdown(
        f"<h1 class='custom-heading' style='margin:0;font-size:16px;'>⚡ Scalper · {Index_Name} "
        f"{data.get('atm_strike') or ''} · {want_tf}</h1>",
        unsafe_allow_html=True,
    )
    ctrl1, ctrl2, ctrl3 = st.columns([0.28, 0.42, 0.30])
    with ctrl1:
        auto_on = st.toggle(
            "Auto-Refresh 5s",
            value=bool(st.session_state.get("enable_main_refresh")),
            key="scalper_auto_toggle",
        )
        if auto_on != bool(st.session_state.get("enable_main_refresh")):
            st.session_state["enable_main_refresh"] = auto_on
            st.rerun()
    with ctrl2:
        opts = ["1 min", "3 min", "5 min"]
        cur = want_tf if want_tf in opts else "3 min"
        tf = st.radio("Bar", opts, horizontal=True, index=opts.index(cur), key="scalper_tf_radio")
        if tf != st.session_state.get("selected_timeframe"):
            st.session_state["selected_timeframe"] = tf
            st.rerun()
    with ctrl3:
        st.metric("Spot", f"{float(data.get('spot_price') or 0):,.0f}")
    st.caption("Spot full width · ATM PE / ATM CE 50-50 below · VA + right VP + Vol / EFI / CVD")

    api = get_smart_api_client()
    ce_tok = _scalper_resolve_atm_token(data, "CE")
    pe_tok = _scalper_resolve_atm_token(data, "PE")
    # CE first so a rate-limit on the second call does not blank the call pane
    ce_df, ce_tf = _scalper_fetch_opt(api, ce_tok, "CE", want_tf)
    pe_df, pe_tf = _scalper_fetch_opt(api, pe_tok, "PE", want_tf)
    spot_src = data.get("df_candles")
    if spot_src is None or getattr(spot_src, "empty", True):
        spot_src = data.get("df_futures")
    spot_sess, _ = _multi_prepare_session(
        data.get("df_futures"), data.get("df_candles"), data.get("spot_price"), want_tf,
        index_name=Index_Name,
    )
    if spot_sess is not None and not spot_sess.empty:
        spot_df = pd.DataFrame({
            "time": spot_sess["time"],
            "open": spot_sess["spot_px"].astype(float) + (spot_sess["open"].astype(float) - spot_sess["close"].astype(float)),
            "high": spot_sess["spot_px"].astype(float) + (spot_sess["high"].astype(float) - spot_sess["close"].astype(float)),
            "low": spot_sess["spot_px"].astype(float) + (spot_sess["low"].astype(float) - spot_sess["close"].astype(float)),
            "close": spot_sess["spot_px"].astype(float),
            "volume": spot_sess["volume"] if "volume" in spot_sess.columns else 1.0,
        })
    else:
        spot_df = spot_src if spot_src is not None else pd.DataFrame()

    st.session_state["_scalp_spot_df"] = spot_df
    st.session_state["_scalp_pe_df"] = pe_df
    st.session_state["_scalp_ce_df"] = ce_df
    if st.session_state.get("gemini_enabled"):
        try:
            maybe_gemini_scalper_setups()
        except Exception:
            pass
    cards = _parse_gemini_setups(st.session_state.get("gemini_regular") or "")
    axis = session_axis_labels(want_tf, Index_Name)

    def _pane(title, dfp, tf_used, key, height=560):
        st.markdown(f"<div class='chart-card'><div class='card-title'>{title} · {tf_used}</div>", unsafe_allow_html=True)
        fig, act = _option_session_figure(dfp, title, Index_Name, tf_used, axis)
        if act and "NO" not in str(act).upper() and "CHOP" not in str(act).upper():
            colr = "#FF5252" if "SHORT" in str(act).upper() else "#00E676"
            st.markdown(
                f"<div style='padding:4px 8px;margin:0 0 4px 0;border:1px solid {colr};border-radius:6px;"
                f"color:{colr};font-weight:800;font-size:11px;'>VA · {act}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"VA · {act or '—'}")
        if fig is not None:
            fig.update_layout(height=height, margin=dict(l=28, r=4, t=6, b=14))
            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.caption("No tape this cycle.")
        st.markdown("</div>", unsafe_allow_html=True)

    side_l, spot_mid, side_r = st.columns([0.17, 0.66, 0.17], gap="small")
    with side_l:
        if cards:
            st.markdown(_setup_card_html(cards[0], 1), unsafe_allow_html=True)
        elif st.session_state.get("gemini_enabled"):
            st.caption("Gemini setup 1…")
    with spot_mid:
        _pane("Spot", spot_df, want_tf, "scalp_sp", height=620)
    with side_r:
        if len(cards) > 1:
            st.markdown(_setup_card_html(cards[1], 2), unsafe_allow_html=True)
        elif cards and "NO TRADE" not in cards[0].upper():
            st.caption("")
        elif st.session_state.get("gemini_enabled") and not cards:
            st.caption("Gemini setup 2…")
    pe_col, ce_col = st.columns(2)
    with pe_col:
        _pane("ATM PE", pe_df, pe_tf, "scalp_pe", height=560)
    with ce_col:
        _pane("ATM CE", ce_df, ce_tf, "scalp_ce", height=560)


@st.fragment(run_every=5 if st.session_state.get("enable_main_refresh") else None)
def live_dashboard_fragment():
    view = st.session_state.get("app_view") or "default"
    if view == "multi" or st.session_state.get("multi_index_mode"):
        render_multi_index_mode()
        return
    if view == "scalper":
        render_scalper_mode()
        return
    # Default only below this line — Scalper/Multi API paths do not run.
    if "data_store" not in st.session_state:
        st.info("Please click '🚀 Fetch Chain & Greeks' in the sidebar to load data.")
        return
    if st.session_state.get("fut_tf_radio") in ("1 min", "2 min", "3 min", "5 min", "15 min"):
        st.session_state["selected_timeframe"] = st.session_state["fut_tf_radio"]
        st.session_state["tf_select_frag"] = st.session_state["fut_tf_radio"]
    want_tf = st.session_state.get("selected_timeframe", "5 min")
    stored = st.session_state.get("data_store") or {}
    need_tf = stored.get("bar_tf") != want_tf
    idx_live = Index_Name or st.session_state.get("_last_index") or "NIFTY"
    live_now, _, _, _ = market_session_state(_ist_now(), idx_live)
    if live_now:
        st.session_state["_closed_refresh_skip"] = False
    auto = bool(st.session_state.get("enable_main_refresh", False))
    intel = bool(st.session_state.get("intel_refresh", False))
    have = st.session_state.get("data_store")
    now_s = datetime.datetime.now().timestamp()
    if have is None or need_tf:
        refreshed_data = fetch_live_data(want_tf)
        if refreshed_data:
            refreshed_data["selected_expiry"] = selected_expiry_str
            refreshed_data["bar_tf"] = want_tf
            st.session_state["data_store"] = refreshed_data
            st.session_state["_full_fetch_ts"] = now_s
    elif auto and live_now:
        # Intelligent: candles + VA only. Never re-pull chain/GEX/IV/ATM.
        st.session_state["data_store"] = refresh_index_tapes(have, want_tf)
        st.session_state["_tape_ts"] = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")
        gex_sec = int(st.session_state.get("gex_refresh_sel") or st.session_state.get("gex_refresh_min") or 5) * 60
        last_full = float(st.session_state.get("_full_fetch_ts") or 0)
        if (now_s - last_full) >= gex_sec:
            refreshed_data = fetch_live_data(want_tf)
            if refreshed_data:
                refreshed_data["selected_expiry"] = selected_expiry_str
                refreshed_data["bar_tf"] = want_tf
                st.session_state["data_store"] = refreshed_data
                st.session_state["_full_fetch_ts"] = now_s
                st.session_state["_gex_ts"] = datetime.datetime.now(
                    pytz.timezone("Asia/Kolkata")
                ).strftime("%H:%M:%S")
    elif auto and not live_now:
        st.session_state["_closed_refresh_skip"] = True

    data = st.session_state["data_store"]
    lvls = data.get("levels", {})

    # UI is loading data – clear any previous status
    clear_load_status()

    # ========== STICKY COMPACT MARKET SUMMARY (cleaned) ==========
    st.markdown("<div class='sticky-summary'>", unsafe_allow_html=True)
    head_l, head_r = st.columns([0.42, 0.58])
    with head_l:
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;'>"
            f"<h1 class='custom-heading' style='margin:0;font-size:17px;'>📊 Market Summary</h1>"
            f"<span style='color:#7CB342;font-size:11px;'>Updated {data.get('timestamp','')} · tape {st.session_state.get('_tape_ts') or '—'} · GEX {st.session_state.get('_gex_ts') or '—'}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        if data.get("is_holiday_fallback", False) or st.session_state.get("_closed_refresh_skip"):
            st.caption("Market closed — last session on screen. Auto-refresh will not re-hit the API until the next open (change TF or Fetch to reload).")
    with head_r:
        r_a, r_b, r_c, r_d = st.columns([1.15, 0.85, 1.15, 1.05])
        with r_a:
            cb_main = st.checkbox("Auto-Refresh 5s", value=st.session_state["enable_main_refresh"], key="cb_main_refresh")
        with r_b:
            st.session_state["atm_live_ok"] = st.checkbox(
                "ATM live", value=st.session_state.get("atm_live_ok", False),
                key="cb_atm_live", help="Fetch ATM CE/PE candles on refresh.",
            )
        with r_c:
            st.session_state["intel_refresh"] = st.checkbox(
                "Intelligent refresh",
                value=st.session_state.get("intel_refresh", False),
                key="cb_intel_refresh",
                help="Candles + VA every 5s. GEX/chain on the interval below.",
            )
        with r_d:
            st.session_state["gex_refresh_min"] = st.selectbox(
                "GEX / chain",
                options=[3, 5],
                index=1 if st.session_state.get("gex_refresh_min", 5) == 5 else 0,
                format_func=lambda m: f"every {m} min",
                key="gex_refresh_sel",
                label_visibility="collapsed",
            )
        if cb_main != st.session_state["enable_main_refresh"]:
            st.session_state["enable_main_refresh"] = cb_main
            st.rerun()

    # Core metrics only
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Spot (Fut)", f"{data['spot_price']:.0f} ({data['F']:.0f})")
    c2.metric("Max Pain", f"{data['max_pain_strike']}")
    c3.metric("Net GEX (OI)", fmt_compact_num(data.get("total_net_gex_oi")))
    c4.metric("ATM IV Rank", f"{data['iv_percentile']:.0f}%")
    c5.metric("PCR", f"{data['pcr']:.2f}")
    c6.metric("C/P OI", f"{data['total_call_oi']//1000}k/{data['total_put_oi']//1000}k")
    c7.metric("Flip", f"{lvls.get('Zero_Gamma_Flip', '–')}")
    straddle_val = lvls.get("Straddle_Cost", 0)
    c8.metric("Straddle", f"₹{straddle_val:.0f}" if straddle_val else "–")

    st.markdown("</div>", unsafe_allow_html=True)

    # ========== SUPERHUMAN DECISION ENGINE ==========
    _sc_src = data.get("df_futures") if data.get("df_futures") is not None and not getattr(data.get("df_futures"), "empty", True) else data.get("df_candles", pd.DataFrame())
    _sc_sig = (
        st.session_state.get("_gex_ts"),
        float(data.get("spot_price") or 0),
        int(len(_sc_src)) if _sc_src is not None and hasattr(_sc_src, "__len__") else 0,
    )
    if st.session_state.get("_scores_sig") == _sc_sig and isinstance(st.session_state.get("_scores_cache"), dict):
        scores = st.session_state["_scores_cache"]
    else:
        scores = compute_superhuman_scores(data, _sc_src if _sc_src is not None else pd.DataFrame())
        st.session_state["_scores_cache"] = scores
        st.session_state["_scores_sig"] = _sc_sig

    if "error" not in scores:
        # ----- Decision Log (persist bias changes during the day) -----
        decision_log = _load_decision_log(Index_Name)
        now_ist = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        should_log = False
        if not decision_log:
            should_log = True
        else:
            last = decision_log[-1]
            if last["bias"] != scores["bias"] or (now_ist - last["ts"]).total_seconds() >= 900:
                should_log = True
        if should_log:
            decision_log.append({
                "ts": now_ist,
                "bias": scores["bias"],
                "composite": scores["composite"],
                "clarity": scores.get("clarity", ""),
            })
            day_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            decision_log = [e for e in decision_log if e["ts"] >= day_start]
            _save_decision_log(decision_log, Index_Name)

        trig = scores.get("dir_trigger") or {}
        st.markdown(
            f"<div style='background:#1A1F2B;border:1px solid #2A2F3A;border-radius:8px;"
            f"padding:6px 12px;margin:6px 0 8px 0;display:flex;flex-wrap:wrap;align-items:center;gap:14px;'>"
            f"<span style='font-weight:700;color:#00E676;font-size:13px;'>🧠 Superhuman</span>"
            f"<span style='font-weight:800;color:{scores['colour']};font-size:14px;'>"
            f"{scores['bias']} ({scores['composite']:+.0f})</span>"
            f"<span style='color:#AAA;font-size:12px;'>{scores.get('clarity','')}</span>"
            f"<span style='font-weight:800;color:{trig.get('colour','#FF9800')};font-size:13px;'>"
            f"⚡ {trig.get('trigger','NO TRIGGER')}</span>"
            f"<span style='color:#CCC;font-size:12px;'>{trig.get('summary','')}</span>"
            f"<span style='color:#888;font-size:11px;margin-left:auto;'>"
            f"L {trig.get('long_hits',0)}/5 · S {trig.get('short_hits',0)}/5</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        with st.expander("▼ Trigger · Decision tree · Intraday log", expanded=False):
            trig = scores.get("dir_trigger") or {}
            st.caption(
                "LONG TRIGGER = ≥3 of GEX/VWAP/OBV/CVD bullish AND EFI>0. "
                "SHORT TRIGGER = ≥3 bearish AND EFI<0. "
                "BIAS (no fire) = 3/4 setup aligned but EFI has not confirmed. "
                "LEAN (no fire) = fewer than 3/4 setup checks; EFI does not matter yet. "
                "NO DIRECTIONAL TRIGGER = setup split."
            )
            t1, t2, t3 = st.columns([1.35, 0.90, 0.75])
            with t1:
                rules = {
                    "Location (GEX)": ("Above put wall / LVN", "Below call wall"),
                    "Trend (VWAP)": ("Price > VWAP", "Price < VWAP"),
                    "Macro Flow (OBV)": ("OBV > MA20", "OBV < MA20"),
                    "Order Delta (CVD)": ("CVD HH", "CVD LL"),
                    "Execution (EFI 13)": ("EFI > 0", "EFI < 0"),
                }
                lines = ["| Metric | L | S | Now |", "|---|---|---|---|"]
                for ch in trig.get("checks") or []:
                    side = "L" if ch.get("long") and not ch.get("short") else ("S" if ch.get("short") and not ch.get("long") else "—")
                    lr, sr = rules.get(ch["name"], ("", ""))
                    lines.append(f"| {ch['name']} | {lr} | {sr} | **{side}** {ch.get('note','')} |")
                st.markdown("\n".join(lines))
            with t2:
                st.markdown(
                    f"**{scores['bias']} ({scores['composite']:+.0f})**  \n"
                    f"- Quiet + long γ → PIN  \n"
                    f"- Big range + long γ → REVERSION  \n"
                    f"- Short γ / wall break → TREND  \n"
                    f"- |C|≤15 → NO EDGE  \n"
                    f"- else → MILD DIR"
                )
            with t3:
                st.markdown("**Log**")
                if not decision_log:
                    st.caption("No log yet.")
                else:
                    for e in decision_log[-8:]:
                        st.caption(f"{e['ts'].strftime('%H:%M')} {e['bias']} ({e['composite']:+.0f})")

        with st.expander("▼ VA Playbook (range fade vs trend acceptance)", expanded=False):
            rows = ["| Code | Setup · Action |", "|---|---|"]
            for k, v in VA_PLAYBOOK.items():
                rows.append(f"| `{k}` | {v[0]} — **{v[1]}** |")
            st.markdown("\n".join(rows))
            st.markdown(
                """
**Regime (RANGE vs TREND vs CHOP)** — scored, not a single switch:
- Value density = session volume sitting between VAL and VAH (high → range / pin).
- Efficiency ratio = |net move| / Σ|bar moves| over 20 bars (high → trend).
- VWAP z-score stretch, session range vs ATR√t, net GEX sign (long-γ leans range, short-γ leans trend).
- RANGE if range-score ≥ 1.4 and beats trend-score by ≥ 0.35; TREND is the mirror; else CHOP.

**Location is statistical, not a tick print:**
- Buffer = max(0.35 × session VWAP-σ, 0.35 × ATR, 1 pt).
- `ABOVE_VAH` only if last ≥ VAH + buffer; `z = (px − VAH) / σ`.
- Persistence: ≥2 of last 3 closes on that side of the level.

**ΔV absorption:** last-12 OLS slope of price vs last-5 signed volume. Probe + non-confirming ΔV = absorb.
**EFI:** last-12 OLS slope. Fade needs EFI *not* expanding with price; acceptance needs EFI expanding *with* ΔV.
                """
            )
            try:
                dfi_now = st.session_state.get("_last_dfi")
                if dfi_now is not None:
                    rec = classify_va_setup(dfi_now, st.session_state.get("data_store"))
                    st.caption(
                        f"Now: {rec.get('regime')} · {rec.get('model')} · {rec.get('action')} · {rec.get('efi_note')}"
                    )
            except Exception:
                pass

        with st.expander("▼ Δ Footprint absorption (proxy setups)", expanded=False):
            st.code(
"""detect_fp_absorptions — last CLOSED bar only, lookback 20 bars, no lookahead
Not bid/ask tape. Shelf = signed close-location volume in 2-pt bins.

z_sweep_dn = (mean(low[-20:-1]) - this.low) / std(low[-20:-1])
z_sweep_up = (this.high - mean(high[-20:-1])) / std(high[-20:-1])
typ, sig   = mean/std of |bin volume| on all prior bins
sell_low   = sum of negative bins in lower 33% of this bar
buy_high   = sum of positive bins in upper 33% of this bar
z_sell     = (sell_low - typ) / sig
z_buy      = (buy_high - typ) / sig
loc        = (close - low) / (high - low)

SETUP 1 BID ABS (long / spring)  — green ▲
  z_sweep_dn >= 1.6
  loc >= 0.72 and close >= open
  z_sell >= 1.4
  Entry: close of that bar
  Stop:  this.low - 2 pts
  Target: next resistance / top-5 ask cluster (discretionary)

SETUP 2 OFFER ABS (short / upthrust) — red ▼
  z_sweep_up >= 1.6
  loc <= 0.28 and close < open
  z_buy >= 1.4
  Entry: close of that bar
  Stop:  this.high + 2 pts
  Target: next support / top-5 bid cluster (discretionary)

If any gate fails → no mark. Caption on the tab shows BID ABS n · OFFER ABS n.
""",
                language="text",
            )
            evs = st.session_state.get("_fp_abs_evs") or []
            st.caption(
                f"Session marks: {sum(1 for e in evs if e.get('side')>0)} bid · "
                f"{sum(1 for e in evs if e.get('side')<0)} offer"
            )

        # Score Breakdown
        with st.expander("▼ Score Breakdown & Details", expanded=False):
            or_txt = f"{scores['or_low']:.0f}-{scores['or_high']:.0f}" if scores.get("or_low") is not None else "N/A"
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"**Gamma** {scores['gamma_regime_score']:+.0f}")
                st.caption(f"OI GEX Rs {scores['total_delta_gex_cr']:.0f} Cr · DTE {scores.get('dte','-')}")
                st.markdown(f"**Exp vs Real** {scores['move_score']:+.0f}")
                st.caption(f"{scores['expected_move_pct']:.2f}% vs {scores['realised_range_pct']:.2f}%")
            with s2:
                st.markdown(f"**Vanna/Charm** {scores['flow_score']:+.0f}")
                st.caption("tanh(VEX+CEX) · pin vs accel")
                st.markdown(f"**OR vs Walls** {scores['or_score']:+.0f}")
                st.caption(f"OR {or_txt} · tod {scores.get('tod_factor',1):.2f}")
            with s3:
                st.markdown(f"**Flip dist** {scores.get('flip_score',0):+.0f}")
                st.caption(f"{scores.get('dist_to_flip_pct',0):.3f}% from flip")
                st.markdown(f"**Composite** {scores['composite']:+.0f} → {scores['bias']}")
                st.caption("0.38g + 0.22 move + 0.15 flow + 0.15 OR + 0.10 flip")

    else:
        st.warning("Could not compute Superhuman scores – insufficient data.")

    # ========== UNDERLYING TECHNICALS – SPOT + FUTURES VWAP ==========
    df_full = data.get("df_candles", pd.DataFrame())
    df_fut  = data.get("df_futures", pd.DataFrame())
    basis   = data.get("basis_info", {})

    if not df_full.empty and len(df_full) >= 3:
        latest_row = df_full.iloc[-1]
        def _cell(col, default=float("nan")):
            try:
                if col in df_full.columns:
                    v = latest_row[col]
                    return float(v) if pd.notna(v) else default
            except Exception:
                pass
            return default
        rsi_val = _cell("rsi", 50.0)
        rsi_status = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
        rsi_badge_cls = "badge-bearish" if rsi_val > 70 else ("badge-bullish" if rsi_val < 30 else "badge-neutral")
        macd_val = _cell("macd", 0.0)
        macd_sig = _cell("macd_signal", 0.0)
        macd_status = "Bullish XO" if macd_val > macd_sig else "Bearish XO"
        macd_badge_cls = "badge-bullish" if macd_val > macd_sig else "badge-bearish"
        if "bb_bandwidth" in df_full.columns:
            recent_bw = pd.to_numeric(df_full["bb_bandwidth"], errors="coerce").tail(20)
            bw_threshold = float(recent_bw.quantile(0.20) or 0)
            is_sqz = _cell("bb_bandwidth", 0.0) <= bw_threshold
        else:
            is_sqz = False
        sqz_status = "Squeeze" if is_sqz else "Expand"
        sqz_badge_cls = "badge-neutral" if is_sqz else "badge-bullish"
        badge_html = (
            f"<span class='status-badge {sqz_badge_cls}'>BB: {sqz_status}</span>"
            f"<span class='status-badge {macd_badge_cls}'>MACD: {macd_status}</span>"
            f"<span class='status-badge {rsi_badge_cls}'>RSI: {rsi_val:.0f} ({rsi_status})</span>"
        )
    else:
        badge_html = ""

    tech_h, tech_b, tf_col = st.columns([0.28, 0.52, 0.20])
    with tech_h:
        heading_ribbon("📈 Underlying Technicals",
            "Spot + Bollinger (20,2). MACD 12/26/9. RSI 14. Squeeze = BB width ≤ 20th pct of last 20 bars.")
    with tech_b:
        if badge_html:
            st.markdown(f"<div style='padding-top:2px;'>{badge_html}</div>", unsafe_allow_html=True)
    with tf_col:
        st.caption(f"TF {st.session_state.get('selected_timeframe','5 min')}")

    if not df_full.empty and len(df_full) >= 20:

        # Spot chart (no VWAP)
        df_full["session_date"] = pd.to_datetime(df_full["time"]).dt.date
        last_3_dates = sorted(df_full["session_date"].unique())[-3:]
        df_chart = df_full[df_full["session_date"].isin(last_3_dates)].copy()
        df_chart["time_str"] = pd.to_datetime(df_chart["time"]).dt.strftime("%d-%b %H:%M")

        min_p = min(df_chart["close"].min(), df_chart["bb_lower"].min())
        max_p = max(df_chart["close"].max(), df_chart["bb_upper"].max())
        padding = (max_p - min_p) * 0.05

        df_chain = pd.DataFrame(data.get("chain_results") or [])
        lvls_tech = data.get("levels") or {}
        tech_left, tech_gex = st.columns([0.50, 0.50])

        with tech_left:
            tab_bb, tab_osc = st.tabs([f"{Index_Name} + BB", "MACD / RSI"])
            with tab_bb:
                fig_px = plt_go.Figure()
                fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["close"], mode="lines", name="Spot", line=dict(color="#00E676", width=2)))
                fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_upper"], mode="lines", name="BB Upper", line=dict(color="rgba(33,150,243,0.5)", width=1)))
                fig_px.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_lower"], mode="lines", name="BB Lower", line=dict(color="rgba(33,150,243,0.5)", width=1), fill="tonexty", fillcolor="rgba(33,150,243,0.05)"))
                fig_px.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    height=320, margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
                    hovermode="x unified", yaxis=dict(range=[min_p - padding, max_p + padding], tickformat="d"),
                    xaxis=dict(type="category", nticks=8),
                )
                st.plotly_chart(fig_px, use_container_width=True)
            with tab_osc:
                fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45])
                colors_macd = np.where(df_chart["macd_hist"] >= 0, "#00E676", "#FF5252")
                fig_ind.add_trace(plt_go.Bar(x=df_chart["time_str"], y=df_chart["macd_hist"], name="Hist", marker_color=colors_macd, showlegend=False))
                fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd"], mode="lines", name=f"MACD [{macd_val:.1f}]", line=dict(color="#2196F3", width=1.3)))
                fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd_signal"], mode="lines", name=f"Sig [{macd_sig:.1f}]", line=dict(color="#FF9800", width=1.3)))
                fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["rsi"], mode="lines", name=f"RSI [{rsi_val:.0f}]", line=dict(color="#E040FB", width=1.3)), row=2, col=1)
                fig_ind.add_hline(y=70, line_dash="dash", line_color="#FF5252", line_width=1, row=2, col=1)
                fig_ind.add_hline(y=30, line_dash="dash", line_color="#00E676", line_width=1, row=2, col=1)
                fig_ind.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    height=320, margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
                    hovermode="x unified",
                )
                fig_ind.update_xaxes(type="category", nticks=6)
                st.plotly_chart(fig_ind, use_container_width=True)
        with tech_gex:
            if not df_chain.empty and "Strike" in df_chain.columns:
                gx1, gx2, gx3, gx4 = st.tabs(["GEX/OI", "GEX/Vol", "Δ-GEX", "DEX OI"])
                spot_g = float(data.get("spot_price") or 0)
                def _gmini(y, h=300):
                    fig = plt_go.Figure()
                    cols = np.where(pd.to_numeric(y, errors="coerce").fillna(0) >= 0, "#00E676", "#FF5252")
                    fig.add_trace(plt_go.Bar(x=df_chain["Strike"], y=y, marker_color=cols, showlegend=False))
                    fig.add_vline(x=spot_g, line_dash="dash", line_color="#FAFAFA")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#11151C", plot_bgcolor="#0E1117",
                                      height=h, margin=dict(l=6, r=6, t=8, b=8), showlegend=False)
                    fig.update_xaxes(tickformat="d")
                    return fig
                with gx1:
                    if "Net_GEX_OI" in df_chain.columns:
                        st.plotly_chart(_gmini(df_chain["Net_GEX_OI"]), use_container_width=True, key=f"gmini_oi_{Index_Name}")
                with gx2:
                    if "Net_GEX_Vol" in df_chain.columns:
                        st.plotly_chart(_gmini(df_chain["Net_GEX_Vol"]), use_container_width=True, key=f"gmini_vol_{Index_Name}")
                with gx3:
                    if "Net_Delta_GEX_OI" in df_chain.columns:
                        st.plotly_chart(_gmini(df_chain["Net_Delta_GEX_OI"]), use_container_width=True, key=f"gmini_dgex_{Index_Name}")
                with gx4:
                    lotn = float(LOT_SIZES.get(Index_Name, 65))
                    c_d = pd.to_numeric(df_chain.get("C_Δ", 0), errors="coerce").fillna(0.0)
                    p_d = pd.to_numeric(df_chain.get("P_Δ", 0), errors="coerce").fillna(0.0)
                    c_oi = pd.to_numeric(df_chain.get("C_OI", 0), errors="coerce").fillna(0.0)
                    p_oi = pd.to_numeric(df_chain.get("P_OI", 0), errors="coerce").fillna(0.0)
                    dex_oi = (c_d * c_oi - p_d.abs() * p_oi) * lotn
                    st.plotly_chart(_gmini(dex_oi), use_container_width=True, key=f"gmini_dex_{Index_Name}")
            else:
                st.caption("GEX unavailable.")

        # ========== ROW1: Futures 50% | Liq Δ 20% | GEX OI 30% ==========
        # ========== ROW2: CVD 50%     | Liq Δ 20% | GEX Vol 30% ==========
        st.markdown("---")
        df_fut = data.get("df_futures", pd.DataFrame())
        df_chain = pd.DataFrame(data.get("chain_results") or [])
        basis = data.get("basis_info", {}) or {}

        def calculate_synced_ranges(primary, secondary):
            """Align zero line of primary & secondary y-axes at the same vertical position."""
            p = pd.Series(primary).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            s = pd.Series(secondary).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            p_max = max(float(p.max()) if len(p) else 0.0, 0.0) or 1.0
            p_min = min(float(p.min()) if len(p) else 0.0, 0.0) or -1.0
            s_max = max(float(s.max()) if len(s) else 0.0, 0.0) or 1.0
            s_min = min(float(s.min()) if len(s) else 0.0, 0.0) or -1.0
            p_frac = abs(p_min) / (abs(p_min) + p_max)
            s_frac = abs(s_min) / (abs(s_min) + s_max)
            frac = min(max(max(p_frac, s_frac), 0.15), 0.85)
            p_total = (abs(p_min) + p_max) * 1.10
            s_total = (abs(s_min) + s_max) * 1.10
            return [-p_total * frac, p_total * (1.0 - frac)], [-s_total * frac, s_total * (1.0 - frac)]

        df_fchart = pd.DataFrame()
        fut_times = []
        latest_session = None
        want_tf = st.session_state.get("selected_timeframe", "5 min")
        tf_min = {"1 min": 1, "2 min": 2, "3 min": 3, "5 min": 5, "15 min": 15, "10 min": 10}.get(want_tf, 3)
        if not df_fut.empty and "time" in df_fut.columns:
            df_fut = attach_bar_flow(df_fut)
        if not df_fut.empty and "time" in df_fut.columns and len(df_fut) >= 8:
            try:
                t = series_to_ist(df_fut["time"])
                dt = t.diff().dt.total_seconds().median()
                native = float(dt) / 60.0 if pd.notna(dt) and dt else tf_min
                if native > 0 and tf_min > native + 0.6:
                    g = df_fut.copy()
                    g["time"] = t
                    g = g.set_index("time").sort_index()
                    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                    if "cvd" in g.columns:
                        agg["cvd"] = "last"
                    if "obv" in g.columns:
                        agg["obv"] = "last"
                    if "efi13" in g.columns:
                        agg["efi13"] = "last"
                    if "signed_flow" in g.columns:
                        agg["signed_flow"] = "sum"
                    ohlc = g.resample(f"{int(tf_min)}min", label="right", closed="right").agg(agg).dropna(subset=["close"])
                    if len(ohlc) >= 5:
                        df_fut = ohlc.reset_index()
                        data["df_futures"] = df_fut
            except Exception:
                pass
        replay_on = bool(st.session_state.get("replay_session_on"))
        replay_day = st.session_state.get("replay_session_date")
        force_day = replay_day if replay_on and replay_day else None
        if not df_fut.empty and len(df_fut) >= 1:
            df_fchart, latest_session, _ = pick_last_nse_session(
                df_fut, min_bars=20, prefer_today=not bool(force_day), force_date=force_day,
                index_name=Index_Name,
            )
            if df_fchart.empty and force_day:
                try:
                    api = get_smart_api_client()
                    tf_lab = st.session_state.get("selected_timeframe", "5 min")
                    api_int, _lb = interval_mapping.get(tf_lab, ("FIVE_MINUTE", 15))
                    fut_tok = (data.get("basis_info") or {}).get("fut_token")
                    fut_ex = (data.get("basis_info") or {}).get("fut_exchange") or "NFO"
                    fetched = fetch_one_session_ohlcv(
                        api, fut_tok, fut_ex, api_int, force_day, Index_Name,
                    )
                    if fetched is not None and not fetched.empty:
                        fetched = attach_bar_flow(fetched, rebuild=True)
                        df_fchart = fetched
                        latest_session = force_day
                        # merge into store so next rerun is cheap
                        try:
                            base = data.get("df_futures")
                            if base is None or getattr(base, "empty", True):
                                data["df_futures"] = fetched.copy()
                            else:
                                data["df_futures"] = pd.concat([base, fetched], ignore_index=True)
                                data["df_futures"] = data["df_futures"].drop_duplicates(subset=["time"])
                        except Exception:
                            pass
                        # spot tape for that day
                        try:
                            spot_tok, spot_ex = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))[:2]
                            sp = fetch_one_session_ohlcv(api, spot_tok, spot_ex, api_int, force_day, Index_Name)
                            if sp is not None and not sp.empty:
                                data["df_candles"] = pd.concat(
                                    [data.get("df_candles", pd.DataFrame()), sp], ignore_index=True
                                ) if data.get("df_candles") is not None else sp
                        except Exception:
                            pass
                except Exception:
                    pass
            if df_fchart.empty:
                df_fut = df_fut.copy()
                df_fut["time"] = series_to_ist(df_fut["time"])
                df_fut["session_date"] = df_fut["time"].dt.date
                latest_session = sorted(df_fut["session_date"].unique())[-1]
                df_fchart = df_fut[df_fut["session_date"] == latest_session].copy().reset_index(drop=True)
                df_fchart["time_str"] = df_fchart["time"].dt.strftime("%H:%M")
            fut_times = df_fchart["time_str"].tolist() if not df_fchart.empty else []

        if not df_chain.empty:
            min_strike_val = float(df_chain["Strike"].min()) - 50
            max_strike_val = float(df_chain["Strike"].max()) + 50
            df_chain["Total_Vol"] = df_chain["C_Vol"] + df_chain["P_Vol"]
            df_chain["Total_OI"] = df_chain["C_OI"] + df_chain["P_OI"]
        else:
            min_strike_val = float(data.get("spot_price", 0) or 0) - 500
            max_strike_val = float(data.get("spot_price", 0) or 0) + 500

        # ----- Compact: Futures+VP+OBV+EFI+CVD | GEX OI + GEX Vol -----
        fut_msg = data.get("fut_fallback_msg", "")
        left_col = st.container()
        right_col = None
        vp = {"ok": False}
        y0 = y1 = None
        sigma_mult = 1.5

        if df_fchart is None or getattr(df_fchart, "empty", True):
            st.warning("Index tape empty this cycle — waiting for today's futures candles (09:15→now). Click Fetch if this stays.")
        with left_col:
            fut_header_col, tf_col, basis_col, band_col = st.columns([0.52, 0.20, 0.14, 0.14])
            with tf_col:
                opts = ["1 min", "2 min", "3 min", "5 min", "15 min"]
                if "fut_tf_radio" not in st.session_state:
                    st.session_state["fut_tf_radio"] = (
                        st.session_state.get("selected_timeframe", "3 min")
                        if st.session_state.get("selected_timeframe") in opts else "3 min"
                    )
                new_tf = st.radio("Bar", opts, horizontal=True, key="fut_tf_radio",
                                  label_visibility="collapsed")
                if new_tf != st.session_state.get("selected_timeframe"):
                    st.session_state["selected_timeframe"] = new_tf
                    st.session_state["tf_select_frag"] = new_tf
                    st.session_state["heatmap_timeframe"] = new_tf
                    st.rerun()
            with fut_header_col:
                expiry_txt = basis.get("fut_expiry", "N/A")
                fb = str(fut_msg or "").replace("**", "").replace("⚠️ ", "")
                fb_html = ""
                if data.get("fut_is_fallback") or "closed" in fb.lower():
                    fb_html = (
                        f"<span style='color:#FFD54F;font-size:11px;white-space:nowrap;'>"
                        f"{fb}</span>"
                    )
                elif fb:
                    fb_html = f"<span style='color:#7CB342;font-size:11px;white-space:nowrap;'>{fb}</span>"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:6px;flex-wrap:wrap;'>"
                    f"<div class='micro-hover' style='display:inline-block;padding:3px 10px;"
                    f"background:#1A1F2B;border:1px solid #3A4150;border-radius:8px;'>"
                    f"<span style='font-weight:700;color:#00E676;font-size:13px;'>"
                    f"📉 {Index_Name} spot + fut VWAP→index + VP ({expiry_txt})</span>"
                    f"<div class='micro-tip'><b>Line</b> = index spot.<br>"
                    f"<b>VWAP / σ</b> = futures VWAP minus bar basis (F−S).<br>"
                    f"VWAP_idx = VWAP_fut − (Fut − Spot).<br>"
                    f"<b>VP</b> futures volume profile shifted by last basis.</div></div>"
                    f"<div class='micro-hover' style='display:inline-block;padding:3px 8px;"
                    f"background:#1A1F2B;border:1px solid #3A4150;border-radius:8px;'>"
                    f"<span style='font-weight:700;color:#00E676;font-size:11px;'>OBV</span>"
                    f"<div class='micro-tip'><b>OBV</b> = Σ sign(ΔClose)×Volume. Executed net volume.</div></div>"
                    f"<div class='micro-hover' style='display:inline-block;padding:3px 8px;"
                    f"background:#1A1F2B;border:1px solid #3A4150;border-radius:8px;'>"
                    f"<span style='font-weight:700;color:#00E676;font-size:11px;'>EFI13</span>"
                    f"<div class='micro-tip'><b>EFI13</b> = EMA13((C−prev)×V). Force / execution.</div></div>"
                    f"<div class='micro-hover' style='display:inline-block;padding:3px 8px;"
                    f"background:#1A1F2B;border:1px solid #3A4150;border-radius:8px;'>"
                    f"<span style='font-weight:700;color:#00E676;font-size:11px;'>CVD</span>"
                    f"<div class='micro-tip'><b>CVD</b> = Σ V×(2(C−L)/(H−L)−1). Market-order-like proxy.</div></div>"
                    f"{fb_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with basis_col:
                if basis.get("basis") is not None:
                    basis_color = "#00E676" if basis["basis"] >= 0 else "#FF5252"
                    st.markdown(
                        f"<div style='text-align:right;font-size:14px;padding-top:2px;'>"
                        f"Basis: <span style='color:{basis_color};font-weight:800;'>"
                        f"{basis['basis']:+.1f}</span></div>",
                        unsafe_allow_html=True
                    )
            with band_col:
                sigma_mult = st.selectbox(
                    "VWAP Bands", options=[1.0, 1.5, 2.0], index=1,
                    format_func=lambda x: f"±{x}σ", key="vwap_sigma_select",
                    label_visibility="collapsed"
                )
            tw, al, av, rp_chk, rp_date = st.columns([0.18, 0.26, 0.20, 0.12, 0.24])
            with rp_chk:
                replay_on = st.checkbox(
                    "Replay day",
                    value=bool(st.session_state.get("replay_session_on")),
                    key="replay_session_chk",
                    help="Pin this pane to one past trading session.",
                )
                st.session_state["replay_session_on"] = replay_on
            with rp_date:
                today_ist = datetime.datetime.now(pytz.timezone("Asia/Kolkata")).date()
                default_day = st.session_state.get("replay_session_date") or today_ist
                try:
                    picked = st.date_input(
                        "Replay date",
                        value=default_day,
                        min_value=today_ist - datetime.timedelta(days=60),
                        max_value=today_ist,
                        key="replay_session_date_input",
                        format="DD-MM-YYYY",
                    )
                except TypeError:
                    picked = st.date_input(
                        "Replay date",
                        value=default_day,
                        min_value=today_ist - datetime.timedelta(days=60),
                        max_value=today_ist,
                        key="replay_session_date_input",
                    )
                if picked and picked != st.session_state.get("replay_session_date"):
                    st.session_state["replay_session_date"] = picked
                    if picked != today_ist:
                        st.session_state["replay_session_on"] = True
            with tw:
                st.session_state["chart_window"] = st.radio(
                    "Window", ["Session (6h)", "3h", "1h"], horizontal=True,
                    key="chart_window_radio",
                )
            with al:
                a1, a2, a3 = st.columns([0.45, 0.35, 0.20])
                with a1:
                    lvl = st.number_input("Alert px", min_value=0.0, step=1.0,
                                         value=float(st.session_state.get("px_alert_lvl") or 0) or None,
                                         placeholder=f"{Index_Name} level", key="px_alert_input",
                                         label_visibility="collapsed")
                with a2:
                    if st.button("Activate Alert", key="px_alert_btn"):
                        if lvl and float(lvl) > 0:
                            st.session_state["px_alert_lvl"] = float(lvl)
                            st.session_state["px_alert_on"] = True
                            st.session_state["px_alert_side"] = None
                with a3:
                    if st.session_state.get("px_alert_on"):
                        st.caption(f"🔔 {st.session_state['px_alert_lvl']:.0f}")
                    else:
                        st.caption("off")
            with av:
                c1, c2, c3, c4 = st.columns([1.15, 1.15, 0.95, 0.85])
                with c1:
                    st.session_state["avwap_on"] = st.checkbox("AVWAP", key="avwap_chk")
                with c2:
                    st.session_state["pdec_labels_on"] = st.checkbox(
                        "VA", value=True, key="pdec_lbl_chk",
                        help="90° VA labels above candles. NO ENTRY is never printed.",
                    )
                with c3:
                    st.session_state["big_trade_on"] = st.checkbox("Big Δ", key="big_trd_chk")
                with c4:
                    _lot_opts = [50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
                    _def_lot = 300 if str(Index_Name).upper() in ("NIFTY", "BANKNIFTY") else 100
                    if "big_trd_min" not in st.session_state:
                        st.session_state["big_trd_min"] = _def_lot
                    st.session_state["big_trade_min"] = st.selectbox(
                        "Min lots", _lot_opts,
                        index=_lot_opts.index(st.session_state["big_trd_min"]) if st.session_state.get("big_trd_min") in _lot_opts else _lot_opts.index(_def_lot),
                        key="big_trd_min", label_visibility="collapsed",
                    )

            if not df_fchart.empty:
                df_fchart = df_fchart.copy()
                df_fchart["tp"] = (df_fchart["high"] + df_fchart["low"] + df_fchart["close"]) / 3.0

                def _calc_session_vwap_bands(group, k):
                    group = group.copy().reset_index(drop=True)
                    cum_vol = cum_tp_vol = cum_sq_dev = 0.0
                    vwaps, stds = [], []
                    for i in range(len(group)):
                        vol = float(group.loc[i, "volume"])
                        tp = float(group.loc[i, "tp"])
                        cum_vol += vol
                        cum_tp_vol += tp * vol
                        vwap = cum_tp_vol / cum_vol if cum_vol > 0 else tp
                        cum_sq_dev += vol * (tp - vwap) ** 2
                        variance = cum_sq_dev / cum_vol if cum_vol > 0 else 0.0
                        vwaps.append(vwap)
                        stds.append(np.sqrt(max(variance, 0.0)))
                    group["vwap"] = vwaps
                    group["vwap_std"] = stds
                    group["vwap_upper"] = group["vwap"] + k * group["vwap_std"]
                    group["vwap_lower"] = group["vwap"] - k * group["vwap_std"]
                    return group

                df_fchart = _calc_session_vwap_bands(df_fchart, sigma_mult)
                latest_vwap = float(df_fchart["vwap"].iloc[-1])
                latest_fut = float(df_fchart["close"].iloc[-1])
                fmin = min(df_fchart["close"].min(), df_fchart["vwap_lower"].min())
                fmax = max(df_fchart["close"].max(), df_fchart["vwap_upper"].max())
                fpad = (fmax - fmin) * 0.06
                y0, y1 = fmin - fpad, fmax + fpad
                vp = compute_session_volume_profile(df_fchart, bin_step=2.0, prominence_factor=0.35)

                dfi = df_fchart.copy().reset_index(drop=True)
                dfi["time"] = series_to_ist(dfi["time"])
                dfi["spot_px"] = np.nan
                df_sp = data.get("df_candles")
                if df_sp is not None and not getattr(df_sp, "empty", True) and "close" in df_sp.columns:
                    try:
                        sp = df_sp.copy()
                        sp["time"] = series_to_ist(sp["time"])
                        sp = sp.dropna(subset=["time"]).sort_values("time")
                        if latest_session is not None:
                            sp = sp[sp["time"].dt.date == latest_session]
                        left = dfi[["time"]].sort_values("time")
                        mapped = pd.merge_asof(
                            left,
                            sp[["time", "close"]].rename(columns={"close": "spot_px"}).sort_values("time"),
                            on="time", direction="nearest",
                            tolerance=pd.Timedelta("6min"),
                        )
                        dfi["spot_px"] = pd.to_numeric(mapped["spot_px"], errors="coerce").values
                    except Exception:
                        dfi["spot_px"] = np.nan
                n_ok = int(pd.to_numeric(dfi["spot_px"], errors="coerce").notna().sum())
                if n_ok < max(5, len(dfi) // 4):
                    # not enough aligned index bars — use futures minus last quoted basis
                    last_s = float(data.get("spot_price") or 0)
                    last_f = float(dfi["close"].iloc[-1])
                    basis_q = last_f - last_s if last_s else float(data.get("basis_info", {}).get("basis", 0) or 0)
                    dfi["spot_px"] = dfi["close"].astype(float) - basis_q
                else:
                    dfi["spot_px"] = pd.to_numeric(dfi["spot_px"], errors="coerce").ffill().bfill()
                dfi["basis_raw"] = dfi["close"].astype(float) - dfi["spot_px"].astype(float)
                # Smooth ONLY the F−S shift. Futures VWAP itself is untouched.
                dfi["basis"] = dfi["basis_raw"].ewm(span=5, min_periods=1, adjust=False).mean()
                dfi["vwap_idx"] = dfi["vwap"].astype(float) - dfi["basis"]
                if "vwap_upper" in dfi.columns:
                    dfi["vwap_upper_idx"] = dfi["vwap_upper"].astype(float) - dfi["basis"]
                    dfi["vwap_lower_idx"] = dfi["vwap_lower"].astype(float) - dfi["basis"]
                smin = float(np.nanmin([dfi["spot_px"].min(), dfi.get("vwap_lower_idx", dfi["vwap_idx"]).min()]))
                smax = float(np.nanmax([dfi["spot_px"].max(), dfi.get("vwap_upper_idx", dfi["vwap_idx"]).max()]))
                spad = (smax - smin) * 0.06 if smax > smin else 20
                y0, y1 = smin - spad, smax + spad
                last_basis = float(dfi["basis"].iloc[-1])
                if vp.get("ok"):
                    vp = dict(vp)
                    for k in ("mids", "poc", "val1", "vah1", "val15", "vah15", "vah", "val",
                              "vah80", "val80", "hvn", "lvn"):
                        if k == "mids" and vp.get("mids") is not None:
                            vp["mids"] = [float(m) - last_basis for m in vp["mids"]]
                        elif k in ("hvn", "lvn") and vp.get(k):
                            vp[k] = [float(m) - last_basis for m in vp[k]]
                        elif k in vp and vp[k] is not None and not isinstance(vp[k], (list, np.ndarray)):
                            try:
                                vp[k] = float(vp[k]) - last_basis
                            except Exception:
                                pass
                    if vp.get("vas"):
                        shifted = []
                        for va in vp["vas"]:
                            nv = dict(va)
                            for kk in ("vah", "val", "poc"):
                                if nv.get(kk) is not None:
                                    nv[kk] = float(nv[kk]) - last_basis
                            shifted.append(nv)
                        vp["vas"] = shifted
                    if vp.get("nodes"):
                        nsh = []
                        for nd in vp["nodes"]:
                            nv = dict(nd)
                            for kk in ("vah", "val", "poc"):
                                if nv.get(kk) is not None:
                                    nv[kk] = float(nv[kk]) - last_basis
                            nsh.append(nv)
                        vp["nodes"] = nsh
                close = dfi["close"].astype(float)
                vol = dfi["volume"].astype(float)
                if "cvd" not in dfi.columns or dfi["cvd"].isna().all():
                    dfi = attach_bar_flow(dfi, rebuild=True)
                elif "obv" not in dfi.columns:
                    dfi = attach_bar_flow(dfi, rebuild=True)
                dfi["obv_ma20"] = dfi["obv"].rolling(20, min_periods=1).mean()

                win = st.session_state.get("chart_window") or "Session (6h)"
                if st.session_state.get("replay_session_on"):
                    win = "Session (6h)"
                if "time" in dfi.columns and win in ("3h", "1h") and len(dfi):
                    tlast = pd.to_datetime(dfi["time"].iloc[-1])
                    hrs = 3 if win == "3h" else 1
                    cut = tlast - pd.Timedelta(hours=hrs)
                    dfi = dfi[pd.to_datetime(dfi["time"]) >= cut].copy().reset_index(drop=True)
                    if "time_str" in dfi.columns:
                        fut_times = dfi["time_str"].tolist()

                axis_times = session_axis_labels(st.session_state.get("selected_timeframe"), Index_Name)
                st.session_state["_axis_times"] = axis_times
                dfi["avwap_idx"] = np.nan
                if st.session_state.get("avwap_on") and st.session_state.get("avwap_time") and len(dfi):
                    at = str(st.session_state["avwap_time"])
                    if "time_str" in dfi.columns and at in set(dfi["time_str"].astype(str)):
                        i0 = int(dfi.index[dfi["time_str"].astype(str) == at][0])
                        sub = dfi.iloc[i0:].copy()
                        tp = (sub["high"] + sub["low"] + sub["close"]) / 3.0
                        vol = sub["volume"].astype(float)
                        cv = vol.cumsum().replace(0, np.nan)
                        av = (tp.astype(float) * vol).cumsum() / cv
                        if "basis" in sub.columns:
                            av_idx = av - sub["basis"].astype(float)
                        else:
                            av_idx = av
                        dfi.loc[sub.index, "avwap_idx"] = av_idx.values

                cvd_st = {"ok": False}
                tf_lab = st.session_state.get("selected_timeframe", "5 min")
                tf_min = 15 if "15" in tf_lab else (3 if "3" in tf_lab else 5)
                idx_o = dfi["open"] if "open" in dfi.columns else dfi["close"]
                idx_h = dfi["high"] if "high" in dfi.columns else dfi["close"]
                idx_l = dfi["low"] if "low" in dfi.columns else dfi["close"]
                idx_c = dfi["spot_px"] if "spot_px" in dfi.columns else dfi["close"]
                dspot = data.get("df_candles")
                if dspot is not None and not getattr(dspot, "empty", True) and "time" in dspot.columns:
                    try:
                        ds = dspot.copy()
                        ds["time"] = series_to_ist(ds["time"])
                        dfi_t = pd.to_datetime(dfi["time"])
                        if getattr(dfi_t.dt, "tz", None) is None:
                            dfi_t = dfi_t.dt.tz_localize("Asia/Kolkata")
                        left = pd.DataFrame({"time": dfi_t})
                        m = pd.merge_asof(left.sort_values("time"),
                                          ds[["time","open","high","low","close"]].sort_values("time"),
                                          on="time", direction="nearest",
                                          tolerance=pd.Timedelta(minutes=tf_min))
                        if m["close"].notna().any():
                            idx_o, idx_h, idx_l, idx_c = m["open"], m["high"], m["low"], m["close"]
                    except Exception:
                        pass
                # Futures volume, mapped to index with *per-bar* basis (not last print).
                try:
                    b = dfi["basis"].astype(float) if "basis" in dfi.columns else 0.0
                    # 3-min (or current TF) basis; also keep a 30-min step for stability
                    if "time" in dfi.columns:
                        t0 = pd.to_datetime(dfi["time"])
                        step = t0.dt.floor("30min")
                        b30 = pd.Series(b).groupby(step).transform("mean")
                        b_use = 0.65 * pd.Series(b).astype(float) + 0.35 * b30.astype(float)
                    else:
                        b_use = pd.Series(b).astype(float)
                    vp_src = pd.DataFrame({
                        "open": dfi["open"].astype(float) - b_use.values,
                        "high": dfi["high"].astype(float) - b_use.values,
                        "low": dfi["low"].astype(float) - b_use.values,
                        "close": dfi["close"].astype(float) - b_use.values,
                        "volume": dfi["volume"].astype(float) if "volume" in dfi.columns else 0.0,
                    })
                    rng = (vp_src["high"] - vp_src["low"]).replace(0, np.nan)
                    med_r = float(rng.median() or 0) or 1.0
                    hi_q = float(vp_src["high"].quantile(0.97))
                    lo_q = float(vp_src["low"].quantile(0.03))
                    keep = (rng <= 4.0 * med_r) & (vp_src["high"] <= hi_q + med_r) & (vp_src["low"] >= lo_q - med_r)
                    if keep.sum() >= 8:
                        vp_src = vp_src[keep]
                    vp2 = compute_session_volume_profile(vp_src, bin_step=2.0, prominence_factor=0.35)
                    if vp2.get("ok"):
                        vp = vp2
                    # Developing VA: last 90 min (or last third of bars)
                    if "time" in dfi.columns and len(vp_src) >= 18:
                        tlast = pd.to_datetime(dfi["time"].iloc[-1])
                        cut = tlast - pd.Timedelta(minutes=90)
                        if len(vp_src) == len(dfi):
                            dev_src = vp_src.loc[pd.to_datetime(dfi["time"]) >= cut]
                        else:
                            dev_src = vp_src.tail(max(18, len(vp_src) // 3))
                        if len(dev_src) >= 12:
                            vpd = compute_session_volume_profile(dev_src, bin_step=2.0, prominence_factor=0.35)
                            if vpd.get("ok") and isinstance(vp, dict):
                                vp = dict(vp)
                                vas = list(vp.get("vas") or [])
                                vas.append({
                                    "vah": vpd.get("vah"), "val": vpd.get("val"),
                                    "poc": vpd.get("poc"), "frac": 0.70, "tag": "dVA", "primary": False,
                                })
                                vp["vas"] = vas
                                vp["dvah"] = vpd.get("vah")
                                vp["dval"] = vpd.get("val")
                                vp["dpoc"] = vpd.get("poc")
                except Exception:
                    pass
                buy_v = np.zeros(len(dfi), dtype=float)
                sell_v = np.zeros(len(dfi), dtype=float)
                if "volume" in dfi.columns:
                    vol = dfi["volume"].astype(float).values
                    up = (dfi["close"].astype(float) >= dfi["open"].astype(float)).values if "open" in dfi.columns else np.ones(len(dfi), bool)
                    buy_v = np.where(up, vol, 0.0)
                    sell_v = np.where(~up, vol, 0.0)
                fig_stack = make_subplots(
                    rows=4, cols=3,
                    column_widths=[0.13, 0.71, 0.16],
                    row_heights=[0.52, 0.14, 0.16, 0.18],
                    shared_xaxes=False,
                    shared_yaxes=False,
                    horizontal_spacing=0.01,
                    vertical_spacing=0.016,
                    specs=[
                        [{}, {}, {}],
                        [None, {}, None],
                        [None, {}, None],
                        [None, {}, None],
                    ],
                )
                try:
                    dp = compute_delta_profile(vp_src if "vp_src" in dir() else None, bin_step=2.0)
                except Exception:
                    dp = {"ok": False}
                if not isinstance(dp, dict) or not dp.get("ok"):
                    try:
                        b_use = dfi["basis"].astype(float) if "basis" in dfi.columns else 0.0
                        vp_src = pd.DataFrame({
                            "open": dfi["open"].astype(float) - b_use,
                            "high": dfi["high"].astype(float) - b_use,
                            "low": dfi["low"].astype(float) - b_use,
                            "close": dfi["close"].astype(float) - b_use,
                            "volume": dfi["volume"].astype(float),
                        })
                        dp = compute_delta_profile(vp_src, bin_step=2.0)
                    except Exception:
                        dp = {"ok": False}
                if dp.get("ok"):
                    dm, dd = [], []
                    for m, dlt in zip(dp["mids"], dp["delta"]):
                        if y0 is not None and not (y0 <= float(m) <= y1):
                            continue
                        dm.append(float(m)); dd.append(float(dlt))
                    fig_stack.add_trace(plt_go.Bar(
                        x=dd, y=dm, orientation="h", showlegend=False, name="ΔP",
                        marker=dict(color=["#00E676" if v >= 0 else "#FF5252" for v in dd]),
                        hovertemplate="Px %{y:.0f}<br>Δ %{x:.0f}<extra></extra>",
                    ), row=1, col=1)
                # price
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["vwap_upper_idx"], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot")), row=1, col=2)
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["vwap_lower_idx"], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(255,152,0,0.08)"), row=1, col=2)
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["vwap_idx"], mode="lines", name="VWAP (idx)",
                    line=dict(color="#FF9800", width=2),
                    hoverinfo="skip"), row=1, col=2)
                if st.session_state.get("avwap_on") and "avwap_idx" in dfi.columns and dfi["avwap_idx"].notna().any():
                    fig_stack.add_trace(plt_go.Scatter(
                        x=dfi["time_str"], y=dfi["avwap_idx"], mode="lines", name="AVWAP",
                        line=dict(color="#CE93D8", width=2, dash="dash"),
                    ), row=1, col=2)
                cd = np.column_stack([
                    dfi["vwap_idx"].astype(float).values,
                    dfi["close"].astype(float).values,
                    dfi["basis"].astype(float).values,
                    dfi.get("basis_raw", dfi["basis"]).astype(float).values,
                    dfi["vwap"].astype(float).values,
                ])
                fig_stack.add_trace(plt_go.Candlestick(
                    x=dfi["time_str"], open=idx_o, high=idx_h, low=idx_l, close=idx_c,
                    name=str(Index_Name), increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
                    increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350",
                    showlegend=True,
                ), row=1, col=2)
                if st.session_state.get("big_trade_on") and "volume" in dfi.columns:
                    vol_raw = dfi["volume"].astype(float)
                    # Angel fut candles: volume is already lots (typically 10^2–10^4 / bar).
                    # Only divide by lot size if the series looks like shares.
                    med = float(vol_raw.replace(0, np.nan).median() or 0)
                    lots = vol_raw / float(LOT_SIZES.get(Index_Name, 65) or 65) if med > 20000 else vol_raw
                    thr = float(st.session_state.get("big_trade_min") or 50)
                    msk = lots >= thr
                    y_px = pd.Series(idx_c, index=dfi.index) if not hasattr(idx_c, "loc") else idx_c
                    if msk.any():
                        sizes = (10 + 20 * (lots[msk] / max(float(lots[msk].max()), 1))).clip(9, 32)
                        colr = np.where(
                            dfi.loc[msk, "close"].astype(float) >= dfi.loc[msk, "open"].astype(float),
                            "#00E676", "#FF5252",
                        ) if "open" in dfi.columns else "#FFD54F"
                        fig_stack.add_trace(plt_go.Scatter(
                            x=dfi.loc[msk, "time_str"],
                            y=y_px.loc[msk],
                            mode="markers", name=f"≥{int(thr)} lots",
                            marker=dict(size=sizes, color=colr, opacity=0.62,
                                        line=dict(width=1, color="#FFF59D")),
                            hovertemplate="%{x} · %{customdata:.0f} lots<extra>big Δ</extra>",
                            customdata=lots[msk],
                        ), row=1, col=2)
                pdec_hist = pdec_session_history(dfi) if st.session_state.get("pdec_labels_on") else []
                lv_w = data.get("levels") or {}
                spot_w = float(data.get("spot_price") or (dfi["spot_px"].iloc[-1] if "spot_px" in dfi.columns else 0) or 0)
                call_w = put_w = None
                dch = data.get("df_chain")
                if dch is None:
                    dch = df_chain if "df_chain" in dir() else None
                try:
                    if dch is not None and not getattr(dch, "empty", True) and "Strike" in dch.columns:
                        dc = dch.copy()
                        dc["Strike"] = pd.to_numeric(dc["Strike"], errors="coerce")
                        above = dc[dc["Strike"] > spot_w]
                        below = dc[dc["Strike"] < spot_w]
                        if not above.empty:
                            key = "C_OI" if "C_OI" in above.columns else ("Net_GEX_OI" if "Net_GEX_OI" in above.columns else None)
                            call_w = float(above.sort_values(key, ascending=False).iloc[0]["Strike"]) if key else float(above["Strike"].min())
                        if not below.empty:
                            key = "P_OI" if "P_OI" in below.columns else ("Net_GEX_OI" if "Net_GEX_OI" in below.columns else None)
                            put_w = float(below.sort_values(key, ascending=False).iloc[0]["Strike"]) if key else float(below["Strike"].max())
                except Exception:
                    call_w = put_w = None
                if not call_w:
                    try:
                        r = float(lv_w.get("GEX_Resistance") or 0)
                        call_w = r if r > spot_w else None
                    except Exception:
                        pass
                if not put_w:
                    try:
                        s = float(lv_w.get("GEX_Support") or 0)
                        put_w = s if s < spot_w else None
                    except Exception:
                        pass
                flip_w = lv_w.get("Zero_Gamma_Flip")
                band = max(spot_w * 0.0004, 5.0)
                try:
                    flip_w = float(flip_w)
                    if flip_w:
                        fig_stack.add_hline(y=flip_w, line_color="#FFB300", line_width=1.4,
                                            line_dash="dot", row=1, col=2)
                        fig_stack.add_annotation(
                            x=axis_times[-1] if axis_times else dfi["time_str"].iloc[-1],
                            y=flip_w, text="Flip", showarrow=False, xanchor="right",
                            font=dict(size=9, color="#FFB300"), row=1, col=2)
                except Exception:
                    pass
                if vp.get("ok"):
                    x_lab = axis_times[-1] if axis_times else dfi["time_str"].iloc[-1]
                    last_y = None
                    for lv in vp_chart_levels(vp):
                        yv = float(lv["price"])
                        if y0 is not None and y1 is not None and not (y0 <= yv <= y1):
                            continue
                        fig_stack.add_hline(
                            y=yv, line_color=lv["color"], line_width=lv["width"],
                            line_dash=lv["dash"], row=1, col=2,
                        )
                        y_txt = yv
                        if last_y is not None and abs(y_txt - last_y) < max((y1 - y0) * 0.012 if y0 is not None else 3.0, 2.0):
                            y_txt = last_y + max((y1 - y0) * 0.012 if y0 is not None else 3.0, 2.0)
                        last_y = y_txt
                        fig_stack.add_annotation(
                            x=x_lab, y=y_txt,
                            text=f"{lv['name']} {yv:.0f}",
                            showarrow=False, xanchor="right",
                            font=dict(size=8, color=lv["color"]),
                            bgcolor="rgba(14,17,23,0.35)",
                            row=1, col=2,
                        )
                pdh = pdl = None
                dspot = data.get("df_candles")
                try:
                    if dspot is not None and not getattr(dspot, "empty", True):
                        ds = dspot.copy()
                        ds["time"] = pd.to_datetime(ds["time"])
                        days = sorted(ds["time"].dt.date.unique())
                        if latest_session and len(days) >= 1:
                            prevs = [d for d in days if d < latest_session]
                            if prevs:
                                prev = ds[ds["time"].dt.date == prevs[-1]]
                                if not prev.empty:
                                    pdh = float(prev["high"].max())
                                    pdl = float(prev["low"].min())
                except Exception:
                    pdh = pdl = None
                for lvl_px, lc, name in ((pdh, "#FF5252", "PDH"), (pdl, "#00E676", "PDL")):
                    if not lvl_px:
                        continue
                    fig_stack.add_hline(y=float(lvl_px), line_color=lc, line_width=1.1,
                                        line_dash="dot", row=1, col=2)
                    fig_stack.add_annotation(
                        x=axis_times[0] if axis_times else dfi["time_str"].iloc[0],
                        y=float(lvl_px), text=name, showarrow=False, xanchor="left",
                        font=dict(size=9, color=lc), row=1, col=2)
                for wall_px, lc, name in ((call_w, "#FF5252", "Call wall"), (put_w, "#00E676", "Put wall")):
                    if not wall_px:
                        continue
                    fig_stack.add_hline(y=float(wall_px), line_color=lc, line_width=1.0,
                                        line_dash="dot", row=1, col=2)
                    fig_stack.add_annotation(
                        x=axis_times[-1] if axis_times else dfi["time_str"].iloc[-1],
                        y=float(wall_px), text=name, showarrow=False, xanchor="right",
                        font=dict(size=9, color=lc), row=1, col=2)
                last_fut = float(dfi["close"].iloc[-1])
                last_sp = float(dfi["spot_px"].iloc[-1])
                last_sp = float(last_sp) if pd.notna(last_sp) else float(data.get("spot_price") or 0)
                fig_stack.add_annotation(
                    x=dfi["time_str"].iloc[-1], y=last_fut,
                    text=f"{last_sp:.0f} ({last_fut:.0f})",
                    showarrow=False, xanchor="left", font=dict(size=11, color="#00E676"),
                    row=1, col=2,
                )
                if vp.get("ok"):
                    mids, vols, colors = [], [], []
                    for m, v in zip(vp["mids"], vp["vol"]):
                        if y0 is not None and not (y0 <= float(m) <= y1):
                            continue
                        mids.append(float(m)); vols.append(float(v))
                        colors.append("#FFD54F" if abs(m - vp["poc"]) < 1e-6 else "rgba(100,181,246,0.7)")
                    fig_stack.add_trace(plt_go.Bar(
                        x=vols, y=mids, orientation="h", showlegend=False, name="VP",
                        marker=dict(color=colors),
                        hovertemplate="Px %{y:.0f}<br>Vol %{x:.0f}<extra></extra>",
                    ), row=1, col=3)

                fig_stack.add_trace(plt_go.Bar(
                    x=dfi["time_str"], y=buy_v, name="Buy ΔV",
                    marker_color="rgba(0,230,118,0.7)", showlegend=False,
                ), row=2, col=2)
                fig_stack.add_trace(plt_go.Bar(
                    x=dfi["time_str"], y=-sell_v, name="Sell ΔV",
                    marker_color="rgba(255,82,82,0.7)", showlegend=False,
                ), row=2, col=2)
                fig_stack.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot", row=2, col=2)
                try:
                    dv_m = float(max(np.nanmax(np.abs(buy_v)), np.nanmax(np.abs(sell_v)), 1.0))
                    fig_stack.update_yaxes(range=[-dv_m * 1.25, dv_m * 1.25], row=2, col=2)
                except Exception:
                    pass
                efi_col = np.where(dfi["efi13"] >= 0, "#00E676", "#FF5252")
                fig_stack.add_trace(plt_go.Bar(
                    x=dfi["time_str"], y=dfi["efi13"], marker_color=efi_col, showlegend=False, name="EFI",
                ), row=3, col=2)
                fig_stack.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot", row=3, col=2)
                cvd_last = float(dfi["cvd"].iloc[-1])
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["cvd"].clip(lower=0),
                    mode="lines", showlegend=False, name="CVD+",
                    line=dict(color="#00E676", width=1.6),
                    fill="tozeroy", fillcolor="rgba(0,230,118,0.22)",
                ), row=4, col=2)
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["cvd"].clip(upper=0),
                    mode="lines", showlegend=False, name="CVD-",
                    line=dict(color="#FF5252", width=1.6),
                    fill="tozeroy", fillcolor="rgba(255,82,82,0.22)",
                ), row=4, col=2)
                fig_stack.add_trace(plt_go.Scatter(
                    x=dfi["time_str"], y=dfi["cvd"], mode="lines", showlegend=False, name="CVD",
                    line=dict(color="#B0BEC5", width=1.2),
                ), row=4, col=2)
                fig_stack.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot", row=4, col=2)
                tf_min_w = {"3 min": 10, "5 min": 8, "15 min": 6}.get(st.session_state.get("selected_timeframe", "5 min"), 8)
                cvd_st = cvd_price_stats(dfi, window=tf_min_w)
                if cvd_st.get("ok"):
                    colr = "#FF5252" if "BEARISH" in cvd_st["div"] else (
                        "#00E676" if "BULLISH" in cvd_st["div"] else "#FFD54F"
                    )
                    fig_stack.add_annotation(
                        text=cvd_st["div"],
                        xref="paper", yref="paper", x=0.01, y=0.02,
                        showarrow=False, font=dict(size=11, color=colr),
                        bgcolor="rgba(16,20,28,0.85)",
                        row=4, col=2,
                    )

                div_ev = scan_cvd_div_events(dfi, window=tf_min_w)
                if div_ev:
                    ylo = float(dfi["close"].min())
                    yhi = float(dfi["close"].max())
                    clo = float(dfi["cvd"].min())
                    chi = float(dfi["cvd"].max())
                    for e in div_ev:
                        tip = (
                            f"{e['div']}<br>"
                            f"Px slope {e['px_slope']:+.4f}/bar p={e['px_p']:.3f}<br>"
                            f"CVD slope {e['cvd_slope']:+.2f}/bar p={e['cvd_p']:.3f}<br>"
                            f"Spearman ρ={e.get('spearman') or 0:+.2f} p={e.get('spearman_p') or 1:.3f}<br>"
                            f"{e['note']}"
                        )
                        fig_stack.add_trace(plt_go.Scatter(
                            x=[e["t"], e["t"]], y=[ylo, yhi],
                            mode="lines+text",
                            line=dict(color="#29B6F6", width=1.4, dash="dash"),
                            text=["", "D"], textposition="top center",
                            textfont=dict(size=10, color="#29B6F6"),
                            hovertemplate=tip + "<extra></extra>",
                            showlegend=False, name="D",
                        ), row=1, col=2)
                        fig_stack.add_trace(plt_go.Scatter(
                            x=[e["t"], e["t"]], y=[clo, chi],
                            mode="lines",
                            line=dict(color="#29B6F6", width=1.2, dash="dash"),
                            hovertemplate=tip + "<extra></extra>",
                            showlegend=False, name="D",
                        ), row=4, col=2)

                def _sess_chg(src):
                    if src is None or getattr(src, "empty", True):
                        return None
                    try:
                        sess, _, _ = pick_last_nse_session(src, min_bars=5)
                    except Exception:
                        sess = src
                    if sess is None or sess.empty:
                        return None
                    o = float(sess["open"].iloc[0]); cl = float(sess["close"].iloc[-1])
                    return cl, cl - o, (cl - o) / o * 100.0 if o else 0.0

                spot_chg = _sess_chg(data.get("df_candles"))
                fut_chg = _sess_chg(df_fchart if not df_fchart.empty else data.get("df_futures"))
                trig = scores.get("dir_trigger") if isinstance(scores, dict) else {}
                st.session_state["_last_dfi"] = dfi
                st.session_state["_last_scores"] = scores if isinstance(scores, dict) else {}
                micro = classify_microstructure(dfi)
                candle_pat = detect_candle_pattern(dfi)
                raw_act = micro.get("action") if isinstance(micro, dict) else ""
                filt_act, filt_why = confirm_pdec_with_candle(raw_act, candle_pat)
                if isinstance(micro, dict):
                    micro["action_raw"] = raw_act
                    micro["action"] = filt_act
                    micro["candle"] = candle_pat.get("name")
                    micro["candle_why"] = filt_why
                flow_pb = classify_flow_playbook(dfi, data)
                try:
                    process_telegram_alerts(data, dfi, scores if isinstance(scores, dict) else {},
                                            micro, flow_pb, cvd_st if "cvd_st" in dir() else {},
                                            index_name=Index_Name)
                except Exception:
                    pass

                xr = [-0.5, max(len(axis_times) - 0.5, 0.5)]
                fig_stack.update_xaxes(rangeslider_visible=False)
                fig_stack.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    height=820, margin=dict(l=40, r=6, t=8, b=18),
                    legend=dict(orientation="h", yanchor="top", y=1.0, x=0.0, font=dict(size=10),
                                bgcolor="rgba(14,17,23,0.4)"),
                    hovermode="x unified", bargap=0.15,
                    xaxis_rangeslider_visible=False,
                )
                fig_stack.update_yaxes(range=[y0, y1], tickformat="d", type="linear", showticklabels=True, row=1, col=1)
                fig_stack.update_yaxes(range=[y0, y1], tickformat="d", type="linear", row=1, col=2)
                fig_stack.update_yaxes(range=[y0, y1], type="linear", showticklabels=False, row=1, col=3)
                fig_stack.update_xaxes(type="linear", showticklabels=True, showgrid=False, row=1, col=1)
                fig_stack.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                                       range=xr, showticklabels=False, row=1, col=2)

                if st.session_state.get("pdec_labels_on") and pdec_hist:
                    hi_s = pd.to_numeric(idx_h, errors="coerce")
                    peak = float(hi_s.max()) if hi_s.notna().any() else float(y1 or 0)
                    rng = float(hi_s.max() - hi_s.min()) if hi_s.notna().any() else 20.0
                    y_lab = peak + max(rng * 0.04, 8.0)
                    y1 = max(float(y1 or peak), y_lab + rng * 0.10)
                    fig_stack.update_yaxes(range=[y0, y1], row=1, col=2)
                    prev = None
                    nlab = 0
                    for rec in pdec_hist:
                        act = str(rec.get("action") or "")
                        if (not act) or ("NO ENTRY" in act.upper()):
                            continue
                        if act == prev:
                            continue
                        prev = act
                        xt = str(rec.get("t") or "")
                        if len(xt) > 5:
                            xt = xt[:5]
                        fig_stack.add_annotation(
                            x=xt, y=y_lab, text=act[:28],
                            showarrow=False, textangle=-90,
                            xanchor="center", yanchor="bottom",
                            font=dict(size=9, color="#FFF59D"),
                            bgcolor="rgba(20,24,32,0.55)",
                            row=1, col=2,
                        )
                        nlab += 1
                    st.session_state["_va_nlab"] = nlab
                fig_stack.update_xaxes(type="linear", showticklabels=False, showgrid=False, row=1, col=3)
                fig_stack.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                                       range=xr, showticklabels=False, row=2, col=2)
                fig_stack.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                                       range=xr, showticklabels=False, row=3, col=2)
                fig_stack.update_xaxes(type="category", categoryorder="array", categoryarray=axis_times,
                                       range=xr, nticks=8, row=4, col=2)
                fig_stack.update_yaxes(tickfont=dict(size=8), title_text="ΔV", row=2, col=2)
                fig_stack.update_yaxes(tickfont=dict(size=8), title_text="EFI", row=3, col=2)
                fig_stack.update_yaxes(tickfont=dict(size=8), title_text="CVD", row=4, col=2)
                if "micro" not in dir() or not isinstance(micro, dict):
                    micro = classify_microstructure(dfi)
                    candle_pat = detect_candle_pattern(dfi)
                    raw_act = micro.get("action") if isinstance(micro, dict) else ""
                    filt_act, filt_why = confirm_pdec_with_candle(raw_act, candle_pat)
                    if isinstance(micro, dict):
                        micro["action_raw"] = raw_act
                        micro["action"] = filt_act
                        micro["candle"] = candle_pat.get("name")
                        micro["candle_why"] = filt_why
                atm_k = data.get("atm_strike")
                flow_choice = st.radio(
                    "Tape",
                    [
                        "1 · Index / ΔV / EFI / CVD",
                        "2 · DEX / Premium",
                        "3 · Δ Footprint",
                        f"4 · ATM CE {atm_k:.0f}" if atm_k else "4 · ATM CE",
                        f"5 · ATM PE {atm_k:.0f}" if atm_k else "5 · ATM PE",
                    ],
                    horizontal=True,
                    key="flow_tab_radio",
                    label_visibility="collapsed",
                )
                tab_flow = tab_dex = tab_fp = tab_atm_ce = tab_atm_pe = None
                if str(flow_choice).startswith("1"):
                    st.markdown("<div class='chart-card'><div class='card-title'>Futures &amp; session flow</div>", unsafe_allow_html=True)
                    if isinstance(micro, dict) and micro.get("candle"):
                        st.caption(
                            f"VA · {micro.get('regime','')} · {micro.get('model','')} · raw {micro.get('action_raw') or '—'} → "
                            f"{micro.get('action')} · {micro.get('candle_why') or ''}"
                        )
                    _fsig = (
                        str(Index_Name),
                        str(st.session_state.get("selected_timeframe")),
                        str(dfi["time"].iloc[-1]) if len(dfi) else "",
                        float(dfi["close"].iloc[-1]) if len(dfi) else 0.0,
                        int(len(dfi)),
                        bool(st.session_state.get("pdec_labels_on")),
                    )
                    if st.session_state.get("_fig_sig") == _fsig and st.session_state.get("_fig_stack") is not None:
                        fig_stack = st.session_state["_fig_stack"]
                    else:
                        st.session_state["_fig_stack"] = fig_stack
                        st.session_state["_fig_sig"] = _fsig
                    st.plotly_chart(fig_stack, use_container_width=True)
                    if pdec_hist:
                        with st.expander(f"VA session log ({len(pdec_hist)} bars)", expanded=False):
                            lines = ["| Time | P Δ E C | Action |", "|---|---|---|"]
                            for rec in pdec_hist:
                                lines.append(f"| {rec['t']} | {rec['glyphs']} | {rec['action']} |")
                            st.markdown("\n".join(lines))
                    if st.session_state.get("avwap_on"):
                        opts_t = list(dfi["time_str"]) if "time_str" in dfi.columns else []
                        if opts_t:
                            cur = st.session_state.get("avwap_time")
                            if cur not in opts_t:
                                cur = opts_t[0]
                            pick = st.selectbox(
                                "AVWAP anchor time (one at a time)",
                                opts_t, index=opts_t.index(cur), key="avwap_pick",
                            )
                            if pick != st.session_state.get("avwap_time"):
                                st.session_state["avwap_time"] = pick
                                st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                elif str(flow_choice).startswith("3"):
                    st.markdown("<div class='chart-card'><div class='card-title'>Δ Footprint + top-5 book</div>", unsafe_allow_html=True)
                    st.caption("Top 5 = **resting** futures book (SmartAPI FULL depth), not executed lots. Bar heatmap is still close-location volume.")
                    b5 = st.session_state.get("book5_hist") or []
                    last = b5[-1] if b5 else {}
                    cbook, cmap = st.columns([0.22, 0.78])
                    with cbook:
                        bids = last.get("bids") or []
                        asks = list(reversed(last.get("asks") or []))
                        rows = ["| Side | Px | Qty |", "|---|---:|---:|"]
                        for a in asks:
                            rows.append(f"| ASK | {a.get('px'):.1f} | {a.get('qty'):.0f} |")
                        rows.append(f"| LTP | {float(last.get('ltp') or 0):.1f} |  |")
                        for b in bids:
                            rows.append(f"| BID | {b.get('px'):.1f} | {b.get('qty'):.0f} |")
                        st.markdown("\n".join(rows) if last else "_No book yet — Auto-Refresh in market hours._")
                    with cmap:
                        if len(b5) >= 3:
                            prices, zs, ts = [], [], []
                            for rec in b5[-180:]:
                                ts.append(rec.get("t"))
                            pxset = sorted({lv["px"] for rec in b5[-180:] for lv in (rec.get("bids") or []) + (rec.get("asks") or []) if lv.get("px")})
                            if pxset:
                                grid = np.zeros((len(pxset), len(b5[-180:])))
                                pxi = {p: i for i, p in enumerate(pxset)}
                                for j, rec in enumerate(b5[-180:]):
                                    for lv in rec.get("bids") or []:
                                        if lv["px"] in pxi:
                                            grid[pxi[lv["px"]], j] += float(lv["qty"] or 0)
                                    for lv in rec.get("asks") or []:
                                        if lv["px"] in pxi:
                                            grid[pxi[lv["px"]], j] -= float(lv["qty"] or 0)
                                fig_b = plt_go.Figure(plt_go.Heatmap(
                                    x=[r.get("t") for r in b5[-180:]], y=pxset, z=grid, zmid=0,
                                    colorscale=[[0, "#B71C1C"], [0.5, "#1B1E24"], [1, "#1B5E20"]],
                                    hovertemplate="%{x} · %{y:.1f}<br>qty signed %{z:.0f}<extra>book5</extra>",
                                ))
                                fig_b.update_layout(template="plotly_dark", height=280,
                                    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                    margin=dict(l=40, r=8, t=8, b=8))
                                st.plotly_chart(fig_b, use_container_width=True)
                        fig_fp = build_delta_footprint_figure(dfi, st.session_state.get("_axis_times"))
                        if isinstance(fig_fp, tuple):
                            fig_fp = fig_fp[0]
                        evs = st.session_state.get("_fp_abs_evs") or []
                        n_bid = sum(1 for e in evs if e.get("side", 0) > 0)
                        n_off = sum(1 for e in evs if e.get("side", 0) < 0)
                        st.caption(
                            f"Triggers this session: BID ABS {n_bid} · OFFER ABS {n_off}. "
                            "Need sweep z≥1.6, shelf z≥1.4, close in outer 28%, then 8-bar cooldown per side. "
                            "No mark = no bar cleared the gate."
                        )
                        if fig_fp is not None:
                            st.plotly_chart(fig_fp, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                def _render_atm_tab(tok, lab):

                    if st.session_state.get("intel_refresh") or (
                        st.session_state.get("enable_main_refresh") and not st.session_state.get("atm_live_ok")
                    ):
                        st.caption("ATM tape paused during 5s index refresh. Enable “ATM live” to fetch.")
                        cached = st.session_state.get(f"_atm_fig_{lab}")
                        if cached:
                            st.plotly_chart(cached, use_container_width=True)
                        return
                    try:
                        api = get_smart_api_client()
                        tf_lab = st.session_state.get("selected_timeframe", "5 min")
                        api_int, _lb = interval_mapping.get(tf_lab, ("FIVE_MINUTE", 15))
                        exch_opt = data.get("opt_exchange") or Exchange
                        if not api or not tok:
                            st.caption("ATM token unavailable this cycle.")
                            return
                        lb = 0 if str(tf_lab).startswith(("1 ", "2 ")) else 1
                        dfo, _fb = fetch_candles_with_holiday_fallback(
                            api, str(tok), exch_opt, api_int, lb, f"{Index_Name}_{lab}"
                        )
                        used_tf = tf_lab
                        if dfo is None or dfo.empty or len(dfo) < 25:
                            dfo, _fb = fetch_candles_with_holiday_fallback(
                                api, str(tok), exch_opt, "THREE_MINUTE", 1, f"{Index_Name}_{lab}_3m"
                            )
                            used_tf = "3 min"
                            st.caption("ATM 1/2-min tape thin on SmartAPI — showing 3-min option candles.")
                        fig_o, act_o = _option_session_figure(
                            dfo, f"ATM {lab}", Index_Name, used_tf,
                            st.session_state.get("_axis_times"),
                        )
                        st.caption(f"VA · {act_o}")
                        if fig_o is not None:
                            st.plotly_chart(fig_o, use_container_width=True)
                    except Exception as e:
                        st.caption(f"ATM {lab} unavailable.")
                if str(flow_choice).startswith("4"):
                    st.markdown("<div class='chart-card'><div class='card-title'>ATM CE</div>", unsafe_allow_html=True)
                    _render_atm_tab(data.get("atm_ce_token"), "CE")
                    st.markdown("</div>", unsafe_allow_html=True)
                if str(flow_choice).startswith("5"):
                    st.markdown("<div class='chart-card'><div class='card-title'>ATM PE</div>", unsafe_allow_html=True)
                    _render_atm_tab(data.get("atm_pe_token"), "PE")
                    st.markdown("</div>", unsafe_allow_html=True)
                if str(flow_choice).startswith("2"):

                    sess_day = latest_session or _ist_now().date()
                    tape = load_flow_tape(Index_Name, sess_day) or list(st.session_state.get("flow_tape") or [])
                    fig_dex = make_subplots(
                        rows=3, cols=2,
                        column_widths=[0.84, 0.16],
                        row_heights=[0.46, 0.27, 0.27],
                        shared_xaxes=False,
                        vertical_spacing=0.06,
                        horizontal_spacing=0.01,
                        specs=[[{}, {}], [{}, None], [{}, None]],
                    )
                    if "vwap_upper_idx" in dfi.columns:
                        fig_dex.add_trace(plt_go.Scatter(
                            x=dfi["time_str"], y=dfi["vwap_upper_idx"], mode="lines", showlegend=False, hoverinfo="skip",
                            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot")), row=1, col=1)
                        fig_dex.add_trace(plt_go.Scatter(
                            x=dfi["time_str"], y=dfi["vwap_lower_idx"], mode="lines", showlegend=False, hoverinfo="skip",
                            line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
                            fill="tonexty", fillcolor="rgba(255,152,0,0.08)"), row=1, col=1)
                    fig_dex.add_trace(plt_go.Scatter(
                        x=dfi["time_str"], y=dfi.get("vwap_idx", dfi["vwap"]), mode="lines", name="VWAP (idx)",
                        line=dict(color="#FF9800", width=2), hoverinfo="skip"), row=1, col=1)
                    fig_dex.add_trace(plt_go.Scatter(
                        x=dfi["time_str"], y=dfi["spot_px"], mode="lines", name=str(Index_Name),
                        line=dict(color="#2196F3", width=2)), row=1, col=1)
                    if vp.get("ok"):
                        fig_dex.add_trace(plt_go.Bar(
                            x=vols if vp.get("ok") else [], y=mids if vp.get("ok") else [],
                            orientation="h", showlegend=False,
                            marker=dict(color=colors if vp.get("ok") else "#64B5F6"),
                        ), row=1, col=2)
                    if tape and fut_times:
                        # Map snaps onto the same category axis as futures (HH:MM)
                        dex_map = {str(t.get("ts"))[-5:]: t for t in tape}
                        xs = list(fut_times)
                        dex_y, pc_y, pp_y = [], [], []
                        for ts in xs:
                            key = str(ts)[-5:]
                            rec = dex_map.get(key)
                            dex_y.append(rec["dex"] if rec else None)
                            pc_y.append((rec["prem_c"] / 1e7) if rec else None)
                            pp_y.append((rec["prem_p"] / 1e7) if rec else None)
                        s = pd.Series(dex_y, dtype=float)
                        ema = s.ewm(span=8, min_periods=1, adjust=False).mean()
                        bar_c = ["#00E676" if (v is not None and v >= 0) else "#FF5252" for v in dex_y]
                        fig_dex.add_trace(plt_go.Bar(
                            x=xs, y=dex_y, name="DEX", marker_color=bar_c, opacity=0.75, showlegend=False,
                        ), row=2, col=1)
                        fig_dex.add_trace(plt_go.Scatter(
                            x=xs, y=ema, name="DEX EMA8",
                            line=dict(color="#FFD54F", width=2),
                        ), row=2, col=1)
                        fig_dex.add_trace(plt_go.Scatter(
                            x=xs, y=pc_y, name="Call prem Cr",
                            line=dict(color="#00E676", width=2), connectgaps=False,
                        ), row=3, col=1)
                        fig_dex.add_trace(plt_go.Scatter(
                            x=xs, y=pp_y, name="Put prem Cr",
                            line=dict(color="#FF5252", width=2), connectgaps=False,
                        ), row=3, col=1)
                    fig_dex.add_hline(y=0, line_width=1, line_color="#FFFFFF", line_dash="dot", row=2, col=1)
                    fig_dex.update_layout(
                        template="plotly_dark", paper_bgcolor="#11151C", plot_bgcolor="#0E1117",
                        height=820, margin=dict(l=40, r=6, t=8, b=18),
                        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
                        hovermode="x unified",
                    )
                    fig_dex.update_xaxes(type="category", categoryorder="array",
                                        categoryarray=axis_times, range=xr,
                                        showticklabels=False, row=1, col=1)
                    fig_dex.update_xaxes(type="category", categoryorder="array",
                                        categoryarray=axis_times, range=xr,
                                        showticklabels=False, row=2, col=1)
                    fig_dex.update_xaxes(type="category", categoryorder="array",
                                        categoryarray=axis_times, range=xr,
                                        nticks=8, row=3, col=1)
                    fig_dex.update_yaxes(title_text="", tickfont=dict(size=8), row=2, col=1)
                    fig_dex.update_yaxes(title_text="", tickfont=dict(size=8), row=3, col=1)
                    st.markdown("<div class='chart-card'><div class='card-title'>DEX flow &amp; net premium</div>", unsafe_allow_html=True)
                    if tape:
                        st.caption(f"DEX / premium time · {sess_day} · {len(tape)} snaps · x = futures axis")
                    else:
                        st.caption("DEX / premium empty until Auto-Refresh stores snaps in market hours. No strike fallback.")
                    st.plotly_chart(fig_dex, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                cap = (f"{st.session_state.get('selected_timeframe','?')} · "
                       f"{len(dfi)} bars · Fut {latest_fut:,.1f} · VWAP {latest_vwap:,.1f} · CVD {cvd_last:,.0f}")
                if cvd_st.get("ok"):
                    cap += (f" · Px slope {cvd_st['px_slope']:+.3f}/bar p={cvd_st['px_p']:.3f}"
                            f" · CVD slope {cvd_st['cvd_slope']:+.1f}/bar p={cvd_st['cvd_p']:.3f}"
                            f" · {cvd_st['div']}")
                    heading_ribbon(
                        f"CVD vs Px · {cvd_st['div']}",
                        f"Window last {cvd_st['win']} bars of <b>this TF only</b>. "
                        f"OLS slope vs bar index. Significant = p&lt;0.05.<br>"
                        f"Price slope {cvd_st['px_slope']:+.4f} /bar · p={cvd_st['px_p']:.3f}<br>"
                        f"CVD slope {cvd_st['cvd_slope']:+.2f} /bar · p={cvd_st['cvd_p']:.3f}<br>"
                        f"Spearman ρ={cvd_st.get('spearman') or 0:+.2f} p={cvd_st.get('spearman_p') or 1:.3f}<br>"
                        f"{cvd_st['div_note']}<br>"
                        "Full divergence = opposing significant slopes <b>and</b> matching swing (HH/LH or LL/HL). "
                        "Slope-only is a watch. 15-min sessions have few bars — p-values stay weak.",
                    )
                if vp.get("ok"):
                    bits = [f"POC {vp['poc']:.0f}"]
                    for lv in vp_chart_levels(vp):
                        if lv["name"] != "POC":
                            bits.append(f"{lv['name']} {lv['price']:.0f}")
                    cap += " · " + " · ".join(bits)
                st.caption(cap)
                def _chip(body, tip, color="#00E676"):
                    return (
                        f"<div class='micro-hover micro-chip'>"
                        f"<div style='color:{color};font-weight:800;font-size:12px;line-height:1.3;text-align:center;'>{body}</div>"
                        f"<div class='micro-tip'>{tip}</div></div>"
                    )
                chips = []
                sc = f"{Index_Name} {spot_chg[0]:,.0f} {spot_chg[1]:+.0f} {spot_chg[2]:+.2f}%" if spot_chg else f"{Index_Name} —"
                fc = f"FUT {fut_chg[0]:,.0f} {fut_chg[1]:+.0f} {fut_chg[2]:+.2f}%" if fut_chg else "FUT —"
                chips.append(_chip(
                    f"{sc}<br>{fc}",
                    f"{Index_Name} spot and near-month futures session open→close.",
                    "#00E676" if (spot_chg and spot_chg[2] >= 0) else "#FF5252",
                ))
                act = (micro or {}).get("action") or ""
                if micro.get("ok") and act and "NO ENTRY" not in str(act).upper():
                    chips.append(_chip(
                        f"VA<br>{act}",
                        f"<b>VA</b><br>{micro.get('micro','')}<br>{micro.get('efi_note','')}<br>"
                        f"regime {micro.get('regime','')} · {micro.get('model','')}",
                        "#00E676",
                    ))
                ch1, ch2 = st.columns([0.18, 0.82])
                with ch1:
                    st.markdown("<div style='margin-top:4px;'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
                with ch2:
                    st.empty()
                if not st.session_state.get("intel_refresh"):
                    b1, b2 = st.columns([0.50, 0.50])
                    with b1:
                        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                        render_liquidity_delta_panel(data, df_fchart, Index_Name, compact=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                        render_block_tape(data, Index_Name)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Futures / VWAP not available.")

        # Delta-GEX (selected) | Heatmap
        # Multi-exp Delta-GEX   | VEX/CEX
        # IV Skew 50%           | Z-Scores 50%
        # Basket Greeks then Strategy Basket Legs
        if not df_chain.empty and lvls:
            st.markdown("---")
            d_left, d_right = st.columns([0.40, 0.60])
            with d_left:
                heading_ribbon(
                    "⚡ VEX / CEX Profile",
                    "<b>VEX (vanna)</b> ≈ −pdf(d1)×d2/σ × OI × lot — dealer vanna vs spot/IV.<br>"
                    "<b>CEX (charm)</b> ≈ ∂Δ/∂t × OI × lot — delta decay into expiry.<br>"
                    "Positive vanna/charm often supports a pin; large negative supports acceleration.",
                )
                fig_vex_cex = make_subplots(specs=[[{"secondary_y": True}]])
                fig_vex_cex.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["VEX"], name="VEX", marker_color="#00E676", opacity=0.75, width=20), secondary_y=False)
                fig_vex_cex.add_trace(plt_go.Scatter(x=df_chain["Strike"], y=df_chain["CEX"], name="CEX", line=dict(color="#2196F3", width=2), mode="lines+markers", marker=dict(size=4)), secondary_y=True)
                fig_vex_cex.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
                fig_vex_cex.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot", annotation_font_size=10)
                vex_range, cex_range = calculate_synced_ranges(df_chain["VEX"].astype(float), df_chain["CEX"].astype(float))
                fig_vex_cex.update_layout(
                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    height=520, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                )
                fig_vex_cex.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
                fig_vex_cex.update_yaxes(range=vex_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF")
                fig_vex_cex.update_yaxes(range=cex_range, secondary_y=True, showgrid=False)
                st.plotly_chart(fig_vex_cex, use_container_width=True)

            with d_right:
                if st.session_state.get("intel_refresh"):
                    st.caption("Intelligent refresh — GEX heatmap skipped.")
                else:
                    render_delta_gex_heatmap(
                        data, Index_Name, selected_expiry_str,
                        st.session_state.get("heatmap_timeframe", "5 min"),
                    )

            # IV Skew 50% | Z-Scores 50%
            if st.session_state.get("intel_refresh"):
                st.caption("Intelligent refresh — IV / Vanna / GEX / book skipped.")
                skew_col, z_col = None, None
            else:
                st.markdown("---")
                skew_col, z_col = st.columns([0.50, 0.50])
            if skew_col is not None:
              with skew_col:
                heading_ribbon(f"📉 IV Skew ({selected_expiry_str})",
            "IV from LTP via Black-Scholes (Brent). OTM tab = puts below spot + calls at/above. "
            "Raw tab = full CE and PE IV curves. VIX tab = India VIX 5-min, last 3 sessions.")
                smart_api = get_smart_api_client()
                if smart_api and not df_master.empty:
                    latest_spot = data["spot_price"]
                    df_chain_iv = fetch_and_compute_full_chain_iv(
                        smart_api, df_master, Index_Name, Exchange, selected_expiry_str,
                        latest_spot, rate_param, strikes_below=strikes_below, strikes_above=strikes_above,
                    )
                    if not df_chain_iv.empty:
                        tab1, tab2, tab3 = st.tabs(["OTM Skew (Puts & Calls)", "Both Raw Curves (CE vs PE)", "INDIA VIX"])
                        with tab1:
                            df_skew = get_clean_otm_skew(df_chain_iv, latest_spot)
                            fig_skew = px.line(df_skew, x="Strike", y="IV_%", markers=True, color_discrete_sequence=["#00bfff"], hover_data=["Option_Type", "LTP"])
                            fig_skew.add_vline(x=latest_spot, line_dash="dash", line_color="white", annotation_text="Spot", annotation_font_size=10)
                            fig_skew.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                            st.plotly_chart(fig_skew, use_container_width=True)
                        with tab2:
                            fig_raw = px.line(df_chain_iv, x="Strike", y="IV_%", color="Option_Type", markers=True, color_discrete_map={"CE": "#00cc96", "PE": "#ff4136"}, hover_data=["LTP"])
                            fig_raw.add_vline(x=latest_spot, line_dash="dash", line_color="white", annotation_text="Spot", annotation_font_size=10)
                            fig_raw.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
                            st.plotly_chart(fig_raw, use_container_width=True)
                        with tab3:
                            df_vix, vix_pct, vix_day = fetch_india_vix_sessions(smart_api)
                            if df_vix.empty:
                                st.info("India VIX candles unavailable.")
                            else:
                                colr = "#00E676" if (vix_pct or 0) >= 0 else "#FF5252"
                                st.markdown(
                                    f"<div style='font-size:18px;font-weight:800;color:{colr};'>"
                                    f"INDIA VIX {float(df_vix['close'].iloc[-1]):.2f}  "
                                    f"{vix_pct:+.2f}% <span style='font-size:12px;font-weight:500;color:#AAA;'>"
                                    f"({vix_day} session)</span></div>",
                                    unsafe_allow_html=True
                                )
                                fig_vix = plt_go.Figure()
                                fig_vix.add_trace(plt_go.Scatter(
                                    x=df_vix["time_str"], y=df_vix["close"], mode="lines",
                                    line=dict(color="#FFD54F", width=2), name="VIX",
                                ))
                                fig_vix.update_layout(
                                    template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                                    height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                )
                                fig_vix.update_xaxes(type="category", nticks=8)
                                fig_vix.update_yaxes(title="VIX")
                                st.plotly_chart(fig_vix, use_container_width=True)
                                st.caption("% is last session open→close. Chart = last 3 trading sessions.")
                    else:
                        st.info("Skew data unavailable.")
                else:
                    st.info("API session needed for skew.")
            if z_col is not None:
              with z_col:
                zscore_analysis_fragment(mode="highlights")

        elif not df_chain.empty:
            st.info("Key levels not available – GEX charts skipped.")

        # Basket Greeks ABOVE Strategy Basket Legs
        st.markdown("---")
        render_basket_metrics_block(data)
        render_basket_table_fullwidth(data)


live_dashboard_fragment()

# --- Raw Z-Score details ---
if st.session_state.get("app_view", "default") == "default":
    st.markdown("---")
    zscore_analysis_fragment(mode="raw")

# Clear loading status once full UI has rendered
try:
    clear_load_status()
except Exception:
    pass
