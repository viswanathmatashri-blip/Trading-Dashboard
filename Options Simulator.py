import os
import logging
import warnings
import time
import datetime
import math
import json
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
.block-container { padding-top: 0.6rem !important; padding-bottom: 0.6rem !important; }
header[data-testid="stHeader"] { background-color: rgba(0, 0, 0, 0) !important; }
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
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

API_KEY = os.getenv("API_KEY", "")
CLIENT_CODE = os.getenv("CLIENT_CODE", "")
PIN = os.getenv("PIN", "")
TOTP_SECRET = os.getenv("TOTP_SECRET", "")

# Initialise Session State Variables
if "basket_legs" not in st.session_state:
    st.session_state["basket_legs"] = []
if "selected_timeframe" not in st.session_state:
    st.session_state["selected_timeframe"] = "5 min"
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
    "SENSEX": 20
}

INDEX_TOKEN_MAP = {
    "NIFTY": ("99926000", "NSE", "NFO"),
    "BANKNIFTY": ("99926009", "NSE", "NFO"),
    "FINNIFTY": ("99926037", "NSE", "NFO"),
    "MIDCPNIFTY": ("99926074", "NSE", "NFO"),
    "SENSEX": ("99919000", "BSE", "BFO")
}

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
def fetch_candles_with_holiday_fallback(smart_api, spot_token, exchange, api_interval, lookback_days=15):
    ist_tz = pytz.timezone("Asia/Kolkata")
    now_dt = datetime.datetime.now(ist_tz)
    
    for offset in range(0, 10):
        target_to = now_dt - datetime.timedelta(days=offset)
        target_from = target_to - datetime.timedelta(days=lookback_days)
        
        candle_param = {
            "exchange": exchange,
            "symboltoken": spot_token,
            "interval": api_interval,
            "fromdate": target_from.strftime("%Y-%m-%d 09:15"),
            "todate": target_to.strftime("%Y-%m-%d 15:30")
        }
        
        candle_res = safe_api_call(smart_api.getCandleData, candle_param)
        time.sleep(0.40)
        if candle_res and candle_res.get("status") and candle_res.get("data"):
            df_candles = pd.DataFrame(candle_res["data"], columns=["time", "open", "high", "low", "close", "volume"])
            df_candles[["open", "high", "low", "close", "volume"]] = df_candles[["open", "high", "low", "close", "volume"]].astype(float)
            if not df_candles.empty:
                return compute_technical_indicators(df_candles), offset > 0

    return pd.DataFrame(), False


def get_near_month_futures_token(df_scrip_master, index_name, fut_exch):
    """Return token of the nearest active futures contract."""
    try:
        ist_now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
        fut_scrips = df_scrip_master[
            (df_scrip_master["exch_seg"] == fut_exch) &
            (df_scrip_master["name"] == index_name) &
            (df_scrip_master["instrumenttype"].isin(["FUTIDX", "FUTSTK"]))
        ].copy()
        fut_scrips["expiry_dt"] = pd.to_datetime(fut_scrips["expiry"], format="%d%b%Y", errors="coerce")
        active = fut_scrips[fut_scrips["expiry_dt"].dt.date >= ist_now.date()].sort_values("expiry_dt")
        if not active.empty:
            return str(active.iloc[0]["token"]), active.iloc[0]["expiry"]
    except Exception:
        pass
    return None, None


def fetch_futures_candles_with_vwap(smart_api, index_name, df_scrip_master, api_interval, lookback_days=15):
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

    market_open = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    is_weekday = now_dt.weekday() < 5
    currently_closed = (not is_weekday) or (now_dt < market_open or now_dt > market_close)

    best_df = pd.DataFrame()
    best_offset = 0

    for offset in range(0, 12):
        target_to = now_dt - datetime.timedelta(days=offset)
        target_from = target_to - datetime.timedelta(days=lookback_days)

        candle_param = {
            "exchange": fut_exch,
            "symboltoken": str(fut_token),
            "interval": api_interval,
            "fromdate": target_from.strftime("%Y-%m-%d 09:15"),
            "todate": target_to.strftime("%Y-%m-%d 15:30")
        }

        candle_res = safe_api_call(smart_api.getCandleData, candle_param)
        time.sleep(0.35)

        if not (candle_res and candle_res.get("status") and candle_res.get("data")):
            continue

        df = pd.DataFrame(candle_res["data"], columns=["time", "open", "high", "low", "close", "volume"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        if df.empty or len(df) < 5:
            continue

        df = compute_technical_indicators(df)
        best_df = df
        best_offset = offset
        break

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

    last_bar_time = pd.to_datetime(best_df["time"].iloc[-1])
    session_date_str = last_bar_time.strftime("%d-%b-%Y")

    if currently_closed or best_offset > 0:
        fallback_msg = (
            f"⚠️ Market is currently closed. Showing last available session: "
            f"**{session_date_str}** (fallback)"
        )
        is_fallback = True
    else:
        fallback_msg = f"Live session • {session_date_str}"
        is_fallback = False

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

with st.sidebar.expander("1. Market Parameters", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        Index_Name = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"])
    
    default_token, spot_exchange, Exchange = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))
    
    with c2:
        st.text_input("Exchange", value=Exchange, disabled=True)

    rate_param = st.number_input("Risk Free Rate (r)", min_value=0.0, max_value=0.15, value=0.07, step=0.01)

    df_options = df_master[
        (df_master["exch_seg"] == Exchange)
        & (df_master["name"] == Index_Name)
        & (df_master["instrumenttype"].isin(["OPTIDX", "OPTSTK"]))
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

with st.sidebar.expander("2. Build Strategy Basket", expanded=True):
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

run_btn = st.sidebar.button("🚀 Fetch Chain & Greeks", use_container_width=True)

interval_mapping = {
    "1 min": ("ONE_MINUTE", 7),
    "3 min": ("THREE_MINUTE", 10),
    "5 min": ("FIVE_MINUTE", 15),
    "10 min": ("TEN_MINUTE", 20),
    "15 min": ("FIFTEEN_MINUTE", 30)
}

# --- SECURE SESSION HANDLER (CACHE PERSISTENT TO PREVENT RATE LIMITS) ---
@st.cache_resource(ttl=3600, show_spinner=False)
def get_smart_api_client():
    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        return None

    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp_token = pyotp.TOTP(TOTP_SECRET).now()
        session = safe_api_call(smart_api.generateSession, CLIENT_CODE, PIN, totp_token)

        if session and session.get("status"):
            return smart_api
    except Exception:
        pass
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
        st.error("Missing credentials or failed to generate SmartAPI session!")
        return None

    try:
        update_p(0.20, f"Fetching Live Spot & Volatility for {Index_Name}...")
        spot_token, spot_exch, opt_exch = INDEX_TOKEN_MAP.get(Index_Name, ("99926000", "NSE", "NFO"))

        spot_resp = safe_api_call(smart_api.ltpData, exchange=spot_exch, tradingsymbol=Index_Name, symboltoken=spot_token)
        time.sleep(0.40)
        spot_price = float(spot_resp["data"]["ltp"]) if spot_resp and spot_resp.get("status") and spot_resp.get("data") else 80000.0 if Index_Name == "SENSEX" else 24500.0

        index_hv = VolatilityEngine.calculate_hv(smart_api, spot_token, spot_exch, days=hv_days)

        update_p(0.35, "Fetching Historical Underlying Candles (with Holiday Fallback)...")
        api_interval, lookback_days = interval_mapping.get(selected_interval_label, ("FIVE_MINUTE", 15))

        df_candles, is_holiday_fallback = fetch_candles_with_holiday_fallback(
            smart_api, spot_token, spot_exch, api_interval, lookback_days
        )

        # ----- Near-month Futures candles + real VWAP -----
        update_p(0.42, "Fetching Near-Month Futures candles for VWAP...")
        df_futures, fut_is_fallback, basis_info, fut_fallback_msg = fetch_futures_candles_with_vwap(
            smart_api, Index_Name, df_master, api_interval, lookback_days
        )

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
            "timestamp": now_dt.strftime("%d-%b-%Y %H:%M:%S IST")
        }

    except Exception:
        if p_bar: p_bar.empty()
        if p_status: p_status.empty()
        st.warning("API Rate limit / sync notice: Retrying on next cycle...")
        return None

if run_btn or "data_store" not in st.session_state:
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
        st.markdown(f"<span style='font-weight:700;color:#00E676;font-size:14px;'>📊 Z-Scores ({Index_Name})</span>", unsafe_allow_html=True)
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
    st.markdown(f"<span style='font-weight:700;color:#00E676;font-size:14px;'>🏛️ Institutional Order Flow ({Index_Name})</span>", unsafe_allow_html=True)
    st.caption("High-conviction flow · Premium ≥ ₹10L or Vol ≥ 500 lots")

    if "data_store" not in st.session_state or not st.session_state["data_store"].get("chain_results"):
        st.info("No option chain data available. Run the main fetch process to enable scanning.")
        return

    chain_data = st.session_state["data_store"]["chain_results"]
    spot_price = st.session_state["data_store"].get("spot_price", 0.0)
    lot_size = LOT_SIZES.get(Index_Name, 65)

    scanned_trades = []

    # Calculate average volume across current chain strikes for unusual volume ratio calculation
    all_volumes = [r["C_Vol"] for r in chain_data] + [r["P_Vol"] for r in chain_data]
    avg_daily_vol = np.mean(all_volumes) if all_volumes and np.mean(all_volumes) > 0 else 1.0

    for row in chain_data:
        strike = row["Strike"]

        # Evaluate Call Option
        c_ltp = row["C_LTP"]
        c_vol = row["C_Vol"]
        c_premium = c_ltp * c_vol * lot_size
        c_lots = c_vol

        if c_premium >= 1000000 or c_lots >= 500:
            vol_ratio = c_vol / avg_daily_vol if avg_daily_vol > 0 else 0.0
            unusual_flag = vol_ratio > 3.0
            
            # Strike Classification (Call Options)
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

        # Evaluate Put Option
        p_ltp = row["P_LTP"]
        p_vol = row["P_Vol"]
        p_premium = p_ltp * p_vol * lot_size
        p_lots = p_vol

        if p_premium >= 1000000 or p_lots >= 500:
            vol_ratio = p_vol / avg_daily_vol if avg_daily_vol > 0 else 0.0
            unusual_flag = vol_ratio > 3.0
            
            # Strike Classification (Put Options)
            strike_role = "Support Level" if strike <= spot_price else "Resistance / In-The-Money Level"

            scanned_trades.append({
                "Strike": strike,
                "Option_Type": "PE",
                "LTP (₹)": p_lots,
                "Total Premium (₹)": round(p_premium, 2),
                "Key Level Classification": strike_role,
                "Volume Ratio": round(vol_ratio, 2),
                "Unusual Vol Flag": "🚨 High Vol" if unusual_flag else "Normal"
            })

    if scanned_trades:
        df_scanner = pd.DataFrame(scanned_trades)

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
    session_start = datetime.datetime.strptime("09:15", "%H:%M").time()
    session_end = datetime.datetime.strptime("15:30", "%H:%M").time()
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
                smart_api, spot_token, spot_exch, api_interval, lookback_days
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
        st.subheader(f"🔥 Delta-Adjusted GEX Heatmap ({index_name})")
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
    st.markdown("<span style='font-weight:700;color:#00E676;font-size:13px;'>🧺 Basket Greeks</span>", unsafe_allow_html=True)
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
    st.markdown("<span style='font-weight:700;color:#00E676;font-size:14px;'>🧺 Strategy Basket – Legs</span>", unsafe_allow_html=True)
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
    Uses volume-location formula (not true buy/sell prints):
        delta = volume * ((close - low) / (high - low) * 2 - 1)
    SmartAPI does not provide aggressor-side volume for index futures.
    """
    df_fut = data.get("df_futures", pd.DataFrame())
    if df_fut is None or df_fut.empty or len(df_fut) < 5:
        st.info("Futures CVD not available – need futures candles with volume.")
        return

    df = df_fut.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["session_date"] = df["time"].dt.date
    latest = sorted(df["session_date"].unique())[-1]
    df = df[df["session_date"] == latest].sort_values("time").reset_index(drop=True)
    if df.empty or "volume" not in df.columns:
        st.info("No volume on futures candles for CVD.")
        return

    # ----- Method 1: Volume-location formula -----
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    loc = ((df["close"] - df["low"]) / hl * 2.0 - 1.0).fillna(0.0)
    # Clamp to [-1, +1] for safety
    loc = loc.clip(-1.0, 1.0)
    df["signed_vol"] = df["volume"] * loc
    df["cvd"] = df["signed_vol"].cumsum()
    df["time_str"] = df["time"].dt.strftime("%H:%M")

    latest_cvd = float(df["cvd"].iloc[-1])
    colour = "#00E676" if latest_cvd >= 0 else "#FF5252"

    st.markdown(
        f"<span style='font-weight:700;color:#00E676;font-size:14px;'>"
        f"📉 Futures CVD (volume-location proxy) — {latest}</span>",
        unsafe_allow_html=True
    )
    st.caption(
        "CVD = cumulative [volume × ((close−low)/(high−low)×2 − 1)]. "
        "Close at high → full +volume; close at low → full −volume; mid-range → near 0. "
        "Proxy only — SmartAPI does not supply buy/sell side volume for index futures."
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
@st.fragment(run_every=15 if st.session_state.get("enable_main_refresh", False) else None)
def live_dashboard_fragment():
    if "data_store" not in st.session_state:
        st.info("Please click '🚀 Fetch Chain & Greeks' in the sidebar to load data.")
        return

    if st.session_state.get("enable_main_refresh", False):
        refreshed_data = fetch_live_data(st.session_state["selected_timeframe"])
        if refreshed_data:
            refreshed_data["selected_expiry"] = selected_expiry_str
            st.session_state["data_store"] = refreshed_data

    data = st.session_state["data_store"]
    lvls = data.get("levels", {})

    # UI is loading data – clear any previous status
    clear_load_status()

    # ========== STICKY COMPACT MARKET SUMMARY (cleaned) ==========
    st.markdown("<div class='sticky-summary'>", unsafe_allow_html=True)
    head_l, head_r = st.columns([0.72, 0.28])
    with head_l:
        st.markdown("<h1 class='custom-heading' style='margin:0;font-size:17px;'>📊 Market Summary</h1>", unsafe_allow_html=True)
        if data.get("is_holiday_fallback", False):
            st.caption("⚠️ Non-trading day – showing last session")
    with head_r:
        cb_main = st.checkbox("Auto-Refresh 15s", value=st.session_state["enable_main_refresh"], key="cb_main_refresh")
        if cb_main != st.session_state["enable_main_refresh"]:
            st.session_state["enable_main_refresh"] = cb_main
            st.rerun()

    # Core metrics only
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Spot (Fut)", f"{data['spot_price']:.0f} ({data['F']:.0f})")
    c2.metric("Max Pain", f"{data['max_pain_strike']}")
    c3.metric("Net GEX (OI)", f"₹{data['total_net_gex_oi']/1e7:.1f} Cr")
    c4.metric("ATM IV Rank", f"{data['iv_percentile']:.0f}%")
    c5.metric("PCR", f"{data['pcr']:.2f}")
    c6.metric("C/P OI", f"{data['total_call_oi']//1000}k/{data['total_put_oi']//1000}k")
    c7.metric("Flip", f"{lvls.get('Zero_Gamma_Flip', '–')}")
    straddle_val = lvls.get("Straddle_Cost", 0)
    c8.metric("Straddle", f"₹{straddle_val:.0f}" if straddle_val else "–")

    # Transparency: what is Net GEX?
    with st.expander("?  What is Net GEX (OI)?", expanded=False):
        expiry_shown = data.get("selected_expiry") or "selected expiry"
        st.caption(
            "**Net GEX (OI)** = sum(Call gamma * OI - Put gamma * OI) * lot * spot^2 * 0.01\n"
            "• Built from **Open Interest**, not trade volume\n"
            f"• Expiry: **{expiry_shown}**\n"
            "• Per-strike bar chart uses **Delta-Adjusted GEX (OI)** (each wing weighted by its delta)\n"
            "• Superhuman **Gamma Regime** uses this same OI-based Net GEX total"
        )

    st.markdown(f"<div class='update-timestamp' style='margin-top:2px'>Updated {data['timestamp']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========== SUPERHUMAN DECISION ENGINE ==========
    scores = compute_superhuman_scores(data, data.get("df_candles", pd.DataFrame()))

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

        st.markdown("---")
        st.markdown(
            f"<div style='background:#1A1F2B;border:1px solid #2A2F3A;border-radius:8px;padding:10px 14px;margin-bottom:8px;'>"
            f"<span style='font-weight:700;color:#00E676;font-size:14px;'>🧠 Superhuman Decision Engine</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        bias_col, score_col, action_col = st.columns([0.28, 0.18, 0.54])
        with bias_col:
            st.markdown(
                f"<div style='font-size:15px;font-weight:700;color:{scores['colour']};'>"
                f"{scores['bias']}</div>",
                unsafe_allow_html=True
            )
        with score_col:
            st.metric("Composite", f"{scores['composite']:+.0f}")
        with action_col:
            st.caption(scores["action"])

        # Clarity message
        st.caption(f"📌 {scores.get('clarity', '')}")

        # Decision Tree + Intraday Log
        with st.expander("▼ Decision Tree & Intraday Log", expanded=False):
            st.markdown(
                f"**Current Composite = {scores['composite']:+.0f} → {scores['bias']}**  \n"
                f"{scores.get('clarity', '')}"
            )

            col_left, col_mid, col_right = st.columns([1.2, 1.0, 1.0])

            # LEFT – Weightage table
            with col_left:
                st.markdown("**Score Breakdown**")
                st.markdown(
                    f"""
| Component | Score | W | Contrib |
|-----------|-------|---|--------|
| Gamma Regime | {scores['gamma_regime_score']:+.0f} | 38% | {0.38*scores['gamma_regime_score']:+.1f} |
| Exp vs Real | {scores['move_score']:+.0f} | 22% | {0.22*scores['move_score']:+.1f} |
| Charm/Vanna | {scores['flow_score']:+.0f} | 15% | {0.15*scores['flow_score']:+.1f} |
| OR vs Walls | {scores['or_score']:+.0f} | 15% | {0.15*scores['or_score']:+.1f} |
| Dist to Flip | {scores.get('flip_score', 0):+.0f} | 10% | {0.10*scores.get('flip_score', 0):+.1f} |
                    """
                )

            # MIDDLE – Final Rule
            with col_mid:
                st.markdown("**Final Rule**")
                st.markdown(
                    """
- Quiet range + strong long γ → **QUIET PIN**
- Big range + strong long γ → **GAMMA REVERSION**
- Short γ / wall break → **TREND / BREAKOUT**
- |Composite| ≤ 15 → **NO EDGE**
- Otherwise → **MILD DIRECTIONAL**
                    """
                )

            # RIGHT – Intraday Log
            with col_right:
                st.markdown("**Intraday Log**")
                if not decision_log:
                    st.caption("No log yet. Keep Auto-Refresh on.")
                else:
                    for e in decision_log:
                        tstr = e["ts"].strftime("%H:%M")
                        st.markdown(f"- **{tstr}** → {e['bias']} ({e['composite']:+.0f})")

        # Score Breakdown
        with st.expander("▼ Score Breakdown & Details", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Gamma Regime**")
                st.metric("", f"{scores['gamma_regime_score']:+.0f}", label_visibility="collapsed")
                st.caption(
                    f"OI Net GEX Rs {scores['total_delta_gex_cr']:.0f} Cr (not trade volume)  \n"
                    f"sum(Call_g - Put_g) * OI * lot * spot^2 * 0.01  \n"
                    f"Expiry: selected chain | DTE={scores.get('dte', '-')} scaled  \n"
                    f"+ Long gamma -> Non-dir | - Short gamma -> Dir"
                )

                st.markdown("**Expected vs Realised**")
                st.metric("", f"{scores['move_score']:+.0f}", label_visibility="collapsed")
                st.caption(
                    f"Expected {scores['expected_move_pct']:.2f}% vs Realised {scores['realised_range_pct']:.2f}%  \n"
                    f"({scores.get('move_context', '')})  \n"
                    f"+ Straddle rich -> Non-dir | - Big move -> Dir"
                )

                st.markdown("**Charm / Vanna Flow**")
                st.metric("", f"{scores['flow_score']:+.0f}", label_visibility="collapsed")
                st.caption(
                    f"Soft tanh(VEX + CEX) scaled to +/-85  \n"
                    f"+ Supports pin -> Non-dir | - Supports accel -> Dir"
                )

            with c2:
                st.markdown("**OR vs GEX Walls**")
                st.metric("", f"{scores['or_score']:+.0f}", label_visibility="collapsed")
                or_txt = f"{scores['or_low']:.0f}-{scores['or_high']:.0f}" if scores.get('or_low') is not None else "N/A"
                st.caption(
                    f"OR {or_txt} vs Walls  \n"
                    f"Time-of-day factor {scores.get('tod_factor', 1):.2f}  \n"
                    f"+ Inside walls -> Non-dir | - Break -> Dir"
                )

                st.markdown("**Distance to Flip**")
                st.metric("", f"{scores.get('flip_score', 0):+.0f}", label_visibility="collapsed")
                st.caption(
                    f"Spot vs Flip distance {scores.get('dist_to_flip_pct', 0):.3f}%  \n"
                    f"Close + long gamma -> pin bonus | Far + short gamma -> accel  \n"
                    f"Weight 10%"
                )

                st.markdown("**Composite**")
                st.metric("", f"{scores['composite']:+.0f}", label_visibility="collapsed")
                st.caption(
                    f"0.38*Gamma + 0.22*Move + 0.15*Flow + 0.15*OR + 0.10*Flip  \n"
                    f"-> **{scores['bias']}**"
                )

    else:
        st.warning("Could not compute Superhuman scores – insufficient data.")

    # ========== UNDERLYING TECHNICALS – SPOT + FUTURES VWAP ==========
    df_full = data.get("df_candles", pd.DataFrame())
    df_fut  = data.get("df_futures", pd.DataFrame())
    basis   = data.get("basis_info", {})

    chart_head_col, tf_col = st.columns([0.75, 0.25])
    with chart_head_col:
        st.markdown("<span style='font-weight:700;color:#00E676;font-size:15px;'>📈 Underlying Technicals</span>", unsafe_allow_html=True)
    with tf_col:
        selected_tf = st.selectbox("TF", ["1 min", "3 min", "5 min", "15 min"],
                                   index=["1 min", "3 min", "5 min", "15 min"].index(st.session_state["selected_timeframe"]),
                                   key="tf_select_frag", label_visibility="collapsed")
        if selected_tf != st.session_state["selected_timeframe"]:
            st.session_state["selected_timeframe"] = selected_tf
            updated_chart_data = fetch_live_data(selected_tf)
            if updated_chart_data:
                st.session_state["data_store"] = updated_chart_data
                st.rerun()

    if not df_full.empty and len(df_full) >= 20:
        latest_row = df_full.iloc[-1]
        rsi_val = latest_row["rsi"]
        rsi_status = "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral")
        rsi_badge_cls = "badge-bearish" if rsi_val > 70 else ("badge-bullish" if rsi_val < 30 else "badge-neutral")
        macd_val = latest_row["macd"]
        macd_sig = latest_row["macd_signal"]
        macd_status = "Bullish XO" if macd_val > macd_sig else "Bearish XO"
        macd_badge_cls = "badge-bullish" if macd_val > macd_sig else "badge-bearish"
        recent_bw = df_full["bb_bandwidth"].tail(20)
        bw_threshold = recent_bw.quantile(0.20)
        is_sqz = latest_row["bb_bandwidth"] <= bw_threshold
        sqz_status = "Squeeze" if is_sqz else "Expand"
        sqz_badge_cls = "badge-neutral" if is_sqz else "badge-bullish"

        st.markdown(
            f"<div style='margin-bottom:4px'>"
            f"<span class='status-badge {sqz_badge_cls}'>BB: {sqz_status}</span>"
            f"<span class='status-badge {macd_badge_cls}'>MACD: {macd_status}</span>"
            f"<span class='status-badge {rsi_badge_cls}'>RSI: {rsi_val:.0f} ({rsi_status})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Spot chart (no VWAP)
        df_full["session_date"] = pd.to_datetime(df_full["time"]).dt.date
        last_3_dates = sorted(df_full["session_date"].unique())[-3:]
        df_chart = df_full[df_full["session_date"].isin(last_3_dates)].copy()
        df_chart["time_str"] = pd.to_datetime(df_chart["time"]).dt.strftime("%d-%b %H:%M")

        min_p = min(df_chart["close"].min(), df_chart["bb_lower"].min())
        max_p = max(df_chart["close"].max(), df_chart["bb_upper"].max())
        padding = (max_p - min_p) * 0.05

        tech_left, tech_right = st.columns([0.58, 0.42])

        with tech_left:
            st.caption("Spot Price + Bollinger (VWAP moved to Futures chart below)")
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

        with tech_right:
            fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45])
            colors_macd = np.where(df_chart["macd_hist"] >= 0, "#00E676", "#FF5252")
            fig_ind.add_trace(plt_go.Bar(x=df_chart["time_str"], y=df_chart["macd_hist"], name="Hist", marker_color=colors_macd, showlegend=False), row=1, col=1)
            fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd"], mode="lines", name=f"MACD [{macd_val:.1f}]", line=dict(color="#2196F3", width=1.3)), row=1, col=1)
            fig_ind.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd_signal"], mode="lines", name=f"Sig [{macd_sig:.1f}]", line=dict(color="#FF9800", width=1.3)), row=1, col=1)
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

        # ---------- FUTURES CHART + REAL VWAP ----------
        st.markdown("---")
        fut_header_col, basis_col, band_col = st.columns([0.55, 0.25, 0.20])
        with fut_header_col:
            expiry_txt = basis.get("fut_expiry", "N/A")
            st.markdown(
                f"<span style='font-weight:700;color:#00E676;font-size:14px;'>"
                f"📉 Near-Month Futures + Real VWAP ({expiry_txt})</span>",
                unsafe_allow_html=True
            )
        with basis_col:
            if basis.get("basis") is not None:
                basis_color = "#00E676" if basis["basis"] >= 0 else "#FF5252"
                st.markdown(
                    f"<div style='text-align:right;font-size:13px;padding-top:4px;'>"
                    f"Basis: <span style='color:{basis_color};font-weight:700;'>"
                    f"{basis['basis']:+.1f}</span> pts</div>",
                    unsafe_allow_html=True
                )
        with band_col:
            sigma_mult = st.selectbox(
                "VWAP Bands",
                options=[1.0, 1.5, 2.0],
                index=1,                    # default 1.5σ
                format_func=lambda x: f"±{x}σ",
                key="vwap_sigma_select",
                label_visibility="collapsed"
            )

        # Fallback / market-closed banner
        fut_msg = data.get("fut_fallback_msg", "")
        if data.get("fut_is_fallback") or "closed" in str(fut_msg).lower():
            st.warning(fut_msg)
        elif fut_msg:
            st.caption(fut_msg)

        if not df_fut.empty and len(df_fut) >= 5:
            df_fut = df_fut.copy()
            df_fut["session_date"] = pd.to_datetime(df_fut["time"]).dt.date
            last_3_fut = sorted(df_fut["session_date"].unique())[-3:]
            df_fchart = df_fut[df_fut["session_date"].isin(last_3_fut)].copy()
            df_fchart["time_str"] = pd.to_datetime(df_fchart["time"]).dt.strftime("%d-%b %H:%M")

            # ----- Calculate VWAP Standard Deviation Bands -----
            # ----- Correct Volume-Weighted VWAP Standard Deviation Bands (causal, resets daily) -----
            df_fchart = df_fchart.copy()
            df_fchart["tp"] = (df_fchart["high"] + df_fchart["low"] + df_fchart["close"]) / 3.0
            
            def _calc_session_vwap_bands(group, k):
                group = group.copy().reset_index(drop=True)
                
                cum_vol = 0.0
                cum_tp_vol = 0.0
                cum_sq_dev = 0.0
                
                vwaps = []
                stds = []
                
                for i in range(len(group)):
                    vol = float(group.loc[i, "volume"])
                    tp  = float(group.loc[i, "tp"])
                    
                    cum_vol += vol
                    cum_tp_vol += tp * vol
                    
                    vwap = cum_tp_vol / cum_vol if cum_vol > 0 else tp
                    
                    # Volume-weighted variance (using current VWAP)
                    # Approximate but widely used method
                    cum_sq_dev += vol * (tp - vwap) ** 2
                    variance = cum_sq_dev / cum_vol if cum_vol > 0 else 0.0
                    sigma = np.sqrt(max(variance, 0.0))
                    
                    vwaps.append(vwap)
                    stds.append(sigma)
                
                group["vwap"] = vwaps
                group["vwap_std"] = stds
                group["vwap_upper"] = group["vwap"] + k * group["vwap_std"]
                group["vwap_lower"] = group["vwap"] - k * group["vwap_std"]
                return group
            
            df_fchart = df_fchart.groupby("session_date", group_keys=False).apply(
                lambda g: _calc_session_vwap_bands(g, sigma_mult)
            )
            latest_vwap = df_fchart["vwap"].iloc[-1] if "vwap" in df_fchart.columns else None
            latest_fut  = df_fchart["close"].iloc[-1]

            fmin = min(df_fchart["close"].min(), df_fchart["vwap_lower"].min())
            fmax = max(df_fchart["close"].max(), df_fchart["vwap_upper"].max())
            fpad = (fmax - fmin) * 0.06

            fig_fut = plt_go.Figure()

            # Upper band
            fig_fut.add_trace(plt_go.Scatter(
                x=df_fchart["time_str"], y=df_fchart["vwap_upper"],
                mode="lines", name=f"+{sigma_mult}σ",
                line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
                showlegend=True
            ))
            # Lower band (fills to upper)
            fig_fut.add_trace(plt_go.Scatter(
                x=df_fchart["time_str"], y=df_fchart["vwap_lower"],
                mode="lines", name=f"−{sigma_mult}σ",
                line=dict(color="rgba(255,152,0,0.35)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(255,152,0,0.08)",
                showlegend=True
            ))
            # VWAP line
            fig_fut.add_trace(plt_go.Scatter(
                x=df_fchart["time_str"], y=df_fchart["vwap"],
                mode="lines", name="VWAP",
                line=dict(color="#FF9800", width=2.2, dash="dot")
            ))
            # Futures price
            fig_fut.add_trace(plt_go.Scatter(
                x=df_fchart["time_str"], y=df_fchart["close"],
                mode="lines", name="Futures",
                line=dict(color="#2196F3", width=2)
            ))

            fig_fut.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0
                ),
                hovermode="x unified",
                yaxis=dict(range=[fmin - fpad, fmax + fpad], tickformat="d"),
                xaxis=dict(type="category", nticks=8),
                title=dict(
                    text=f"Futures: {latest_fut:,.1f}  |  VWAP: {latest_vwap:,.1f}  |  Bands: ±{sigma_mult}σ",
                    x=0.01,
                    font=dict(size=12)
                )
            )
            st.plotly_chart(fig_fut, use_container_width=True)

            st.caption(
                f"Orange dotted line = VWAP (real futures volume). "
                f"Shaded area = ±{sigma_mult}σ deviation from VWAP. "
                "This is the institutional benchmark."
            )
        else:
            st.info("Futures candles / VWAP not available. " + (str(fut_msg) if fut_msg else "Click Fetch or wait for next refresh."))

        # ---------- FUTURES CVD (directly under Futures VWAP chart) ----------
        st.markdown("---")
        render_futures_cvd_chart(data)

    # --- GEX CHARTS ---
    st.markdown("---")
    df_chain = pd.DataFrame(data.get("chain_results") or [])
    # --- GEX CHARTS ---
    st.markdown("---")
    df_chain = pd.DataFrame(data.get("chain_results") or [])

    if not df_chain.empty and lvls:
        df_chain["Total_Vol"] = df_chain["C_Vol"] + df_chain["P_Vol"]
        df_chain["Total_OI"] = df_chain["C_OI"] + df_chain["P_OI"]
        min_strike_val = df_chain["Strike"].min() - 50
        max_strike_val = df_chain["Strike"].max() + 50

        def calculate_synced_ranges(v1_pos, v1_neg, v2_pos, v2_neg):
            y1_max = max(v1_pos.max(), 1.0)
            y1_min = min(v1_neg.min(), -1.0)
            y2_max = max(v2_pos.max(), 1.0)
            y2_min = min(v2_neg.min(), -1.0)
            ratio1 = abs(y1_min) / max(y1_max, 1e-5)
            ratio2 = abs(y2_min) / max(y2_max, 1e-5)
            max_ratio = max(ratio1, ratio2)
            range1 = [-y1_max * max_ratio * 1.05, y1_max * 1.05]
            range2 = [-y2_max * max_ratio * 1.05, y2_max * 1.05]
            return range1, range2

        # GEX vs OI + GEX vs Volume
        gex_l, gex_r = st.columns(2)
        with gex_l:
            st.markdown("<span style='font-weight:700;color:#00E676;font-size:14px;'>📈 GEX vs OI</span>", unsafe_allow_html=True)
            gex_oi_colors = np.where(df_chain["Net_GEX_OI"] >= 0, "#006400", "#8B0000")
            fig_oi = make_subplots(specs=[[{"secondary_y": True}]])
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["Net_GEX_OI"], name="Net GEX", marker_color=gex_oi_colors, opacity=0.85, width=25), secondary_y=True)
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["C_OI"], name="Call OI", marker_color="#2E7D32", opacity=0.55), secondary_y=False)
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=-df_chain["P_OI"], name="Put OI", marker_color="#C62828", opacity=0.55), secondary_y=False)
            fig_oi.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
            fig_oi.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot", annotation_font_size=10)
            fig_oi.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip", annotation_font_size=10)
            oi1_range, oi2_range = calculate_synced_ranges(df_chain["C_OI"], -df_chain["P_OI"], df_chain["Net_GEX_OI"].clip(lower=0), df_chain["Net_GEX_OI"].clip(upper=0))
            fig_oi.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=380, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
            fig_oi.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_oi.update_yaxes(range=oi1_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF")
            fig_oi.update_yaxes(range=oi2_range, secondary_y=True, showgrid=False)
            st.plotly_chart(fig_oi, use_container_width=True)

        with gex_r:
            st.markdown("<span style='font-weight:700;color:#00E676;font-size:14px;'>📊 GEX vs Volume</span>", unsafe_allow_html=True)
            gex_vol_colors = np.where(df_chain["Net_GEX_Vol"] >= 0, "#006400", "#8B0000")
            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["Net_GEX_Vol"], name="Net GEX", marker_color=gex_vol_colors, opacity=0.85, width=25), secondary_y=True)
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["C_Vol"], name="Call Vol", marker_color="#81C784", opacity=0.55), secondary_y=False)
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=-df_chain["P_Vol"], name="Put Vol", marker_color="#FF8A80", opacity=0.55), secondary_y=False)
            fig_vol.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
            fig_vol.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot", annotation_font_size=10)
            fig_vol.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip", annotation_font_size=10)
            v1_range, v2_range = calculate_synced_ranges(df_chain["C_Vol"], -df_chain["P_Vol"], df_chain["Net_GEX_Vol"].clip(lower=0), df_chain["Net_GEX_Vol"].clip(upper=0))
            fig_vol.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=380, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
            fig_vol.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_vol.update_yaxes(range=v1_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF")
            fig_vol.update_yaxes(range=v2_range, secondary_y=True, showgrid=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        # Delta-GEX + Heatmap
        st.markdown("---")
        d_left, d_right = st.columns([0.40, 0.60])
        with d_left:
            st.markdown("<span style='font-weight:700;color:#00E676;font-size:14px;'>🎯 Delta-Adjusted GEX</span>", unsafe_allow_html=True)
            delta_gex_colors = np.where(df_chain["Net_Delta_GEX_OI"] >= 0, "#00E676", "#FF5252")
            fig_delta_gex = plt_go.Figure()
            fig_delta_gex.add_trace(plt_go.Bar(
                x=df_chain["Strike"], y=df_chain["Net_Delta_GEX_OI"], name="Δ-GEX",
                marker_color=delta_gex_colors, opacity=0.85, width=25,
            ))
            fig_delta_gex.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
            fig_delta_gex.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot", annotation_font_size=10)
            fig_delta_gex.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip", annotation_font_size=10)
            fig_delta_gex.update_layout(
                template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                height=520, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
            )
            fig_delta_gex.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_delta_gex.update_yaxes(showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF")
            st.plotly_chart(fig_delta_gex, use_container_width=True)

        with d_right:
            render_delta_gex_heatmap(
                data, Index_Name, selected_expiry_str,
                st.session_state.get("heatmap_timeframe", "5 min"),
            )

     

        render_live_alert_ribbon(data)
        render_basket_table_fullwidth(data)

        # VEX/CEX + Skew
        st.markdown("---")
        vex_col, skew_col = st.columns([0.60, 0.40])
        with vex_col:
            st.markdown("<span style='font-weight:700;color:#00E676;font-size:14px;'>⚡ VEX / CEX Profile</span>", unsafe_allow_html=True)
            fig_vex_cex = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vex_cex.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["VEX"], name="VEX", marker_color="#00E676", opacity=0.75, width=20), secondary_y=False)
            fig_vex_cex.add_trace(plt_go.Scatter(x=df_chain["Strike"], y=df_chain["CEX"], name="CEX", line=dict(color="#2196F3", width=2), mode="lines+markers", marker=dict(size=4)), secondary_y=True)
            fig_vex_cex.add_hline(y=0, line_width=1.2, line_color="#FFFFFF")
            fig_vex_cex.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot", annotation_font_size=10)
            vex_range, cex_range = calculate_synced_ranges(df_chain["VEX"].clip(lower=0), df_chain["VEX"].clip(upper=0), df_chain["CEX"].clip(lower=0), df_chain["CEX"].clip(upper=0))
            fig_vex_cex.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=400, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
            fig_vex_cex.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_vex_cex.update_yaxes(range=vex_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF")
            fig_vex_cex.update_yaxes(range=cex_range, secondary_y=True, showgrid=False)
            st.plotly_chart(fig_vex_cex, use_container_width=True)

        with skew_col:
            st.markdown(f"<span style='font-weight:700;color:#00E676;font-size:14px;'>📉 IV Skew ({selected_expiry_str})</span>", unsafe_allow_html=True)
            smart_api = get_smart_api_client()
            if smart_api and not df_master.empty:
                latest_spot = data["spot_price"]
                df_chain_iv = fetch_and_compute_full_chain_iv(
                    smart_api, df_master, Index_Name, Exchange, selected_expiry_str,
                    latest_spot, rate_param, strikes_below=strikes_below, strikes_above=strikes_above,
                )
                if not df_chain_iv.empty:
                    tab1, tab2 = st.tabs(["OTM Skew (Puts & Calls)", "Both Raw Curves (CE vs PE)"])
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
                else:
                    st.info("Skew data unavailable.")
            else:
                st.info("API session needed for skew.")
    elif not df_chain.empty:
        st.info("Key levels not available – GEX charts skipped.")
live_dashboard_fragment()

# --- Full-width: Institutional Order Flow, then Raw Z-Score details ---
st.markdown("---")
institutional_order_flow_scanner_fragment()
st.markdown("---")
zscore_analysis_fragment(mode="raw")

# Clear loading status once full UI has rendered
try:
    clear_load_status()
except Exception:
    pass
