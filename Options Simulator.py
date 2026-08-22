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
.block-container { padding-top: 1.0rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { background-color: rgba(0, 0, 0, 0) !important; }
h1, h2, h3, .custom-heading { color: #00E676 !important; font-size: 20px !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; }
div[data-testid="stMetricValue"] { font-size: 18px !important; color: #00E676 !important; }
.update-timestamp { font-size: 13px; color: #00E676; font-weight: 600; text-align: right; }
div[data-baseweb="select"] > div { background-color: #1E222D !important; color: #FAFAFA !important; border-color: #363C4E !important; }
.stButton>button { border-radius: 6px; font-weight: 600; }
.status-badge { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; margin-right: 8px; }
.badge-bullish { background-color: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; }
.badge-bearish { background-color: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid #FF5252; }
.badge-neutral { background-color: rgba(255, 152, 0, 0.15); color: #FF9800; border: 1px solid #FF9800; }
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

# --- TECHNICAL INDICATOR ENGINE ---
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
        st.session_state["data_store"] = new_data

# --- Z-SCORE ANALYSIS FRAGMENT ---
@st.fragment(run_every=300 if st.session_state.get("enable_zscore_refresh", False) else None)
def zscore_analysis_fragment():
    st.markdown("---")
    head_c1, head_c2 = st.columns([0.65, 0.35])
    with head_c1:
        st.subheader(f"📊 Market Z-Scores for {Index_Name}")
        st.caption("ℹ️ **Method 1 Active**: Evaluates continuous Spot Index Historical Volatility (HV) and Near-Month Futures Volume.")
    with head_c2:
        st.write("")
        enable_z_ref = st.checkbox("Enable Z-Score Auto-Refresh (5 min)", value=st.session_state["enable_zscore_refresh"], key="cb_zscore_refresh")
        if enable_z_ref != st.session_state["enable_zscore_refresh"]:
            st.session_state["enable_zscore_refresh"] = enable_z_ref
            st.rerun()

    btn_c1, btn_c2 = st.columns([0.30, 0.70])
    with btn_c1:
        calc_z_btn = st.button("🔄 Compute / Refresh Z-Scores", use_container_width=True)

    progress_holder = st.container()

    smart_api = get_smart_api_client()

    if calc_z_btn or st.session_state["zscore_data_store"].empty:
        if smart_api:
            z_df = fetch_futures_zscores_method1(smart_api, Index_Name, hv_days, df_master, progress_container=progress_holder)
            st.session_state["zscore_data_store"] = z_df
        else:
            st.error("SmartAPI Session is uninitialized. Cannot calculate Z-Scores.")

    z_df = st.session_state["zscore_data_store"]

    if not z_df.empty:
        st.markdown("### Recent Highlights (Last 5 Trading Days)")
        last_5 = z_df.tail(5).copy()

        def style_z(val):
            if val >= 1.5 or val <= -1.5:
                return 'background-color: #ff4d4d; color: white; font-weight: bold;'
            elif 0.5 <= val < 1.5 or -1.5 < val <= -0.5:
                return 'background-color: #ffea80; color: black;'
            else:
                return 'background-color: #b3ffb3; color: black;'

        summary_cols = ['Futures_Volume_Z', 'Volatility_Proxy_Z']
        
        styled_z_df = last_5[summary_cols].sort_index(ascending=False).style.map(
            style_z, subset=summary_cols
        ).format({col: "{:.2f}" for col in summary_cols})

        st.dataframe(styled_z_df, use_container_width=True)

        with st.expander("🔍 Inspect Raw Calculated Daily Totals & HV Details (All Evaluated Dates)", expanded=False):
            st.info(f"Showing all **{len(z_df)}** trading days used in evaluating rolling averages and std dev.")
            
            display_cols = [
                'Futures_Close', 'Futures_Volume', 'Vol_Mean', 'Vol_Std', 'Futures_Volume_Z',
                'Volatility_Proxy', 'HV_Mean', 'HV_Std', 'Volatility_Proxy_Z'
            ]
            
            st.dataframe(
                z_df[display_cols].sort_index(ascending=False).style.format({
                    'Futures_Close': "{:,.2f}",
                    'Futures_Volume': "{:,.0f}",
                    'Vol_Mean': "{:,.0f}",
                    'Vol_Std': "{:,.0f}",
                    'Futures_Volume_Z': "{:.2f}",
                    'Volatility_Proxy': "{:.4f}",
                    'HV_Mean': "{:.4f}",
                    'HV_Std': "{:.4f}",
                    'Volatility_Proxy_Z': "{:.2f}"
                }),
                use_container_width=True
            )
    else:
        st.info("Click '🔄 Compute / Refresh Z-Scores' above to load Futures volume & HV Z-Score analysis.")

# --- INSTITUTIONAL ORDER FLOW SCANNER MODULE ---
def institutional_order_flow_scanner_fragment():
    st.markdown("---")
    st.subheader(f"🏛️ Institutional Order Flow Scanner ({Index_Name})")
    st.caption("Filters out retail noise to isolate high-conviction institutional trades based on premium thresholds and key support/resistance strike levels.")

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


def render_delta_gex_heatmap(data: dict, index_name: str, expiry_str: str, heatmap_tf_label: str):
    """Render strike × session-time Delta-Adjusted GEX heatmap + Exit / Long alert ribbon."""
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

    # OTM Call IV (nearest OTM call) and OTM Put IV (nearest OTM put)
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

    # Volume from latest 5-min candle vs 20-period MA
    df_candles = data.get("df_candles", pd.DataFrame())
    vol_ratio = 0.0
    if not df_candles.empty and "volume" in df_candles.columns and len(df_candles) >= 21:
        recent_vol = float(df_candles["volume"].iloc[-1])
        ma20 = float(df_candles["volume"].iloc[-21:-1].mean())
        vol_ratio = recent_vol / ma20 if ma20 > 0 else 0.0

    # ---- accumulate history for heatmap + alerts ----
    now_ist = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    gex_hist = st.session_state.get("gex_heatmap_history", [])
    alert_hist = st.session_state.get("alert_metrics_history", [])

    # Reset histories if strike window changes significantly
    if gex_hist and (len(gex_hist[-1]["strikes"]) != len(strikes) or
                     abs(gex_hist[-1]["strikes"][0] - strikes[0]) > 1 or
                     abs(gex_hist[-1]["strikes"][-1] - strikes[-1]) > 1):
        gex_hist = []
        alert_hist = []

    should_append = True
    if alert_hist:
        last_ts = alert_hist[-1]["ts"]
        if (now_ist - last_ts).total_seconds() < 25:
            should_append = False

    if should_append:
        gex_hist.append({"ts": now_ist, "strikes": list(strikes), "gex_cr": list(gex_cr)})
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
        cutoff = now_ist - datetime.timedelta(hours=6)
        gex_hist = [h for h in gex_hist if h["ts"] >= cutoff]
        alert_hist = [h for h in alert_hist if h["ts"] >= cutoff]
        st.session_state["gex_heatmap_history"] = gex_hist
        st.session_state["alert_metrics_history"] = alert_hist

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

    # ---- evolving GEX matrix from history ----
    # Place each live snapshot at its real wall-clock time so the same strike
    # can show different intensity / colour across the session (like the reference).
    gex_matrix = np.full((n_strikes, n_times), np.nan, dtype=float)
    current_vec = np.array(gex_cr, dtype=float)

    if not gex_hist:
        # First load – paint current profile; later refreshes will add variation
        gex_matrix[:] = current_vec[:, None]
    else:
        # Build a time-series of GEX vectors aligned to candle timestamps
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
                vec = np.array([src_map.get(min(src_strikes, key=lambda x: abs(x - s)), 0.0) for s in strikes])
            hist_times.append(ht)
            hist_vecs.append(vec)

        for t_idx, t in enumerate(session_times):
            if getattr(t, "tzinfo", None) is None:
                t = pytz.timezone("Asia/Kolkata").localize(t)
            # nearest snapshot whose time is <= candle time
            chosen_idx = None
            for i in range(len(hist_times) - 1, -1, -1):
                if hist_times[i] <= t:
                    chosen_idx = i
                    break
            if chosen_idx is None:
                # before first snapshot – use first snapshot (will be overwritten as history grows)
                gex_matrix[:, t_idx] = hist_vecs[0]
            else:
                gex_matrix[:, t_idx] = hist_vecs[chosen_idx]

        # Forward-fill any remaining NaNs with the latest known vector
        last_valid = current_vec
        for t_idx in range(n_times):
            if np.isnan(gex_matrix[0, t_idx]):
                gex_matrix[:, t_idx] = last_valid
            else:
                last_valid = gex_matrix[:, t_idx]

    # Symmetric colour scale centred at 0, but use a robust percentile so extreme
    # single-strike spikes don't wash out the rest of the surface (matches ref look)
    finite_vals = gex_matrix[np.isfinite(gex_matrix)]
    if len(finite_vals) > 0:
        p95 = float(np.nanpercentile(np.abs(finite_vals), 95))
        max_abs_gex = max(p95, 0.05)
    else:
        max_abs_gex = 0.05

    gex_colorscale = [
        [0.0, "#b71c1c"],   # deep red (strong short-gamma)
        [0.25, "#e53935"],
        [0.45, "#ffee58"],  # neutral / low GEX
        [0.55, "#ffee58"],
        [0.75, "#43a047"],
        [1.0, "#1b5e20"],   # deep green (strong long-gamma)
    ]

    hm_c1, hm_c2 = st.columns([0.65, 0.35])
    with hm_c1:
        st.subheader(f"🔥 Delta-Adjusted GEX Heatmap ({index_name})")
    with hm_c2:
        st.markdown(
            f"<div class='update-timestamp'>Expiry: {expiry_str} | Session: {session_date_str} | {heatmap_tf_label} | Snapshots: {len(gex_hist)}</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Strike vs IST session time. Colour = live Delta-Adjusted Net GEX (₹ Cr) from successive SmartAPI refreshes. "
        "Same strike can change shade/intensity as new snapshots arrive (enable Auto-Refresh). "
        "Green = long-gamma / pinning; red = short-gamma / acceleration. Cyan line = spot path. "
        f"History snapshots collected: {len(gex_hist)}."
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
            text=f"<b>{index_name} Delta-Adjusted GEX Heatmap</b> ({heatmap_tf_label}) | Session: {session_date_str} | Expiry: {expiry_str}",
            x=0.01,
        ),
        xaxis_title="IST Time",
        yaxis_title="Strike Price",
        height=680,
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

    # ---------- Status ribbon (dropdowns) ----------
    # NOTE: st.expander title does NOT support HTML – badges are rendered inside the body.
    st.markdown("---")
    st.markdown("### 🚨 Live Alert Status (below GEX Heatmap)")

    def _badge_html(flag, yes_txt="YES", no_txt="NO"):
        if flag:
            return (
                f"<span style='background:rgba(255,82,82,0.25);color:#FF5252;"
                f"padding:3px 10px;border-radius:4px;font-weight:700;"
                f"border:1px solid #FF5252;font-size:13px'>{yes_txt}</span>"
            )
        return (
            f"<span style='background:rgba(0,230,118,0.15);color:#00E676;"
            f"padding:3px 10px;border-radius:4px;font-weight:700;"
            f"border:1px solid #00E676;font-size:13px'>{no_txt}</span>"
        )

    def _sub_html(flag, label):
        colour = "#FF5252" if flag else "#00E676"
        txt = "Yes" if flag else "No"
        return f"<span style='color:{colour};font-weight:600'>{label} – {txt}</span>"

    # ---- Situation 1 ----
    exit_label = "YES" if exit_alert else "NO"
    with st.expander(
        f"Situation 1 · Non-Directional Exit / Risk-Off Alert  →  {exit_label}",
        expanded=exit_alert,
    ):
        st.markdown(
            f"<div style='margin-bottom:8px'>Overall: {_badge_html(exit_alert)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style='line-height:1.9;font-size:14px'>
            {_sub_html(crit_a_exit, 'Wall Collapse (≥25% pos-GEX drop / 10 min)')}<br>
            {_sub_html(crit_b_exit, 'Vol Expansion (OTM IV ↑ ≥0.8% / 5 min)')}<br>
            {_sub_html(crit_c_exit, 'GEX Flip (spot within 0.3% of flip or in –GEX zone)')}<br>
            <br>
            <small style='color:#AAA'>
            Wall drop: {wall_drop:.1f}% &nbsp;|&nbsp;
            Call IV Δ: {call_iv_chg:+.2f}% &nbsp;|&nbsp;
            Put IV Δ: {put_iv_chg:+.2f}% &nbsp;|&nbsp;
            Dist to flip: {dist_to_flip_pct:.2f}%
            </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Situation 2 ----
    long_label = "YES" if long_alert else "NO"
    with st.expander(
        f"Situation 2 · Long Position Entry (Directional Breakout)  →  {long_label}",
        expanded=long_alert,
    ):
        st.markdown(
            f"<div style='margin-bottom:8px'>Overall: {_badge_html(long_alert)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style='line-height:1.9;font-size:14px'>
            {_sub_html(crit_a_long, 'Gamma Fuel (neg-GEX above spot OR pos-GEX collapse ≥40%)')}<br>
            {_sub_html(crit_b_long, 'Aggressive Demand (OTM Call IV ↑ ≥1.5% / 5 min)')}<br>
            {_sub_html(crit_c_long, 'Volume Confirmation (5-min vol ≥ 1.5× 20-MA)')}<br>
            <br>
            <small style='color:#AAA'>
            Neg GEX above: {cur['neg_gex_above']:.2f} Cr &nbsp;|&nbsp;
            Pos collapse: {pos_collapse:.1f}% &nbsp;|&nbsp;
            Call IV Δ: {call_iv_chg:+.2f}% &nbsp;|&nbsp;
            Vol ratio: {cur['vol_ratio']:.2f}×
            </small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- LIVE DASHBOARD FRAGMENT ---
@st.fragment(run_every=15 if st.session_state.get("enable_main_refresh", False) else None)
def live_dashboard_fragment():
    if "data_store" not in st.session_state:
        st.info("Please click '🚀 Fetch Chain & Greeks' in the sidebar to load data.")
        return

    if st.session_state.get("enable_main_refresh", False):
        refreshed_data = fetch_live_data(st.session_state["selected_timeframe"])
        if refreshed_data:
            st.session_state["data_store"] = refreshed_data

    data = st.session_state["data_store"]

    with st.container(border=True):
        col_title, col_status = st.columns([0.65, 0.35])
        with col_title:
            st.markdown("<h1 class='custom-heading'>📊 Option Chain Technical Analysis</h1>", unsafe_allow_html=True)
            if data.get("is_holiday_fallback", False):
                st.warning("⚠️ Today is a non-trading day/holiday. Technical charts are displaying the latest available trading session.")
        with col_status:
            st.write("")
            cb_main = st.checkbox("Enable Auto-Refresh (15s)", value=st.session_state["enable_main_refresh"], key="cb_main_refresh")
            if cb_main != st.session_state["enable_main_refresh"]:
                st.session_state["enable_main_refresh"] = cb_main
                st.rerun()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Spot (Syn. Fut)", f"{data['spot_price']:.2f} ({data['F']:.2f})")
        m2.metric("Max Pain", f"{data['max_pain_strike']}")
        m3.metric("Net GEX (OI)", f"₹{data['total_net_gex_oi']/1e7:.2f} Cr")
        m4.metric("ATM IV Rank", f"{data['iv_percentile']:.1f}%")
        m5.metric("PCR (OI)", f"{data['pcr']:.2f}")
        m6.metric("Total Call / Put OI", f"{data['total_call_oi'] // 1000}k / {data['total_put_oi'] // 1000}k")

        st.markdown(f"<div class='update-timestamp'>Updated as on {data['timestamp']}</div>", unsafe_allow_html=True)

    df_full = data.get("df_candles", pd.DataFrame())

    chart_head_col, tf_col = st.columns([0.70, 0.30])
    with chart_head_col:
        st.markdown("<span style='font-weight: 700; color: #00E676; font-size: 18px;'>📈 Underlying Technical Charts</span>", unsafe_allow_html=True)
    with tf_col:
        selected_tf = st.selectbox("Timeframe", ["1 min", "3 min", "5 min", "15 min"], index=["1 min", "3 min", "5 min", "15 min"].index(st.session_state["selected_timeframe"]), key="tf_select_frag")
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
        macd_status = "Bullish Crossover" if macd_val > macd_sig else "Bearish Crossover"
        macd_badge_cls = "badge-bullish" if macd_val > macd_sig else "badge-bearish"

        recent_bw = df_full["bb_bandwidth"].tail(20)
        bw_threshold = recent_bw.quantile(0.20)
        is_sqz = latest_row["bb_bandwidth"] <= bw_threshold
        sqz_status = "Squeeze Active" if is_sqz else "Normal Expansion"
        sqz_badge_cls = "badge-neutral" if is_sqz else "badge-bullish"

        st.markdown(
            f"""
            <div style='margin-bottom: 8px;'>
                <span class='status-badge {sqz_badge_cls}'>BB Squeeze: {sqz_status}</span>
                <span class='status-badge {macd_badge_cls}'>MACD Status: {macd_status}</span>
                <span class='status-badge {rsi_badge_cls}'>RSI (14): {rsi_val:.1f} ({rsi_status})</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        df_full["session_date"] = pd.to_datetime(df_full["time"]).dt.date
        last_3_dates = sorted(df_full["session_date"].unique())[-3:]
        df_chart = df_full[df_full["session_date"].isin(last_3_dates)].copy()
        df_chart["time_str"] = pd.to_datetime(df_chart["time"]).dt.strftime("%d-%b %H:%M")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.25, 0.20])

        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["close"], mode="lines", name="Spot Price", line=dict(color="#00E676", width=2)), row=1, col=1)
        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_upper"], mode="lines", name="BB Upper (20, 2)", line=dict(color="rgba(33, 150, 243, 0.5)", width=1)), row=1, col=1)
        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["bb_lower"], mode="lines", name="BB Lower (20, 2)", line=dict(color="rgba(33, 150, 243, 0.5)", width=1), fill='tonexty', fillcolor='rgba(33, 150, 243, 0.05)'), row=1, col=1)
        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["vwap"], mode="lines", name="VWAP (Intraday)", line=dict(color="#FF9800", width=2, dash="dot")), row=1, col=1)

        colors_macd = np.where(df_chart["macd_hist"] >= 0, "#00E676", "#FF5252")
        fig.add_trace(plt_go.Bar(x=df_chart["time_str"], y=df_chart["macd_hist"], name="MACD Hist", marker_color=colors_macd), row=2, col=1)
        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd"], mode="lines", name=f"MACD (12, 26, 9) [{macd_val:.2f}]", line=dict(color="#2196F3", width=1.5)), row=2, col=1)
        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["macd_signal"], mode="lines", name=f"Signal [{macd_sig:.2f}]", line=dict(color="#FF9800", width=1.5)), row=2, col=1)

        fig.add_trace(plt_go.Scatter(x=df_chart["time_str"], y=df_chart["rsi"], mode="lines", name=f"RSI (14) [{rsi_val:.1f}]", line=dict(color="#E040FB", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF5252", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00E676", line_width=1, row=3, col=1)

        min_p = min(df_chart["close"].min(), df_chart["vwap"].min(), df_chart["bb_lower"].min())
        max_p = max(df_chart["close"].max(), df_chart["vwap"].max(), df_chart["bb_upper"].max())
        padding = (max_p - min_p) * 0.05

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            height=500,
            margin=dict(l=20, r=20, t=10, b=10),
            showlegend=True,
            hovermode="x unified"
        )
        fig.update_yaxes(range=[min_p - padding, max_p + padding], tickformat="d", row=1, col=1)
        fig.update_xaxes(type="category", nticks=12)
        fig.update_traces(hovertemplate="%{y:.2f}", row=1, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # --- STRATEGY BASKET DISPLAY & MANAGEMENT ---
    st.markdown("---")
    st.subheader("🧺 Strategy Basket Analytics")

    if st.session_state["basket_legs"]:
        market_data_store = data.get("market_data", {})
        basket_tokens_info = data.get("basket_tokens_info", {})
        F_val = data.get("F", data["spot_price"])
        T_val = data.get("T", 1e-5)
        hv_val = data.get("index_hv", 0.15)

        calculated_legs = []
        tot_pnl = 0.0
        tot_delta = 0.0
        tot_gamma = 0.0
        tot_theta = 0.0
        tot_vega = 0.0

        for idx, leg in enumerate(st.session_state["basket_legs"]):
            k = leg["strike"]
            t = leg["type"]
            act = leg["action"]
            qty = leg["qty"]

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

            # MODIFIED: Scaled by Units/Qty and Direction
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
                "Leg": idx + 1,
                "Action": act,
                "Strike": k,
                "Type": t,
                "Qty": qty,
                "Entry (₹)": round(entry_p, 2),
                "LTP (₹)": round(ltp, 2),
                "P&L (₹)": round(leg_pnl, 2),
                "Delta (Δ ₹)": round(pos_delta, 2),
                "Gamma (γ)": round(pos_gamma, 4),
                "Theta (θ ₹/Day)": round(pos_theta, 2),
                "Vega (ν ₹)": round(pos_vega, 2)
            })

        with st.container(border=True):
            st.markdown("**Combined Basket Summary**")
            b_m1, b_m2, b_m3, b_m4, b_m5 = st.columns(5)
            b_m1.metric("Net P&L (₹)", f"₹{tot_pnl:,.2f}")
            b_m2.metric("Net Delta (Δ ₹)", f"₹{tot_delta:,.2f}")
            b_m3.metric("Net Gamma (γ)", f"{tot_gamma:.4f}")
            b_m4.metric("Net Theta (θ ₹/Day)", f"₹{tot_theta:,.2f}")
            b_m5.metric("Net Vega (ν ₹)", f"₹{tot_vega:,.2f}")

        df_basket_display = pd.DataFrame(calculated_legs)
        st.markdown("**Individual Legs Breakdown**")
        
        # Display Basket Legs Table with dynamic delete buttons
        col_tbl, col_del = st.columns([0.85, 0.15])
        with col_tbl:
            st.dataframe(df_basket_display, use_container_width=True)
        with col_del:
            st.markdown("**Manage Legs**")
            for idx in range(len(st.session_state["basket_legs"])):
                if st.button(f"❌ Remove #{idx + 1}", key=f"del_leg_{idx}"):
                    st.session_state["basket_legs"].pop(idx)
                    sync_local_storage()
                    st.rerun()

        # --- MODIFIED: STRATEGY BASKET HISTORICAL ANALYTICS CHARTS ---
        with st.expander("📊 Basket Historical Analytics (Cumulative Theta Decay %, Daily Theta, Vanna, Charm)", expanded=False):
            st.caption("Visualizing combined strategy basket time-series decay and second-order greeks at 1-hr interval sessions. Holidays excluded.")
            smart_api = get_smart_api_client()
            if smart_api and not df_master.empty:
                b_hist_dfs = []
                for leg in st.session_state["basket_legs"]:
                    k = leg["strike"]
                    t = leg["type"]
                    act = leg["action"]
                    qty = leg["qty"]
                    mult = 1.0 if act == "BUY" else -1.0

                    tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, k, t)
                    if tok:
                        leg_df = fetch_history(smart_api, tok, days=30, spot_token=default_token, spot_exchange=spot_exchange)
                        if not leg_df.empty:
                            leg_df['Expiry_Date'] = target_expiry_dt.date()
                            expiry = pd.to_datetime(leg_df['Expiry_Date']).dt.tz_localize(None)
                            timestamp = pd.to_datetime(leg_df['Raw_Timestamp']).dt.tz_localize(None)
                            leg_df['DTE'] = (expiry - timestamp).dt.total_seconds() / 86400
                            leg_df['DTE'] = leg_df['DTE'].apply(lambda x: max(x, 0.001))
                            leg_df['T'] = leg_df['DTE'] / 365.0
                            
                            greeks_df = leg_df.apply(compute_greeks, axis=1, K=k, r=rate_param, option_type=t)
                            greeks_df.columns = ['IV_%', 'Theta', 'Theta_Decay_Pct', 'Gamma', 'Vanna', 'Charm', 'Extrinsic_Val', 'Is_Pure_Intrinsic']
                            
                            init_p = leg_df.iloc[0]['Close']
                            
                            # Multiply by Qty / Units entered in sidebar
                            leg_df['Theta_Scaled'] = greeks_df['Theta'] * qty * mult
                            leg_df['Vanna_Scaled'] = greeks_df['Vanna'] * qty * mult
                            leg_df['Charm_Scaled'] = greeks_df['Charm'] * qty * mult
                            
                            # Theta Decay standalone computation (Absolute theta decay accrued)
                            init_theta = abs(greeks_df.iloc[0]['Theta']) * qty
                            cumulative_theta_loss = (init_p - leg_df['Close']) * qty if mult > 0 else (leg_df['Close'] - init_p) * qty
                            
                            leg_df['Theta_Accrued'] = cumulative_theta_loss
                            leg_df['Total_Expected_Theta'] = init_theta if init_theta > 0 else 1.0
                            
                            b_hist_dfs.append(leg_df[['Date', 'Raw_Timestamp', 'Theta_Scaled', 'Vanna_Scaled', 'Charm_Scaled', 'Theta_Accrued', 'Total_Expected_Theta']])

                if b_hist_dfs:
                    combined_basket_df = b_hist_dfs[0].copy()
                    for next_df in b_hist_dfs[1:]:
                        combined_basket_df = pd.merge(combined_basket_df, next_df, on=['Date', 'Raw_Timestamp'], how='inner', suffixes=('', '_sub'))
                        for c_col in ['Theta_Scaled', 'Vanna_Scaled', 'Charm_Scaled', 'Theta_Accrued', 'Total_Expected_Theta']:
                            combined_basket_df[c_col] = combined_basket_df[c_col] + combined_basket_df[f"{c_col}_sub"]
                            combined_basket_df.drop(columns=[f"{c_col}_sub"], inplace=True)

                    combined_basket_df = combined_basket_df.sort_values('Raw_Timestamp', ascending=True).reset_index(drop=True)

                    # MODIFIED: Calculate Cumulative Theta Decay Percentage (Max 100%)
                    total_init_exp_theta = max(abs(combined_basket_df.iloc[0]['Total_Expected_Theta']), 1e-5)
                    combined_basket_df['Cumulative_Theta_Decay_Pct'] = (combined_basket_df['Theta_Accrued'] / total_init_exp_theta) * 100.0
                    combined_basket_df['Cumulative_Theta_Decay_Pct'] = combined_basket_df['Cumulative_Theta_Decay_Pct'].clip(lower=0.0, upper=100.0)

                    # Helper function to plot basket greeks with non-trading days/holidays excluded using categorical x-axis
                    def plot_basket_metric(df, y_col, title, y_label, color, is_pct=False):
                        fig_b = px.line(df, x='Date', y=y_col, title=title, markers=True)
                        fig_b.update_traces(line_color=color)
                        # Hide non-trading days/holidays by setting x-axis type to "category"
                        fig_b.update_xaxes(type="category", title="Trading Session Intervals (Oldest ➔ Present)", autorange=True)
                        if is_pct:
                            fig_b.update_yaxes(title=y_label, range=[0, 105], ticksuffix="%")
                        else:
                            fig_b.update_yaxes(title=y_label)
                        fig_b.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                        return fig_b

                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.plotly_chart(plot_basket_metric(combined_basket_df, 'Cumulative_Theta_Decay_Pct', '1. Cumulative Theta Decay of Basket (%)', 'Cumulative Decay (%)', '#00E676', is_pct=True), use_container_width=True)
                    with bc2:
                        st.plotly_chart(plot_basket_metric(combined_basket_df, 'Theta_Scaled', '2. Daily Theta Decay Expected (₹ / Day)', 'Daily Theta (₹/Day)', '#FF9800'), use_container_width=True)

                    bc3, bc4 = st.columns(2)
                    with bc3:
                        st.plotly_chart(plot_basket_metric(combined_basket_df, 'Vanna_Scaled', '3. Combined Basket Vanna (dGamma / dVol ₹)', 'Vanna (₹)', '#00BFFF'), use_container_width=True)
                    with bc4:
                        st.plotly_chart(plot_basket_metric(combined_basket_df, 'Charm_Scaled', '4. Combined Basket Charm (Delta Decay ₹ / Day)', 'Charm (₹/Day)', '#E040FB'), use_container_width=True)
                else:
                    st.info("Unable to fetch historical data for strategy basket contracts.")
            else:
                st.info("API session uninitialized. Cannot calculate strategy basket time-series charts.")

    else:
        st.info("No legs added to strategy basket yet. Use sidebar **2. Build Strategy Basket** to add positions.")

    # --- OPTION CHAIN KEY LEVELS ---
    st.markdown("---")
    st.subheader("🎯 Option Chain Derived Key Levels & Targets")

    lvls = data.get("levels", {})
    if lvls:
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        row1_col1.metric("OI Resistance 1 (R1)", f"{lvls['R1']}")
        row1_col2.metric("OI Resistance 2 (R2)", f"{lvls['R2']}")
        row1_col3.metric("OI Support 1 (S1)", f"{lvls['S1']}")
        row1_col4.metric("OI Support 2 (S2)", f"{lvls['S2']}")

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        row2_col1.metric("GEX Support / Magnet", f"{lvls['GEX_Support']}")
        row2_col2.metric("GEX Resistance / Magnet", f"{lvls['GEX_Resistance']}")
        row2_col3.metric("Vol Accelerator (-GEX)", f"{lvls['GEX_Accelerator'] if lvls['GEX_Accelerator'] else 'None'}")
        row2_col4.metric("Zero Gamma / Flip Point", f"{lvls['Zero_Gamma_Flip']}")

        # --- GEX CHARTS ---
        st.markdown("---")
        df_chain = pd.DataFrame(data["chain_results"])

        if not df_chain.empty:
            df_chain["Total_Vol"] = df_chain["C_Vol"] + df_chain["P_Vol"]
            df_chain["Total_OI"] = df_chain["C_OI"] + df_chain["P_OI"]

            # Compute shared explicit Strike boundaries to force exact X-axis sync across all GEX charts
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

            # 1. GEX/OI CHART
            c_head1, c_head2 = st.columns([0.65, 0.35])
            with c_head1:
                st.subheader("📈 OI-Based Net Gamma Exposure vs Open Interest (OI)")
            with c_head2:
                st.markdown(f"<div class='update-timestamp'>Expiry: {selected_expiry_str} | Updated: {data['timestamp']}</div>", unsafe_allow_html=True)
            
            gex_oi_colors = np.where(df_chain["Net_GEX_OI"] >= 0, "#006400", "#8B0000")
            fig_oi = make_subplots(specs=[[{"secondary_y": True}]])
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["Net_GEX_OI"], name="Net Gamma (OI-Based)", marker_color=gex_oi_colors, opacity=0.85, width=25, hovertemplate="Strike: %{x}<br>Net GEX (OI): ₹%{y:,.0f}<extra></extra>"), secondary_y=True)
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["C_OI"], name="Call OI", marker_color="#2E7D32", opacity=0.6, hovertemplate="Strike: %{x}<br>Call OI: %{y:,}<extra></extra>"), secondary_y=False)
            fig_oi.add_trace(plt_go.Bar(x=df_chain["Strike"], y=-df_chain["P_OI"], name="Put OI", marker_color="#C62828", opacity=0.6, hovertemplate="Strike: %{x}<br>Put OI: %{customdata:,}<extra></extra>", customdata=df_chain["P_OI"]), secondary_y=False)
            fig_oi.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_oi.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            fig_oi.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip Point")
            oi1_range, oi2_range = calculate_synced_ranges(df_chain["C_OI"], -df_chain["P_OI"], df_chain["Net_GEX_OI"].clip(lower=0), df_chain["Net_GEX_OI"].clip(upper=0))
            fig_oi.update_layout(title="Strike-wise OI-Based Net Gamma & Open Interest Profile (Call OI Above / Put OI Below)", template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=480, barmode="overlay", margin=dict(l=20, r=20, t=40, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_oi.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_oi.update_yaxes(title_text="Put OI (Below) | Call OI (Above)", range=oi1_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            fig_oi.update_yaxes(title_text="Net GEX (OI-Based ₹)", range=oi2_range, secondary_y=True, showgrid=False, zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            st.plotly_chart(fig_oi, use_container_width=True)

            # 2. GEX/TRADE VOLUME CHART
            c_head1, c_head2 = st.columns([0.65, 0.35])
            with c_head1:
                st.subheader("📊 Volume-Based Net Gamma Exposure vs Trading Volume")
            with c_head2:
                st.markdown(f"<div class='update-timestamp'>Expiry: {selected_expiry_str} | Updated: {data['timestamp']}</div>", unsafe_allow_html=True)
            
            gex_vol_colors = np.where(df_chain["Net_GEX_Vol"] >= 0, "#006400", "#8B0000")
            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["Net_GEX_Vol"], name="Net Gamma (Vol-Based)", marker_color=gex_vol_colors, opacity=0.85, width=25, hovertemplate="Strike: %{x}<br>Net GEX (Vol): ₹%{y:,.0f}<extra></extra>"), secondary_y=True)
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["C_Vol"], name="Call Volume", marker_color="#81C784", opacity=0.6, hovertemplate="Strike: %{x}<br>Call Vol: %{y:,}<extra></extra>"), secondary_y=False)
            fig_vol.add_trace(plt_go.Bar(x=df_chain["Strike"], y=-df_chain["P_Vol"], name="Put Volume", marker_color="#FF8A80", opacity=0.6, hovertemplate="Strike: %{x}<br>Put Vol: %{customdata:,}<extra></extra>", customdata=df_chain["P_Vol"]), secondary_y=False)
            fig_vol.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_vol.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            fig_vol.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip Point")
            v1_range, v2_range = calculate_synced_ranges(df_chain["C_Vol"], -df_chain["P_Vol"], df_chain["Net_GEX_Vol"].clip(lower=0), df_chain["Net_GEX_Vol"].clip(upper=0))
            fig_vol.update_layout(title="Strike-wise Volume-Based Net Gamma & Volume Profile (Call Vol Above / Put Vol Below)", template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=480, barmode="overlay", margin=dict(l=20, r=20, t=40, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_vol.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_vol.update_yaxes(title_text="Put Vol (Below) | Call Vol (Above)", range=v1_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            fig_vol.update_yaxes(title_text="Net GEX (Vol-Based ₹)", range=v2_range, secondary_y=True, showgrid=False, zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            st.plotly_chart(fig_vol, use_container_width=True)

            # 3. DELTA-ADJUSTED GEX CHART
            c_head1, c_head2 = st.columns([0.65, 0.35])
            with c_head1:
                st.subheader("🎯 Delta-Adjusted Net Gamma Exposure Profile")
            with c_head2:
                st.markdown(f"<div class='update-timestamp'>Expiry: {selected_expiry_str} | Updated: {data['timestamp']}</div>", unsafe_allow_html=True)
            
            delta_gex_colors = np.where(df_chain["Net_Delta_GEX_OI"] >= 0, "#00E676", "#FF5252")
            fig_delta_gex = make_subplots(specs=[[{"secondary_y": False}]])
            fig_delta_gex.add_trace(plt_go.Bar(
                x=df_chain["Strike"], 
                y=df_chain["Net_Delta_GEX_OI"], 
                name="Delta-Adjusted Net GEX", 
                marker_color=delta_gex_colors, 
                opacity=0.85, 
                width=25, 
                hovertemplate="Strike: %{x}<br>Delta-Adjusted GEX: ₹%{y:,.0f}<extra></extra>"
            ))
            fig_delta_gex.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_delta_gex.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            fig_delta_gex.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip Point")
            fig_delta_gex.update_layout(
                title="Strike-wise Delta-Adjusted Net Gamma Exposure Profile", 
                template="plotly_dark", 
                paper_bgcolor="#0E1117", 
                plot_bgcolor="#0E1117", 
                height=480, 
                margin=dict(l=20, r=20, t=40, b=10), 
                hovermode="x unified", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_delta_gex.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_delta_gex.update_yaxes(title_text="Delta-Adjusted Net GEX (₹)", showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            st.plotly_chart(fig_delta_gex, use_container_width=True)

            # 3b. DELTA-ADJUSTED GEX HEATMAP (selected index)
            st.markdown("---")
            render_delta_gex_heatmap(
                data,
                Index_Name,
                selected_expiry_str,
                st.session_state.get("heatmap_timeframe", "5 min"),
            )

            # 4. VEX & CEX CHART
            c_head1, c_head2 = st.columns([0.65, 0.35])
            with c_head1:
                st.subheader("⚡ VEX (Vega Exposure) and CEX (Charm Exposure) Profile")
            with c_head2:
                st.markdown(f"<div class='update-timestamp'>Expiry: {selected_expiry_str} | Updated: {data['timestamp']}</div>", unsafe_allow_html=True)
            
            fig_vex_cex = make_subplots(specs=[[{"secondary_y": True}]])
            fig_vex_cex.add_trace(plt_go.Bar(x=df_chain["Strike"], y=df_chain["VEX"], name="VEX (Vega Exposure)", marker_color="#00E676", opacity=0.75, width=20, hovertemplate="Strike: %{x}<br>VEX: ₹%{y:,.2f}<extra></extra>"), secondary_y=False)
            fig_vex_cex.add_trace(plt_go.Scatter(x=df_chain["Strike"], y=df_chain["CEX"], name="CEX (Charm Exposure)", line=dict(color="#2196F3", width=2.5), mode="lines+markers", hovertemplate="Strike: %{x}<br>CEX: ₹%{y:,.2f}<extra></extra>"), secondary_y=True)
            fig_vex_cex.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_vex_cex.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            vex_range, cex_range = calculate_synced_ranges(df_chain["VEX"].clip(lower=0), df_chain["VEX"].clip(upper=0), df_chain["CEX"].clip(lower=0), df_chain["CEX"].clip(upper=0))
            fig_vex_cex.update_layout(title="Strike-wise VEX & CEX Profile", template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=450, margin=dict(l=20, r=20, t=40, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_vex_cex.update_xaxes(type="linear", tickformat="d", dtick=100, range=[min_strike_val, max_strike_val])
            fig_vex_cex.update_yaxes(title_text="VEX (Vega Exposure ₹)", range=vex_range, secondary_y=False, showgrid=True, gridcolor="#262930", zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            fig_vex_cex.update_yaxes(title_text="CEX (Charm Exposure ₹)", range=cex_range, secondary_y=True, showgrid=False, zeroline=True, zerolinecolor="#FFFFFF", zerolinewidth=1.5)
            st.plotly_chart(fig_vex_cex, use_container_width=True)

    # --- INCORPORATED BETA MODULE SECTIONS ---
    st.markdown("---")
    head_skew_c1, head_skew_c2 = st.columns([0.65, 0.35])
    with head_skew_c1:
        st.subheader(f"6️⃣ Market Volatility Skew Profile ({selected_expiry_str})")
    with head_skew_c2:
        st.markdown(f"<div class='update-timestamp'>Expiry: {selected_expiry_str} | Updated: {data.get('timestamp', '')}</div>", unsafe_allow_html=True)

    smart_api = get_smart_api_client()
    if smart_api and not df_master.empty:
        latest_spot = data['spot_price']
        
        df_chain_iv = fetch_and_compute_full_chain_iv(
            smart_api, 
            df_master, 
            Index_Name,
            Exchange,
            selected_expiry_str, 
            latest_spot, 
            rate_param, 
            strikes_below=strikes_below, 
            strikes_above=strikes_above
        )

        if not df_chain_iv.empty:
            tab1, tab2 = st.tabs(["Unsmoothed Market Skew (OTM Puts & Calls)", "Both Raw Curves (CE vs PE)"])
            
            with tab1:
                df_skew = get_clean_otm_skew(df_chain_iv, latest_spot)
                fig_skew = px.line(
                    df_skew, 
                    x='Strike', 
                    y='IV_%', 
                    title=f"{Index_Name} Pure OTM Volatility Skew (Spot: ₹{latest_spot:.2f})",
                    markers=True,
                    color_discrete_sequence=['#00bfff'],
                    hover_data=['Option_Type', 'LTP']
                )
                fig_skew.add_vline(x=latest_spot, line_dash="dash", line_color="white", annotation_text="Current Spot")
                fig_skew.update_xaxes(title="Strike Price (₹)")
                fig_skew.update_yaxes(title="Implied Volatility (IV %)")
                fig_skew.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_skew, use_container_width=True)
                
            with tab2:
                fig_raw = px.line(
                    df_chain_iv, 
                    x='Strike', 
                    y='IV_%', 
                    color='Option_Type', 
                    title=f"Raw CE & PE IV across Strikes (Spot: ₹{latest_spot:.2f})",
                    markers=True,
                    color_discrete_map={'CE': '#00cc96', 'PE': '#ff4136'},
                    hover_data=['LTP']
                )
                fig_raw.add_vline(x=latest_spot, line_dash="dash", line_color="white", annotation_text="Current Spot")
                fig_raw.update_xaxes(title="Strike Price (₹)")
                fig_raw.update_yaxes(title="Implied Volatility (IV %)")
                fig_raw.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_raw, use_container_width=True)

            with st.expander("🔻 Volatility Crash Analysis & Historical Option Decay Details", expanded=False):
                st.markdown("### Raw Volatility Values & Volatility Crash Status")
                st.dataframe(df_chain_iv, use_container_width=True)

live_dashboard_fragment()
institutional_order_flow_scanner_fragment()
zscore_analysis_fragment()
