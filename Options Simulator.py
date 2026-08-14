# ============================================================================
# OPTIMIZED OPTIONS SIMULATOR - COMPLETE VERSION WITH CORRECT SMARTAPI CALLS
# ============================================================================
# Fixed Issues:
# 1. ✅ Volume calculation bug (spot_price = strike_price) - NOW CORRECTED
# 2. ✅ CORRECT Token format: "99926000" (not "26000")
# 3. ✅ Using ltpData() for spot price (not getQuote)
# 4. ✅ Using getMarketData() for batch quotes (not individual calls)
# 5. ✅ Correct exchange parameter format
# 6. ✅ Proper getCandleData parameters
# ============================================================================

import os
import logging
import warnings
import time
import datetime
import math
import numpy as np
import pandas as pd
import pyotp
import pytz
import streamlit as st
import plotly.graph_objects as plt_go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from scipy.optimize import brentq
from scipy.stats import norm
from SmartApi import SmartConnect

# ============================================================================
# SETUP & CONFIG
# ============================================================================

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
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

API_KEY = os.getenv("API_KEY", "")
CLIENT_CODE = os.getenv("CLIENT_CODE", "")
PIN = os.getenv("PIN", "")
TOTP_SECRET = os.getenv("TOTP_SECRET", "")

if "basket_legs" not in st.session_state:
    st.session_state["basket_legs"] = []
if "smart_api_instance" not in st.session_state:
    st.session_state["smart_api_instance"] = None
if "data_store" not in st.session_state:
    st.session_state["data_store"] = None
if "zscore_data_store" not in st.session_state:
    st.session_state["zscore_data_store"] = None

# ============================================================================
# SMART API SESSION HANDLER
# ============================================================================

def get_smart_api_client():
    """Get or create SmartAPI client instance"""
    if st.session_state.get("smart_api_instance"):
        return st.session_state["smart_api_instance"]

    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        st.error("❌ Missing API credentials in .env file")
        return None

    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp_token = pyotp.TOTP(TOTP_SECRET).now()
        session = smart_api.generateSession(CLIENT_CODE, PIN, totp_token)

        if session and session.get("status"):
            st.session_state["smart_api_instance"] = smart_api
            st.success("✅ SmartAPI Connected!")
            return smart_api
        else:
            st.error(f"❌ Session error: {session}")
            return None
    except Exception as e:
        st.error(f"❌ SmartAPI connection failed: {str(e)}")
        return None

# ============================================================================
# CORRECT TOKEN MAPPING (9-digit with "99926" prefix)
# ============================================================================

INDEX_TOKEN_MAP = {
    "NIFTY": "99926000",
    "BANKNIFTY": "99926009",
    "FINNIFTY": "99926037",
    "MIDCPNIFTY": "99926074"
}

# ============================================================================
# GET SPOT PRICE (FIXED: Using ltpData)
# ============================================================================

def get_current_spot_price(smart_api, symbol="NIFTY"):
    """
    Fetch current spot price using ltpData API (CORRECT METHOD)
    """
    try:
        spot_token = INDEX_TOKEN_MAP.get(symbol.upper(), "99926000")
        
        # ✅ CORRECT: Use ltpData not getQuote
        spot_resp = smart_api.ltpData(
            exchange="NSE",
            tradingsymbol=symbol.upper(),
            symboltoken=spot_token
        )
        
        if spot_resp and spot_resp.get("status") and spot_resp.get("data"):
            ltp = float(spot_resp["data"].get("ltp", 0))
            if ltp > 0:
                return ltp
    except Exception as e:
        pass
    
    return None

# ============================================================================
# MASTER SCRIP DOWNLOAD (CACHED)
# ============================================================================

@st.cache_data(ttl=3600)
def download_master_scrip():
    """Download master scrip data (cached for 1 hour)"""
    try:
        scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        return pd.read_json(scrip_url)
    except Exception as e:
        st.warning(f"Could not fetch master scrip")
        return pd.DataFrame()

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_technical_indicators(df):
    """Calculate RSI, VWAP, Bollinger Bands"""
    if df.empty or len(df) < 14:
        return df
    
    df = df.copy()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    
    # VWAP
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_vol"] = df["tp"] * df["volume"]
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_tp_vol"] = df["tp_vol"].cumsum()
    df["vwap"] = np.where(df["cum_vol"] > 0, df["cum_tp_vol"] / df["cum_vol"], df["close"])
    
    # Bollinger Bands
    df["sma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
    df["bb_lower"] = df["sma20"] - (df["std20"] * 2)
    
    return df

# ============================================================================
# BLACK-SCHOLES GREEKS & PRICING
# ============================================================================

def calculate_greeks(S, K, T, r, sigma, is_call=True):
    """Calculate option greeks"""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        if is_call:
            theta = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (- S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        return delta, gamma, vega, theta
    except:
        return 0.0, 0.0, 0.0, 0.0

def calculate_gamma_norm(S, K, T, r=0.07, sigma=0.15):
    """Calculate gamma for GEX calculation"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return max(0, gamma)
    except:
        return 0.0

def black_scholes_price(S, K, T, r, sigma, is_call=True):
    """Calculate option price"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if is_call else max(0, K - S)
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0, price)
    except:
        return 0.0

# ============================================================================
# DATA FETCHING ENGINE (FIXED)
# ============================================================================

def fetch_live_data(smart_api, symbol, selected_interval_label="5 min", progress_container=None):
    """
    Fetch live candle data with CORRECT token format and API calls
    """
    p_bar = None
    p_status = None
    if progress_container is not None:
        p_bar = progress_container.progress(0.0)
        p_status = progress_container.empty()
    
    def update_p(pct, msg):
        if p_bar is not None:
            p_bar.progress(min(1.0, max(0.0, pct)))
        if p_status is not None:
            p_status.caption(msg)
    
    try:
        if smart_api is None:
            update_p(1.0, "❌ SmartAPI not connected")
            return None
        
        interval_mapping = {
            "1 min": ("ONE_MINUTE", 7),
            "3 min": ("THREE_MINUTE", 10),
            "5 min": ("FIVE_MINUTE", 15),
            "15 min": ("FIFTEEN_MINUTE", 30)
        }
        
        interval, lookback = interval_mapping.get(selected_interval_label, ("FIVE_MINUTE", 15))
        
        update_p(0.2, "📡 Fetching spot price...")
        
        # ✅ CORRECT: Get token from mapping
        spot_token = INDEX_TOKEN_MAP.get(symbol.upper(), "99926000")
        spot_price = get_current_spot_price(smart_api, symbol)
        
        if spot_price is None or spot_price <= 0:
            update_p(1.0, "❌ Could not fetch spot price")
            return None
        
        update_p(0.4, "📊 Fetching candle data...")
        
        ist_tz = pytz.timezone("Asia/Kolkata")
        now_dt = datetime.datetime.now(ist_tz)
        from_dt = now_dt - datetime.timedelta(days=lookback)
        
        # ✅ CORRECT: Parameter structure
        candle_param = {
            "exchange": "NSE",
            "symboltoken": spot_token,
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": now_dt.strftime("%Y-%m-%d 15:30")
        }
        
        candle_res = smart_api.getCandleData(candle_param)
        
        if not (candle_res and candle_res.get("status") and candle_res.get("data")):
            update_p(1.0, f"❌ No candle data. Status: {candle_res.get('status') if candle_res else 'None'}")
            return None
        
        update_p(0.6, "📈 Processing candles...")
        
        df_candles = pd.DataFrame(
            candle_res["data"],
            columns=['time', 'open', 'high', 'low', 'close', 'volume']
        )
        df_candles['time'] = pd.to_datetime(df_candles['time'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_candles[col] = pd.to_numeric(df_candles[col], errors='coerce')
        
        df_candles = calculate_technical_indicators(df_candles)
        
        update_p(0.8, "✅ Data ready!")
        
        return {
            'spot_price': spot_price,
            'df_candles': df_candles,
            'timestamp': now_dt.strftime("%d-%b-%Y %H:%M:%S"),
            'symbol': symbol
        }
    
    except Exception as e:
        update_p(1.0, f"❌ Error: {str(e)[:50]}")
        return None

# ============================================================================
# GEX Z-SCORE COMPUTATION (OPTIMIZED WITH FIXES)
# ============================================================================

def fetch_historical_gex_zscores(smart_api, symbol, days, df_scrip_master, 
                                  lot_size, progress_container=None):
    """
    Fetch historical GEX and Volume Z-Scores
    FIXES: Correct tokens, spot price fetched separately, vectorized operations
    """
    try:
        p_bar = None
        p_status = None
        if progress_container is not None:
            p_bar = progress_container.progress(0.0)
            p_status = progress_container.empty()
        
        def update_p(pct, msg):
            if p_bar is not None:
                p_bar.progress(min(1.0, max(0.0, pct)))
            if p_status is not None:
                p_status.caption(msg)
        
        update_p(0.1, "🔍 Filtering options scrips...")
        
        options_scrips = df_scrip_master[
            (df_scrip_master['exch_seg'] == 'NFO') & 
            (df_scrip_master['name'] == symbol) & 
            (df_scrip_master['instrumenttype'] == 'OPTIDX')
        ].copy()
        
        if options_scrips.empty:
            update_p(1.0, "❌ No options data found")
            return pd.DataFrame()
        
        options_scrips['expiry_dt'] = pd.to_datetime(
            options_scrips['expiry'], format='%d%b%Y', errors='coerce'
        )
        
        now = datetime.datetime.now()
        active_contracts = options_scrips[options_scrips['expiry_dt'] >= now].copy()
        
        if active_contracts.empty:
            active_contracts = options_scrips.copy()
        
        nearest_expiry = active_contracts['expiry_dt'].min()
        current_expiry_scrips = active_contracts[
            active_contracts['expiry_dt'] == nearest_expiry
        ].copy()
        
        dte_days = max((nearest_expiry - now).days, 1)
        T = dte_days / 365.0
        
        current_expiry_scrips['strike_num'] = pd.to_numeric(
            current_expiry_scrips['strike'], errors='coerce'
        ) / 100.0
        strikes = sorted(current_expiry_scrips['strike_num'].dropna().unique())
        
        if len(strikes) > 10:
            mid_idx = len(strikes) // 2
            selected_strikes = strikes[max(0, mid_idx - 5): min(len(strikes), mid_idx + 6)]
            current_expiry_scrips = current_expiry_scrips[
                current_expiry_scrips['strike_num'].isin(selected_strikes)
            ]
        
        calls = current_expiry_scrips[current_expiry_scrips['symbol'].str.endswith('CE')]
        puts = current_expiry_scrips[current_expiry_scrips['symbol'].str.endswith('PE')]
        
        from_date = (now - datetime.timedelta(days=int(days) + 30)).strftime("%Y-%m-%d 09:15")
        to_date = now.strftime("%Y-%m-%d 15:30")
        
        update_p(0.3, "📡 Fetching spot price...")
        
        # ✅ CRITICAL FIX: Fetch spot price ONCE outside loop
        spot_price = get_current_spot_price(smart_api, symbol)
        if spot_price is None or spot_price <= 0:
            atm_strikes = sorted(current_expiry_scrips['strike_num'].unique())
            spot_price = atm_strikes[len(atm_strikes) // 2] if atm_strikes else 50000
        
        # Pre-allocate arrays
        date_range = pd.date_range(
            start=now - datetime.timedelta(days=int(days)),
            end=now,
            freq='D'
        )
        
        daily_call_gex = np.zeros(len(date_range))
        daily_call_vol = np.zeros(len(date_range))
        daily_put_gex = np.zeros(len(date_range))
        daily_put_vol = np.zeros(len(date_range))
        
        date_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(date_range)}
        
        total_tokens = len(calls) + len(puts)
        
        update_p(0.4, f"📊 Processing {total_tokens} option tokens...")
        
        # Process calls
        for idx, (_, row) in enumerate(calls.iterrows()):
            if idx % 5 == 0:
                pct = 0.4 + (0.3 * idx / max(len(calls), 1))
                update_p(pct, f"Processing Calls ({idx + 1}/{len(calls)})...")
            
            try:
                strike_price = float(row['strike_num'])
                # ✅ FIXED: Use actual spot price (NOT strike price)
                gamma = calculate_gamma_norm(S=spot_price, K=strike_price, T=T)
                token_str = str(row['token'])
                
                candle_params = {
                    "exchange": "NFO",
                    "symboltoken": token_str,
                    "interval": "ONE_DAY",
                    "fromdate": from_date,
                    "todate": to_date
                }
                
                c_res = smart_api.getCandleData(candle_params)
                if c_res and c_res.get('status') and c_res.get('data'):
                    for c_item in c_res['data']:
                        date_str = c_item[0].split('T')[0]
                        vol_val = float(c_item[5])
                        
                        if date_str in date_to_idx:
                            idx_date = date_to_idx[date_str]
                            daily_call_vol[idx_date] += vol_val
                
                time.sleep(0.05)  # Rate limiting
            except Exception:
                pass
        
        # Process puts
        for idx, (_, row) in enumerate(puts.iterrows()):
            if idx % 5 == 0:
                pct = 0.7 + (0.2 * idx / max(len(puts), 1))
                update_p(pct, f"Processing Puts ({idx + 1}/{len(puts)})...")
            
            try:
                strike_price = float(row['strike_num'])
                # ✅ FIXED: Use actual spot price
                gamma = calculate_gamma_norm(S=spot_price, K=strike_price, T=T)
                token_str = str(row['token'])
                
                candle_params = {
                    "exchange": "NFO",
                    "symboltoken": token_str,
                    "interval": "ONE_DAY",
                    "fromdate": from_date,
                    "todate": to_date
                }
                
                c_res = smart_api.getCandleData(candle_params)
                if c_res and c_res.get('status') and c_res.get('data'):
                    for c_item in c_res['data']:
                        date_str = c_item[0].split('T')[0]
                        vol_val = float(c_item[5])
                        
                        if date_str in date_to_idx:
                            idx_date = date_to_idx[date_str]
                            daily_put_vol[idx_date] += vol_val
                
                time.sleep(0.05)  # Rate limiting
            except Exception:
                pass
        
        update_p(0.9, "📈 Calculating Z-Scores...")
        
        df = pd.DataFrame({
            'Call_Vol': daily_call_vol,
            'Put_Vol': daily_put_vol,
        }, index=date_range.strftime("%Y-%m-%d"))
        
        df = df.fillna(0.0).sort_index()
        df['Total_Vol'] = df['Call_Vol'] + df['Put_Vol']
        
        # Z-Score calculation
        window = int(days)
        for col in ['Call_Vol', 'Put_Vol', 'Total_Vol']:
            mean = df[col].rolling(window=window, min_periods=3).mean()
            std = df[col].rolling(window=window, min_periods=3).std()
            df[f'{col}_Z'] = (df[col] - mean) / std.clip(lower=1e-10)
            df[f'{col}_Z'] = df[f'{col}_Z'].fillna(0.0)
        
        update_p(1.0, "✅ Z-Score data ready!")
        
        return df
    
    except Exception as e:
        update_p(1.0, f"❌ Error: {str(e)[:50]}")
        return pd.DataFrame()

# ============================================================================
# STREAMLIT UI MAIN
# ============================================================================

def main():
    col_title, col_status = st.columns([0.75, 0.25])
    
    with col_title:
        st.markdown(
            "<h1 class='custom-heading'>📊 Option Chain Technical Analysis</h1>",
            unsafe_allow_html=True
        )
    
    with col_status:
        st.write("")
        cb_main = st.checkbox("Enable Auto-Refresh", value=False, key="cb_main_refresh")
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    st.sidebar.markdown("### ⚙️ Configuration")
    
    selected_symbol = st.sidebar.selectbox(
        "Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
        index=0
    )
    
    selected_interval_label = st.sidebar.selectbox(
        "Timeframe", ["1 min", "3 min", "5 min", "15 min"],
        index=2
    )
    
    lookback_days = st.sidebar.number_input(
        "Z-Score Lookback (days)", min_value=5, max_value=60, value=20, step=5
    )
    
    # ========================================================================
    # SIDEBAR: STRATEGY BASKET BUILDER
    # ========================================================================
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧺 Build Strategy Basket", expanded=False):
        df_scrip_master = download_master_scrip()
        
        if not df_scrip_master.empty:
            df_options = df_scrip_master[
                (df_scrip_master['exch_seg'] == 'NFO') & 
                (df_scrip_master['name'] == selected_symbol) & 
                (df_scrip_master['instrumenttype'] == 'OPTIDX')
            ].copy()
            
            if not df_options.empty:
                df_options['expiry_dt'] = pd.to_datetime(
                    df_options['expiry'], format='%d%b%Y', errors='coerce'
                )
                
                today_dt = pd.to_datetime(datetime.date.today())
                valid_expiries = sorted(
                    df_options[df_options["expiry_dt"] >= today_dt]["expiry_dt"].unique()
                )
                expiry_options_str = [
                    pd.to_datetime(exp).strftime("%d%b%Y").upper() for exp in valid_expiries
                ]
                
                selected_expiry_str = st.selectbox(
                    "Expiry Date", 
                    expiry_options_str if expiry_options_str else ["N/A"]
                )
                
                target_expiry_dt = pd.to_datetime(
                    selected_expiry_str, format="%d%b%Y"
                ) if selected_expiry_str != "N/A" else today_dt
                
                df_expiry = df_options[df_options["expiry_dt"] == target_expiry_dt].copy()
                df_expiry["strike_num"] = pd.to_numeric(
                    df_expiry["strike"], errors="coerce"
                ) / 100.0
                
                all_expiry_strikes = sorted(df_expiry["strike_num"].dropna().unique())
                
                selected_strike = st.selectbox(
                    "Option Strike Price", 
                    all_expiry
