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

# Setup .streamlit/config.toml programmatically for forced dark theme
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

# Custom CSS Styling (Includes anti-dimming and hidden spinner rules)
custom_css = """
<style>
/* Disable Streamlit's default screen dimming/opacity reduction on rerun */
.stApp {
    opacity: 1 !important;
}

/* Hide the native running spinner / loading indicator overlay that causes flashes */
[data-testid="stStatusWidget"], .stSpinner {
    display: none !important;
}

/* Keep running fragments fully opaque */
div[data-testid="stFragment"] {
    opacity: 1 !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117 !important;
    color: #FAFAFA !important;
}

section[data-testid="stSidebar"] {
    width: 310px !important;
}

.block-container {
    padding-top: 1.0rem !important;
    padding-bottom: 1rem !important;
}

header[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0) !important;
}

h1, h2, h3, .custom-heading {
    color: #00E676 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stMetricValue"] {
    font-size: 18px !important;
    color: #00E676 !important;
}

.update-timestamp {
    font-size: 13px;
    color: #00E676;
    font-weight: 600;
    text-align: right;
}

div[data-baseweb="select"] > div {
    background-color: #1E222D !important;
    color: #FAFAFA !important;
    border-color: #363C4E !important;
}

.stButton>button {
    border-radius: 6px;
    font-weight: 600;
}

.status-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-right: 8px;
}
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

LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50
}

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
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        pdf_d1 = cls._norm_pdf(d1)

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = (S * pdf_d1 * math.sqrt(T)) / 100.0

        if flag.lower() == "c":
            delta = cls._norm_cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls._norm_cdf(d2)) / 365.0
        else:
            delta = cls._norm_cdf(d1) - 1.0
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls._norm_cdf(-d2)) / 365.0

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

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
            hist_data = smart_api.getCandleData(param)
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

# --- OPTION CHAIN & LEVEL CALCULATOR ---
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

    r1, r2 = min(r1, r2), max(r1, r2)
    s1, s2 = max(s1, s2), min(s1, s2)

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

    df["date_group"] = pd.to_datetime(df["time"]).dt.date
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_vol"] = df["tp"] * df["volume"]

    df["cum_vol"] = df.groupby("date_group")["volume"].cumsum()
    df["cum_tp_vol"] = df.groupby("date_group")["tp_vol"].cumsum()
    df["vwap"] = np.where(df["cum_vol"] > 0, df["cum_tp_vol"] / df["cum_vol"], df["close"])

    return df

@st.cache_data(ttl=3600)
def download_master_scrip():
    scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    return pd.read_json(scrip_url)

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

# --- GEX & TRADE VOLUME Z-SCORE COMPUTATION HELPERS ---
def calculate_gamma_norm(S, K, T, r=0.07, sigma=0.15):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def fetch_historical_gex_zscores(smart_api, symbol, days, df_scrip_master, lot_size, progress_container=None):
    try:
        options_scrips = df_scrip_master[
            (df_scrip_master['exch_seg'] == 'NFO') & 
            (df_scrip_master['name'] == symbol) & 
            (df_scrip_master['instrumenttype'] == 'OPTIDX')
        ].copy()

        if options_scrips.empty:
            return pd.DataFrame()

        options_scrips['expiry_dt'] = pd.to_datetime(options_scrips['expiry'], format='%d%b%Y', errors='coerce')
        now = datetime.datetime.now()
        active_contracts = options_scrips[options_scrips['expiry_dt'] >= now].copy()
        if active_contracts.empty:
            active_contracts = options_scrips.copy()

        nearest_expiry = active_contracts['expiry_dt'].min()
        current_expiry_scrips = active_contracts[active_contracts['expiry_dt'] == nearest_expiry].copy()

        dte_days = max((nearest_expiry - now).days, 1)
        T = dte_days / 365.0

        current_expiry_scrips['strike_num'] = pd.to_numeric(current_expiry_scrips['strike'], errors='coerce') / 100.0
        strikes = sorted(current_expiry_scrips['strike_num'].dropna().unique())
        if len(strikes) > 10:
            mid_idx = len(strikes) // 2
            selected_strikes = strikes[max(0, mid_idx - 5): min(len(strikes), mid_idx + 5)]
            current_expiry_scrips = current_expiry_scrips[current_expiry_scrips['strike_num'].isin(selected_strikes)]

        calls = current_expiry_scrips[current_expiry_scrips['symbol'].str.endswith('CE')]
        puts = current_expiry_scrips[current_expiry_scrips['symbol'].str.endswith('PE')]

        from_date = (now - datetime.timedelta(days=int(days) + 30)).strftime("%Y-%m-%d 09:15")
        to_date = now.strftime("%Y-%m-%d 15:30")

        daily_call_gex, daily_call_vol = {}, {}
        daily_put_gex, daily_put_vol = {}, {}

        total_tokens = len(calls) + len(puts)
        processed_count = 0

        p_bar = None
        p_status = None
        if progress_container is not None:
            p_bar = progress_container.progress(0.0)
            p_status = progress_container.empty()

        def process_options(df_tokens, target_gex, target_vol, is_call=True):
            nonlocal processed_count
            for _, row in df_tokens.iterrows():
                processed_count += 1
                if p_bar is not None and total_tokens > 0:
                    pct = min(1.0, processed_count / total_tokens)
                    p_bar.progress(pct)
                    opt_label = "Calls" if is_call else "Puts"
                    if p_status is not None:
                        p_status.caption(f"⏳ Fetching historical data for {symbol} {opt_label} ({processed_count}/{total_tokens})...")

                strike_price = float(row['strike_num'])
                spot_price = strike_price
                gamma = calculate_gamma_norm(S=spot_price, K=strike_price, T=T)
                token_str = str(row['token'])

                candle_params = {
                    "exchange": "NFO",
                    "symboltoken": token_str,
                    "interval": "ONE_DAY",
                    "fromdate": from_date,
                    "todate": to_date
                }
                try:
                    c_res = smart_api.getCandleData(candle_params)
                    if c_res and c_res.get('status') and c_res.get('data'):
                        for c_item in c_res['data']:
                            date_str = c_item[0].split('T')[0]
                            vol_val = float(c_item[5])
                            target_vol[date_str] = target_vol.get(date_str, 0.0) + vol_val
                except Exception:
                    pass

                oi_params = {
                    "exchange": "NFO",
                    "symboltoken": token_str,
                    "interval": "ONE_DAY",
                    "fromdate": from_date,
                    "todate": to_date
                }
                try:
                    oi_res = smart_api.getOIData(oi_params)
                    if oi_res and oi_res.get('status') and oi_res.get('data'):
                        for item in oi_res['data']:
                            date_str = item.get('time', '').split('T')[0]
                            oi_val = float(item.get('oi', 0))
                            gex_val = gamma * oi_val * lot_size * (spot_price ** 2) * 0.01
                            if not is_call:
                                gex_val = -gex_val
                            target_gex[date_str] = target_gex.get(date_str, 0.0) + gex_val
                except Exception:
                    pass

        process_options(calls, daily_call_gex, daily_call_vol, is_call=True)
        process_options(puts, daily_put_gex, daily_put_vol, is_call=False)

        if p_status is not None:
            p_status.caption("✅ Calculating Z-Score rolling metrics...")

        df = pd.DataFrame({
            'Call_GEX': daily_call_gex,
            'Put_GEX': daily_put_gex,
            'Call_Vol': daily_call_vol,
            'Put_Vol': daily_put_vol
        }).fillna(0.0).sort_index()

        df['Net_GEX'] = df['Call_GEX'] + df['Put_GEX']

        target_cols = ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_Vol', 'Put_Vol']
        for col in target_cols:
            mean = df[col].rolling(window=int(days), min_periods=3).mean()
            std = df[col].rolling(window=int(days), min_periods=3).std()
            z = (df[col] - mean) / std.replace(0, np.nan)
            df[f'{col}_Z'] = z.fillna(0.0)

        if p_bar is not None:
            p_bar.empty()
        if p_status is not None:
            p_status.empty()

        return df.fillna(0.0)
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
        Index_Name = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
    with c2:
        Exchange = st.selectbox("Exchange", ["NFO", "BFO"])

    rate_param = st.number_input("Risk Free Rate (r)", min_value=0.0, max_value=0.15, value=0.10, step=0.01)

    df_options = df_master[
        (df_master["exch_seg"] == Exchange)
        & (df_master["name"] == Index_Name)
        & (df_master["instrumenttype"].isin(["OPTIDX", "OPTSTK"]))
    ].copy()

    df_options["expiry_dt"] = pd.to_datetime(df_options["expiry"], format="%d%b%Y")
    today_dt = pd.to_datetime(datetime.date.today())
    valid_expiries = sorted(df_options[df_options["expiry_dt"] >= today_dt]["expiry_dt"].unique())
    expiry_options_str = [pd.to_datetime(exp).strftime("%d%b%Y").upper() for exp in valid_expiries]

    selected_expiry_str = st.selectbox("Expiry Date", expiry_options_str if expiry_options_str else ["N/A"])

    c3, c4 = st.columns(2)
    with c3:
        strikes_below = st.number_input("Below ATM", min_value=1, max_value=50, value=10, step=1)
    with c4:
        strikes_above = st.number_input("Above ATM", min_value=1, max_value=50, value=10, step=1)

    hv_days = st.slider("HV Lookback (Days)", min_value=10, max_value=90, value=30, step=5)

target_expiry_dt = pd.to_datetime(selected_expiry_str, format="%d%b%Y") if selected_expiry_str != "N/A" else today_dt
df_expiry = df_options[df_options["expiry_dt"] == target_expiry_dt].copy()
df_expiry["strike_num"] = pd.to_numeric(df_expiry["strike"], errors="coerce") / 100.0

all_expiry_strikes = sorted(df_expiry["strike_num"].dropna().unique())

with st.sidebar.expander("2. Build Strategy Basket", expanded=True):
    selected_strike = st.selectbox("Option Strike Price", all_expiry_strikes if all_expiry_strikes else [24500])

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        opt_type = st.selectbox("Option Type", ["CE", "PE"])
    with b_col2:
        trade_action = st.selectbox("Trade Action", ["BUY", "SELL"])

    entry_price_input = st.number_input("Entry Price (₹) [0 for LTP]", min_value=0.0, value=0.0, step=0.5)
    qty_lots = st.number_input("Quantity / Units", min_value=1, value=65, step=1)

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
            st.success("Leg Added!")
            st.rerun()
    with c_btn2:
        if st.button("Clear Basket", use_container_width=True):
            st.session_state["basket_legs"] = []
            st.rerun()

run_btn = st.sidebar.button("🚀 Fetch Chain & Greeks", use_container_width=True)

interval_mapping = {
    "1 min": ("ONE_MINUTE", 7),
    "3 min": ("THREE_MINUTE", 10),
    "5 min": ("FIVE_MINUTE", 15),
    "15 min": ("FIFTEEN_MINUTE", 30)
}

# --- SECURE SESSION HANDLER (PREVENTS RATE LIMIT BAN) ---
def get_smart_api_client():
    if "smart_api_instance" in st.session_state and st.session_state["smart_api_instance"] is not None:
        return st.session_state["smart_api_instance"]

    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        return None

    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp_token = pyotp.TOTP(TOTP_SECRET).now()
        session = smart_api.generateSession(CLIENT_CODE, PIN, totp_token)

        if session.get("status"):
            st.session_state["smart_api_instance"] = smart_api
            return smart_api
    except Exception:
        pass
    return None

# --- DATA FETCHING ENGINE ---
def fetch_live_data(selected_interval_label="5 min"):
    smart_api = get_smart_api_client()
    if not smart_api:
        st.error("Missing credentials or failed to generate SmartAPI session! Check Render Environment Variables.")
        return None

    try:
        index_token_map = {"NIFTY": "99926000", "BANKNIFTY": "99926009", "FINNIFTY": "99926037", "MIDCPNIFTY": "99926074"}
        spot_token = index_token_map.get(Index_Name, "99926000")

        spot_resp = smart_api.ltpData(exchange="NSE", tradingsymbol=Index_Name, symboltoken=spot_token)
        spot_price = float(spot_resp["data"]["ltp"]) if spot_resp and spot_resp.get("status") and spot_resp.get("data") else 24500.0

        index_hv = VolatilityEngine.calculate_hv(smart_api, spot_token, "NSE", days=hv_days)

        ist_tz = pytz.timezone("Asia/Kolkata")
        now_dt = datetime.datetime.now(ist_tz)

        api_interval, lookback_days = interval_mapping.get(selected_interval_label, ("FIVE_MINUTE", 15))
        from_dt = now_dt - datetime.timedelta(days=lookback_days)

        candle_param = {
            "exchange": "NSE",
            "symboltoken": spot_token,
            "interval": api_interval,
            "fromdate": from_dt.strftime("%Y-%m-%d 09:15"),
            "todate": now_dt.strftime("%Y-%m-%d 15:30")
        }
        candle_res = smart_api.getCandleData(candle_param)
        df_candles = pd.DataFrame()

        if candle_res and candle_res.get("status") and candle_res.get("data"):
            df_candles = pd.DataFrame(candle_res["data"], columns=["time", "open", "high", "low", "close", "volume"])
            df_candles[["open", "high", "low", "close", "volume"]] = df_candles[["open", "high", "low", "close", "volume"]].astype(float)
            df_candles = compute_technical_indicators(df_candles)

        expiry_datetime = target_expiry_dt.replace(hour=15, minute=30, second=0).tz_localize("Asia/Kolkata")
        time_diff_seconds = (expiry_datetime - now_dt).total_seconds()
        T = max(time_diff_seconds / (365.0 * 24 * 3600), 1e-5)

        atm_strike = min(all_expiry_strikes, key=lambda x: abs(x - spot_price)) if all_expiry_strikes else spot_price
        atm_idx = all_expiry_strikes.index(atm_strike) if all_expiry_strikes else 0
        filtered_strikes = all_expiry_strikes[max(0, atm_idx - strikes_below): min(len(all_expiry_strikes), atm_idx + strikes_above + 1)] if all_expiry_strikes else []

        tokens_to_fetch = set()
        strike_mapping = []

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

        market_data = {}
        chunk_size = 40
        for i in range(0, len(tokens_to_fetch_list), chunk_size):
            chunk = tokens_to_fetch_list[i:i + chunk_size]
            res = smart_api.getMarketData("FULL", {Exchange: chunk})
            if res and res.get("status") and res.get("data") and res["data"].get("fetched"):
                for item in res["data"]["fetched"]:
                    vol_val = (
                        item.get("volume") or
                        item.get("totTrdVol") or
                        item.get("volumeTraded") or
                        item.get("tradeVolume") or
                        item.get("v") or 0
                    )

                    market_data[str(item["symbolToken"])] = {
                        "ltp": float(item.get("ltp", 0.0)),
                        "oi": int(item.get("opnInterest", 0)),
                        "pnl_oi": int(item.get("netChange", 0)),
                        "volume": int(vol_val)
                    }
            time.sleep(0.1) # Prevents rate-limiting

        atm_c_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, int(atm_strike), "CE")
        atm_p_tok = get_smartapi_token(df_expiry, Index_Name, target_expiry_dt, int(atm_strike), "PE")

        atm_c_ltp = market_data.get(atm_c_tok, {}).get("ltp", 0.0)
        atm_p_ltp = market_data.get(atm_p_tok, {}).get("ltp", 0.0)

        F = atm_strike + math.exp(rate_param * T) * (atm_c_ltp - atm_p_ltp) if (atm_c_ltp > 0 and atm_p_ltp > 0) else spot_price

        lot_size = LOT_SIZES.get(Index_Name, 25)
        chain_results = []
        total_call_oi = total_put_oi = 0
        total_net_gex_oi = 0.0
        total_net_gex_vol = 0.0
        all_ivs = []

        for row in strike_mapping:
            K = row["strike"]
            c_info = market_data.get(row["call_tok"], {"ltp": 0.0, "oi": 0, "pnl_oi": 0, "volume": 0})
            p_info = market_data.get(row["put_tok"], {"ltp": 0.0, "oi": 0, "pnl_oi": 0, "volume": 0})

            iv = VolatilityEngine.calculate_iv(c_info["ltp"] if K >= F else p_info["ltp"], F, K, T, rate_param, "c" if K >= F else "p")
            if iv == 0.0:
                iv = index_hv

            c_greeks = VolatilityEngine.calculate_greeks(F, K, T, rate_param, iv, "c")
            p_greeks = VolatilityEngine.calculate_greeks(F, K, T, rate_param, iv, "p")

            call_gex_oi = c_greeks["gamma"] * c_info["oi"] * spot_price * lot_size
            put_gex_oi = p_greeks["gamma"] * p_info["oi"] * spot_price * lot_size
            net_gex_oi = call_gex_oi - put_gex_oi
            total_net_gex_oi += net_gex_oi

            call_gex_vol = c_greeks["gamma"] * c_info["volume"] * spot_price * lot_size
            put_gex_vol = p_greeks["gamma"] * p_info["volume"] * spot_price * lot_size
            net_gex_vol = call_gex_vol - put_gex_vol
            total_net_gex_vol += net_gex_vol

            if iv > 0: all_ivs.append(iv)
            total_call_oi += c_info["oi"]
            total_put_oi += p_info["oi"]

            chain_results.append({
                "C_Vol": c_info["volume"], "C_ΔOI": c_info["pnl_oi"], "C_OI": c_info["oi"],
                "C_Δ": round(c_greeks["delta"], 2), "C_γ": round(c_greeks["gamma"], 4),
                "C_θ": round(c_greeks["theta"], 2), "C_ν": round(c_greeks["vega"], 2),
                "C_IV_val": iv, "C_IV": f"{iv * 100:.1f}%", "C_LTP": c_info["ltp"],
                "Strike": K,
                "Net_GEX_OI": round(net_gex_oi, 2),
                "Net_GEX_Vol": round(net_gex_vol, 2),
                "P_LTP": p_info["ltp"], "P_IV_val": iv, "P_IV": f"{iv * 100:.1f}%",
                "P_Δ": round(p_greeks["delta"], 2), "P_γ": round(p_greeks["gamma"], 4),
                "P_θ": round(p_greeks["theta"], 2), "P_ν": round(p_greeks["vega"], 2),
                "P_OI": p_info["oi"], "P_ΔOI": p_info["pnl_oi"], "P_Vol": p_info["volume"]
            })

        pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else 0.0
        iv_percentile = 0.0
        if all_ivs:
            min_iv, max_iv = min(all_ivs), max(all_ivs)
            atm_c_iv = VolatilityEngine.calculate_iv(atm_c_ltp, F, atm_strike, T, rate_param, "c")
            if max_iv > min_iv and atm_c_iv > 0:
                iv_percentile = ((atm_c_iv - min_iv) / (max_iv - min_iv)) * 100.0

        max_pain_strike = VolatilityEngine.calculate_max_pain(chain_results)
        levels = calculate_support_resistance_targets(chain_results, spot_price, max_pain_strike)

        return {
            "spot_price": spot_price, "F": F, "T": T, "index_hv": index_hv, "iv_percentile": iv_percentile,
            "pcr": pcr, "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "total_net_gex_oi": total_net_gex_oi, "total_net_gex_vol": total_net_gex_vol,
            "max_pain_strike": max_pain_strike, "levels": levels, "df_candles": df_candles,
            "chain_results": chain_results, "market_data": market_data, "basket_tokens_info": basket_tokens_info,
            "timestamp": now_dt.strftime("%d-%b-%Y %H:%M:%S IST")
        }

    except Exception as e:
        if "exceeding access rate" in str(e).lower() or "access denied" in str(e).lower():
            st.session_state["smart_api_instance"] = None
        st.warning(f"API Rate limit / sync notice: Retrying on next cycle...")
        return None

if run_btn or "data_store" not in st.session_state:
    new_data = fetch_live_data(st.session_state["selected_timeframe"])
    if new_data:
        st.session_state["data_store"] = new_data

# --- SEPARATE FRAGMENT FOR GEX & VOLUME Z-SCORE ENGINE (5-MIN REFRESH OPTION) ---
@st.fragment(run_every=300 if st.session_state.get("enable_zscore_refresh", False) else None)
def zscore_analysis_fragment():
    st.markdown("---")
    head_c1, head_c2 = st.columns([0.65, 0.35])
    with head_c1:
        st.subheader(f"⚡ GEX & Trade Volume Z-Scores Analysis ({Index_Name})")
        st.caption(f"Lookback Window: **{hv_days} Days** (Uses HV Lookback setting from sidebar)")
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
            lot_size = LOT_SIZES.get(Index_Name, 25)
            z_df = fetch_historical_gex_zscores(smart_api, Index_Name, hv_days, df_master, lot_size, progress_container=progress_holder)
            st.session_state["zscore_data_store"] = z_df
        else:
            st.error("SmartAPI Session is uninitialized. Cannot calculate Z-Scores.")

    z_df = st.session_state["zscore_data_store"]

    if not z_df.empty:
        last_5 = z_df.tail(5).copy()

        def style_z(val):
            if val >= 1.5:
                return 'background-color: #ff4d4d; color: white; font-weight: bold;'
            elif val <= -1.5:
                return 'background-color: #4da6ff; color: white; font-weight: bold;'
            elif 0.5 <= val < 1.5:
                return 'background-color: #ffea80; color: black;'
            else:
                return 'background-color: #b3ffb3; color: black;'

        out_cols = ['Call_GEX_Z', 'Put_GEX_Z', 'Net_GEX_Z', 'Call_Vol_Z', 'Put_Vol_Z']
        
        styled_z_df = last_5[out_cols].style.applymap(
            style_z,
            subset=out_cols
        ).format({col: "{:.2f}" for col in out_cols})

        st.dataframe(styled_z_df, use_container_width=True)

        with st.expander("🔍 Inspect Raw Calculated Daily Totals (GEX & Volume)"):
            st.dataframe(z_df[['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_Vol', 'Put_Vol']].tail(10), use_container_width=True)
    else:
        st.info("Click '🔄 Compute / Refresh Z-Scores' above to load historical Z-Score analysis.")

# --- AUTO-REFRESHING FRAGMENT FOR MAIN DASHBOARD (5s REFRESH OPTION) ---
@st.fragment(run_every=5 if st.session_state.get("enable_main_refresh", False) else None)
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
        with col_status:
            st.write("")
            cb_main = st.checkbox("Enable Auto-Refresh (5s)", value=st.session_state["enable_main_refresh"], key="cb_main_refresh")
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

    # --- STRATEGY BASKET DISPLAY SECTION ---
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

        for leg in st.session_state["basket_legs"]:
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
                "Action": act,
                "Strike": k,
                "Type": t,
                "Qty": qty,
                "Entry (₹)": round(entry_p, 2),
                "LTP (₹)": round(ltp, 2),
                "P&L (₹)": round(leg_pnl, 2),
                "Delta (Δ)": round(pos_delta, 2),
                "Gamma (γ)": round(pos_gamma, 4),
                "Theta (θ)": round(pos_theta, 2),
                "Vega (ν)": round(pos_vega, 2)
            })

        with st.container(border=True):
            st.markdown("**Combined Basket Summary**")
            b_m1, b_m2, b_m3, b_m4, b_m5 = st.columns(5)
            b_m1.metric("Net P&L (₹)", f"₹{tot_pnl:,.2f}")
            b_m2.metric("Net Delta (Δ)", f"{tot_delta:.2f}")
            b_m3.metric("Net Gamma (γ)", f"{tot_gamma:.4f}")
            b_m4.metric("Net Theta (θ)", f"{tot_theta:.2f}")
            b_m5.metric("Net Vega (ν)", f"{tot_vega:.2f}")

        df_basket_display = pd.DataFrame(calculated_legs)
        st.markdown("**Individual Legs Breakdown**")
        st.dataframe(df_basket_display, use_container_width=True)
    else:
        st.info("No legs added to strategy basket yet. Use sidebar **2. Build Strategy Basket** to add positions.")

    # --- OPTION CHAIN DERIVED KEY LEVELS & TARGETS ---
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

        # --- NET GAMMA vs VOLUME & OI SEPARATE CHARTS ---
        st.markdown("---")
        df_chain = pd.DataFrame(data["chain_results"])

        if not df_chain.empty:
            df_chain["Total_Vol"] = df_chain["C_Vol"] + df_chain["P_Vol"]
            df_chain["Total_OI"] = df_chain["C_OI"] + df_chain["P_OI"]

            def calculate_synced_ranges(v1_pos, v1_neg, v2_pos, v2_neg):
                y1_max = max(v1_pos.max(), 1.0)
                y1_min = min(v1_neg.min(), -1.0)
                y2_max = max(v2_pos.max(), 1.0)
                y2_min = min(v2_neg.min(), -1.0)

                ratio1 = abs(y1_min) / y1_max
                ratio2 = abs(y2_min) / y2_max
                max_ratio = max(ratio1, ratio2)

                range1 = [-y1_max * max_ratio * 1.05, y1_max * 1.05]
                range2 = [-y2_max * max_ratio * 1.05, y2_max * 1.05]
                return range1, range2

            # CHART 1: NET GAMMA EXPOSURE (VOLUME-BASED) vs VOLUME
            st.subheader("📊 Volume-Based Net Gamma Exposure vs Trading Volume")

            gex_vol_colors = np.where(df_chain["Net_GEX_Vol"] >= 0, "#006400", "#8B0000")

            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])

            fig_vol.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=df_chain["Net_GEX_Vol"],
                    name="Net Gamma (Vol-Based)",
                    marker_color=gex_vol_colors,
                    opacity=0.85,
                    width=25,
                    hovertemplate="Strike: %{x}<br>Net GEX (Vol): ₹%{y:,.0f}<extra></extra>"
                ),
                secondary_y=True
            )

            fig_vol.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=df_chain["C_Vol"],
                    name="Call Volume",
                    marker_color="#81C784",
                    opacity=0.6,
                    hovertemplate="Strike: %{x}<br>Call Vol: %{y:,}<extra></extra>"
                ),
                secondary_y=False
            )

            fig_vol.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=-df_chain["P_Vol"],
                    name="Put Volume",
                    marker_color="#FF8A80",
                    opacity=0.6,
                    hovertemplate="Strike: %{x}<br>Put Vol: %{customdata:,}<extra></extra>",
                    customdata=df_chain["P_Vol"]
                ),
                secondary_y=False
            )

            fig_vol.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_vol.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            fig_vol.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip Point")

            v1_range, v2_range = calculate_synced_ranges(
                df_chain["C_Vol"], -df_chain["P_Vol"],
                df_chain["Net_GEX_Vol"], df_chain["Net_GEX_Vol"]
            )

            fig_vol.update_layout(
                title="Strike-wise Volume-Based Net Gamma & Volume Profile (Call Vol Above / Put Vol Below)",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=480,
                barmode="overlay",
                margin=dict(l=20, r=20, t=40, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_vol.update_xaxes(type="linear", tickformat="d", dtick=100)
            fig_vol.update_yaxes(
                title_text="Put Vol (Below) | Call Vol (Above)",
                range=v1_range,
                secondary_y=False,
                showgrid=True,
                gridcolor="#262930",
                zeroline=True,
                zerolinecolor="#FFFFFF",
                zerolinewidth=1.5
            )
            fig_vol.update_yaxes(
                title_text="Net GEX (Vol-Based ₹)",
                range=v2_range,
                secondary_y=True,
                showgrid=False,
                zeroline=True,
                zerolinecolor="#FFFFFF",
                zerolinewidth=1.5
            )

            st.plotly_chart(fig_vol, use_container_width=True)

            # CHART 2: NET GAMMA EXPOSURE (OI-BASED) vs OPEN INTEREST
            st.subheader("📈 OI-Based Net Gamma Exposure vs Open Interest (OI)")

            gex_oi_colors = np.where(df_chain["Net_GEX_OI"] >= 0, "#006400", "#8B0000")

            fig_oi = make_subplots(specs=[[{"secondary_y": True}]])

            fig_oi.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=df_chain["Net_GEX_OI"],
                    name="Net Gamma (OI-Based)",
                    marker_color=gex_oi_colors,
                    opacity=0.85,
                    width=25,
                    hovertemplate="Strike: %{x}<br>Net GEX (OI): ₹%{y:,.0f}<extra></extra>"
                ),
                secondary_y=True
            )

            fig_oi.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=df_chain["C_OI"],
                    name="Call OI",
                    marker_color="#2E7D32",
                    opacity=0.6,
                    hovertemplate="Strike: %{x}<br>Call OI: %{y:,}<extra></extra>"
                ),
                secondary_y=False
            )

            fig_oi.add_trace(
                plt_go.Bar(
                    x=df_chain["Strike"],
                    y=-df_chain["P_OI"],
                    name="Put OI",
                    marker_color="#C62828",
                    opacity=0.6,
                    hovertemplate="Strike: %{x}<br>Put OI: %{customdata:,}<extra></extra>",
                    customdata=df_chain["P_OI"]
                ),
                secondary_y=False
            )

            fig_oi.add_hline(y=0, line_width=1.5, line_color="#FFFFFF")
            fig_oi.add_vline(x=data["spot_price"], line_dash="dash", line_color="#FAFAFA", annotation_text="Spot")
            fig_oi.add_vline(x=lvls["Zero_Gamma_Flip"], line_dash="dot", line_color="#FF9800", annotation_text="Flip Point")

            oi1_range, oi2_range = calculate_synced_ranges(
                df_chain["C_OI"], -df_chain["P_OI"],
                df_chain["Net_GEX_OI"], df_chain["Net_GEX_OI"]
            )

            fig_oi.update_layout(
                title="Strike-wise OI-Based Net Gamma & Open Interest Profile (Call OI Above / Put OI Below)",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                height=480,
                barmode="overlay",
                margin=dict(l=20, r=20, t=40, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_oi.update_xaxes(type="linear", tickformat="d", dtick=100)
            fig_oi.update_yaxes(
                title_text="Put OI (Below) | Call OI (Above)",
                range=oi1_range,
                secondary_y=False,
                showgrid=True,
                gridcolor="#262930",
                zeroline=True,
                zerolinecolor="#FFFFFF",
                zerolinewidth=1.5
            )
            fig_oi.update_yaxes(
                title_text="Net GEX (OI-Based ₹)",
                range=oi2_range,
                secondary_y=True,
                showgrid=False,
                zeroline=True,
                zerolinecolor="#FFFFFF",
                zerolinewidth=1.5
            )

            st.plotly_chart(fig_oi, use_container_width=True)

# Run main dashboard fragment
live_dashboard_fragment()

# Run separate isolated Z-Score engine fragment
zscore_analysis_fragment()
