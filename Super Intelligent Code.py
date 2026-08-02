import os
import time
import json
import requests
import numpy as np
import pandas as pd
import pyotp
import streamlit as st
from scipy.stats import norm
from vollib.black_scholes.greeks.analytical import gamma, vega, theta
from vollib.black_scholes.implied_volatility import implied_volatility
from SmartApi import SmartConnect

# ==============================================================================
# STREAMLIT CONFIGURATION & SECURE AUTHENTICATION
# ==============================================================================
st.set_page_config(page_title="Institutional Quant Iron Condor", page_icon="⚡", layout="wide")

def get_secret(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val:
        return env_val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# Safely fetch credentials
API_KEY = get_secret("API_KEY")
CLIENT_CODE = get_secret("CLIENT_CODE")
PIN = get_secret("PIN")
TOTP_SECRET = get_secret("TOTP_SECRET")

# QUANT PARAMETERS
RISK_FREE_RATE = 0.068       # Benchmark Repo rate (~6.8%)
LOT_SIZE = 65                # NIFTY Lot Size
MIN_OI_THRESHOLD = 2500      # Optimized to allow protective outer wings
TRADING_DAYS_PER_YEAR = 252.0

@st.cache_resource(ttl=3600)
def authenticate():
    if not API_KEY or not CLIENT_CODE:
        st.error("🔑 Credentials missing! Configure Streamlit secrets or environment variables.")
        st.stop()
    smartApi = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    session = smartApi.generateSession(CLIENT_CODE, PIN, totp)
    if not isinstance(session, dict) or not session.get('status'):
        msg = session.get('message', 'Authentication Failed') if isinstance(session, dict) else 'Invalid Auth Response'
        raise ConnectionError(f"SmartAPI Login Failed: {msg}")
    return smartApi


@st.cache_data(ttl=1800)
def get_nifty_option_chain():
    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        url = "https://margincalculator.angelbroking.com/OpenAPI_Data/files/OpenAPIScripMaster.json"
        response = requests.get(url, headers=headers, timeout=15)
        
    df = pd.DataFrame(response.json())
    df.columns = [str(c).lower() for c in df.columns]
    
    symbol_col = 'name' if 'name' in df.columns else 'symbol'
    
    nifty_opts = df[
        (df[symbol_col].astype(str).str.upper() == 'NIFTY') & 
        (df['instrumenttype'].astype(str).str.upper() == 'OPTIDX') & 
        (df['exch_seg'].astype(str).str.upper() == 'NFO')
    ].copy()
    
    if 'tradingsymbol' not in nifty_opts.columns:
        nifty_opts['tradingsymbol'] = nifty_opts['symbol'] if 'symbol' in nifty_opts.columns else nifty_opts[symbol_col]
        
    nifty_opts['strike'] = nifty_opts['strike'].astype(float) / 100.0
    nifty_opts['expiry_dt'] = pd.to_datetime(nifty_opts['expiry'], format='%d%b%Y')
    
    today = pd.Timestamp.now().normalize()
    upcoming_expiries = nifty_opts[nifty_opts['expiry_dt'] >= today]['expiry_dt']
    nearest_expiry = upcoming_expiries.min()
    
    chain = nifty_opts[nifty_opts['expiry_dt'] == nearest_expiry].copy()
    return chain, nearest_expiry


# ==============================================================================
# 2. QUANT ENGINE WITH REAL-TIME GREEKS & LIQUIDITY FILTERS
# ==============================================================================
def run_quant_engine(smartApi, chain, spot_price, expiry_dt):
    now = pd.Timestamp.now()
    expiry_end = expiry_dt + pd.Timedelta(hours=15, minutes=30)
    remaining_seconds = max((expiry_end - now).total_seconds(), 3600)
    T = remaining_seconds / (365.0 * 86400.0)
    
    records = []
    strikes = chain[(chain['strike'] >= spot_price * 0.92) & (chain['strike'] <= spot_price * 1.08)]['strike'].unique()
    strikes.sort()
    
    for K in strikes:
        ce_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('CE'))]
        pe_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('PE'))]
        
        if ce_row.empty or pe_row.empty:
            continue
            
        ce_symbol, ce_token = ce_row.iloc[0]['tradingsymbol'], ce_row.iloc[0]['token']
        pe_symbol, pe_token = pe_row.iloc[0]['tradingsymbol'], pe_row.iloc[0]['token']
        
        try:
            ce_res = smartApi.ltpData("NFO", ce_symbol, ce_token)
            pe_res = smartApi.ltpData("NFO", pe_symbol, pe_token)
            
            if not isinstance(ce_res, dict) or not ce_res.get('status') or not isinstance(ce_res.get('data'), dict):
                continue
            if not isinstance(pe_res, dict) or not pe_res.get('status') or not isinstance(pe_res.get('data'), dict):
                continue

            ce_price = float(ce_res['data'].get('ltp', 0))
            pe_price = float(pe_res['data'].get('ltp', 0))
            ce_oi = float(ce_res['data'].get('openinterest', 0))
            pe_oi = float(pe_res['data'].get('openinterest', 0))
            
            # Liquidity Filter
            if ce_price <= 0 or pe_price <= 0 or ce_oi < MIN_OI_THRESHOLD or pe_oi < MIN_OI_THRESHOLD:
                continue

            ce_iv = implied_volatility(ce_price, spot_price, K, T, RISK_FREE_RATE, 'c')
            pe_iv = implied_volatility(pe_price, spot_price, K, T, RISK_FREE_RATE, 'p')
            
            ce_g = gamma('c', spot_price, K, T, RISK_FREE_RATE, ce_iv)
            pe_g = gamma('p', spot_price, K, T, RISK_FREE_RATE, pe_iv)
            ce_v = vega('c', spot_price, K, T, RISK_FREE_RATE, ce_iv)
            pe_v = vega('p', spot_price, K, T, RISK_FREE_RATE, pe_iv)
            ce_t = theta('c', spot_price, K, T, RISK_FREE_RATE, ce_iv)
            pe_t = theta('p', spot_price, K, T, RISK_FREE_RATE, pe_iv)
            
            net_gex = ((pe_oi * pe_g) - (ce_oi * ce_g)) * (spot_price ** 2) * 0.01 / 1e6
            
            records.append({
                'strike': K, 'ce_price': ce_price, 'pe_price': pe_price,
                'ce_iv': ce_iv, 'pe_iv': pe_iv, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
                'ce_gamma': ce_g, 'pe_gamma': pe_g,
                'ce_vega': ce_v, 'pe_vega': pe_v,
                'ce_theta': ce_t, 'pe_theta': pe_t,
                'gex': net_gex
            })
        except Exception:
            continue
            
    df_quant = pd.DataFrame(records)
    return df_quant, T


# ==============================================================================
# 3. OPTIMIZER WITH STRICT RISK CONTROLS & ADAPTIVE BUFFERS
# ==============================================================================
def find_optimal_iron_condor(df, spot, T):
    if df is None or df.empty:
        return None
        
    best_score = -np.inf
    optimal_condor = None
    strikes = np.sort(df['strike'].unique())
    
    # Adaptive OTM Distance (Minimum 0.4% distance or IV-based expected move)
    avg_chain_iv = np.mean(df['ce_iv'].tolist() + df['pe_iv'].tolist()) if not df.empty else 0.12
    expected_1d_move = spot * (avg_chain_iv * np.sqrt(1 / 365.0))
    MIN_OTM_BUFFER = max(spot * 0.004, expected_1d_move * 0.5)
    
    for i in range(len(strikes)):
        long_put_k = strikes[i]
        
        for j in range(i + 1, len(strikes)):
            short_put_k = strikes[j]
            wing_width = short_put_k - long_put_k
            
            if short_put_k >= spot or (spot - short_put_k) < MIN_OTM_BUFFER:
                continue
                
            for k in range(j + 1, len(strikes)):
                short_call_k = strikes[k]
                
                if short_call_k <= spot or (short_call_k - spot) < MIN_OTM_BUFFER:
                    continue
                    
                long_call_k = short_call_k + wing_width  # Symmetric Wings
                if long_call_k not in strikes:
                    continue
                    
                lp_row = df[df['strike'] == long_put_k].iloc[0]
                sp_row = df[df['strike'] == short_put_k].iloc[0]
                sc_row = df[df['strike'] == short_call_k].iloc[0]
                lc_row = df[df['strike'] == long_call_k].iloc[0]
                
                raw_credit = (sp_row['pe_price'] + sc_row['ce_price']) - (lp_row['pe_price'] + lc_row['ce_price'])
                
                # Proportional Slippage Deduction (4% of premium earned)
                net_credit = raw_credit * 0.96
                max_loss = wing_width - net_credit
                
                # Zero Compromise on Loss Safety Rules
                if net_credit <= 1.0 or max_loss <= 0:
                    continue
                    
                net_gamma = (sp_row['pe_gamma'] + sc_row['ce_gamma']) - (lp_row['pe_gamma'] + lc_row['ce_gamma'])
                net_vega = (sp_row['pe_vega'] + sc_row['ce_vega']) - (lp_row['pe_vega'] + lc_row['ce_vega'])
                net_theta = (sp_row['pe_theta'] + sc_row['ce_theta']) - (lp_row['pe_theta'] + lc_row['ce_theta'])
                
                # Black-Scholes Delta / Normal Distribution PoP
                sigma_avg = np.mean([sp_row['pe_iv'], sc_row['ce_iv']])
                d1_upper = (np.log(spot / short_call_k) + (RISK_FREE_RATE + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
                d1_lower = (np.log(spot / short_put_k) + (RISK_FREE_RATE + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
                pop = norm.cdf(d1_upper) - norm.cdf(d1_lower)
                
                if pop < 0.52:  # Strict probability floor
                    continue
                    
                base_margin = max_loss * LOT_SIZE
                buffered_margin = base_margin * 1.30  # Safety Buffer: +30% Margin Cushion
                expected_value = (net_credit * pop) - (max_loss * (1 - pop))
                
                if expected_value <= 0:  # Hard safety rule: Positive EV strictly required
                    continue
                    
                utility_score = (expected_value * LOT_SIZE / buffered_margin) / (1.0 + (net_gamma * 20))
                
                if utility_score > best_score:
                    best_score = utility_score
                    optimal_condor = {
                        'long_put': long_put_k,
                        'short_put': short_put_k,
                        'short_call': short_call_k,
                        'long_call': long_call_k,
                        'wing_width': wing_width,
                        'raw_credit_pts': raw_credit,
                        'net_credit_pts': net_credit,
                        'net_credit_rupees': net_credit * LOT_SIZE,
                        'max_loss_rupees': max_loss * LOT_SIZE,
                        'stop_loss_level': net_credit * 2.0 * LOT_SIZE,  # Hard Stop-Loss @ 2x credit earned
                        'base_margin': base_margin,
                        'buffered_margin': buffered_margin,
                        'pop': pop * 100,
                        'expected_value': expected_value * LOT_SIZE,
                        'net_gamma': net_gamma,
                        'net_vega_shock_5pct': net_vega * 0.05 * LOT_SIZE,
                        'net_theta_daily': (net_theta / TRADING_DAYS_PER_YEAR) * LOT_SIZE,
                        'return_on_margin': (net_credit * LOT_SIZE / buffered_margin) * 100
                    }
                
    return optimal_condor


# ==============================================================================
# 4. STREAMLIT DASHBOARD UI
# ==============================================================================
st.title("⚡ Dynamic Risk-Managed Iron Condor Engine")

live_mode = st.sidebar.checkbox("Enable Live Refresh (5s)", value=False)

try:
    smartApi = authenticate()
    chain, expiry_dt = get_nifty_option_chain()
    
    spot_res = smartApi.ltpData("NSE", "NIFTY", "99926000")
    if not isinstance(spot_res, dict) or not spot_res.get('status') or not isinstance(spot_res.get('data'), dict):
        st.warning("⚠️ Spot Price Fetch Warning: Market closed or feed delayed. Using fallback benchmark.")
        spot_price = 24383.60
    else:
        spot_price = float(spot_res['data']['ltp'])
    
    df_quant, T = run_quant_engine(smartApi, chain, spot_price, expiry_dt)
    result = find_optimal_iron_condor(df_quant, spot_price, T)

    st.subheader("📌 Live Market Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NIFTY Spot Price", f"₹{spot_price:,.2f}")
    m2.metric("Target Expiry", expiry_dt.strftime('%d-%b-%Y'))
    
    if result:
        m3.metric("Model PoP", f"{result['pop']:.1f}%")
        m4.metric("Buffered RoM", f"{result['return_on_margin']:.2f}%")

        st.markdown("---")
        st.subheader("🎯 Liquidity-Filtered Option Legs")
        
        trade_df = pd.DataFrame([
            {"Leg": "[1] BUY PUT (Outer Wing)", "Strike": f"{result['long_put']:.0f} PE", "Distance": f"-{spot_price - result['long_put']:.0f} pts"},
            {"Leg": "[2] SELL PUT (Inner Strike)", "Strike": f"{result['short_put']:.0f} PE", "Distance": f"-{spot_price - result['short_put']:.0f} pts"},
            {"Leg": "[3] SELL CALL (Inner Strike)", "Strike": f"{result['short_call']:.0f} CE", "Distance": f"+{result['short_call'] - spot_price:.0f} pts"},
            {"Leg": "[4] BUY CALL (Outer Wing)", "Strike": f"{result['long_call']:.0f} CE", "Distance": f"+{result['long_call'] - spot_price:.0f} pts"},
        ])
        st.table(trade_df)

        st.subheader("📊 Financials & Risk Buffers")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Net Credit (Post-Slippage)", f"₹{result['net_credit_rupees']:,.2f}", f"{result['net_credit_pts']:.2f} pts")
        f2.metric("Max Loss Limit", f"₹{result['max_loss_rupees']:,.2f}")
        f3.metric("Required Margin (+30% Cushion)", f"₹{result['buffered_margin']:,.2f}")
        f4.metric("Net EV per Trade", f"₹{result['expected_value']:,.2f}")

        st.markdown("---")
        st.subheader("🛡️ Dynamic Risk & Sensitivity Metrics")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Hard Stop-Loss Threshold", f"₹{result['stop_loss_level']:,.2f}", "2x Credit Limit", delta_color="inverse")
        r2.metric("Daily Theta Decay (+)", f"₹{abs(result['net_theta_daily']):,.2f}/day")
        r3.metric("Vega Risk (+5% IV Shock)", f"₹{result['net_vega_shock_5pct']:,.2f}", delta_color="inverse")
        r4.metric("Net Position Gamma", f"{result['net_gamma']:.4f}")

    else:
        st.warning("No valid Iron Condor setup met the minimum PoP, EV, and safety criteria for this tick.")

except Exception as e:
    st.error(f"Execution Error: {str(e)}")

if live_mode:
    time.sleep(5.0)
    st.rerun()
