import time
import json
import requests
import numpy as np
import pandas as pd
import pyotp
import streamlit as st
from scipy.interpolate import UnivariateSpline
from vollib.black_scholes.greeks.analytical import gamma
from vollib.black_scholes.implied_volatility import implied_volatility
from SmartApi import SmartConnect

# ==============================================================================
# STREAMLIT CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Quant Iron Condor Live", page_icon="⚡", layout="wide")

API_KEY = "o2b7s4Oo"
CLIENT_CODE = "AACK311190"
PIN = "8547"
TOTP_SECRET = "YCRQCDQ7NPUHKYH7RS73NXQ5VE"

RISK_FREE_RATE = 0.068  # Benchmark Repo rate (~6.8%)
LOT_SIZE = 65           # NIFTY Lot Size


# ==============================================================================
# 1. AUTHENTICATION & INSTRUMENT PARSER
# ==============================================================================
@st.cache_resource(ttl=3600)
def authenticate():
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
# 2. QUANTITATIVE SURFACE & GEX ENGINE
# ==============================================================================
def run_quant_engine(smartApi, chain, spot_price, expiry_dt):
    T = max((expiry_dt - pd.Timestamp.now()).days / 365.0, 0.001)
    records = []
    
    strikes = chain[(chain['strike'] >= spot_price * 0.94) & (chain['strike'] <= spot_price * 1.06)]['strike'].unique()
    strikes.sort()
    
    for K in strikes:
        ce_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('CE'))]
        pe_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('PE'))]
        
        if ce_row.empty or pe_row.empty:
            continue
            
        ce_symbol = ce_row.iloc[0]['tradingsymbol']
        ce_token = ce_row.iloc[0]['token']
        pe_symbol = pe_row.iloc[0]['tradingsymbol']
        pe_token = pe_row.iloc[0]['token']
        
        try:
            ce_res = smartApi.ltpData("NFO", ce_symbol, ce_token)
            pe_res = smartApi.ltpData("NFO", pe_symbol, pe_token)
            
            # SAFE DICTIONARY CHECKS
            if not isinstance(ce_res, dict) or not ce_res.get('status') or not isinstance(ce_res.get('data'), dict):
                continue
            if not isinstance(pe_res, dict) or not pe_res.get('status') or not isinstance(pe_res.get('data'), dict):
                continue

            ce_price = float(ce_res['data'].get('ltp', 0))
            pe_price = float(pe_res['data'].get('ltp', 0))
            ce_oi = float(ce_res['data'].get('openinterest', 100000))
            pe_oi = float(pe_res['data'].get('openinterest', 100000))
            
            if ce_price <= 0 or pe_price <= 0:
                continue

            ce_iv = implied_volatility(ce_price, spot_price, K, T, RISK_FREE_RATE, 'c')
            pe_iv = implied_volatility(pe_price, spot_price, K, T, RISK_FREE_RATE, 'p')
            ce_gamma = gamma('c', spot_price, K, T, RISK_FREE_RATE, ce_iv)
            net_gex = (pe_oi - ce_oi) * ce_gamma * (spot_price ** 2) * 0.01 / 1e6
            
            records.append({
                'strike': K, 'ce_price': ce_price, 'pe_price': pe_price,
                'ce_iv': ce_iv, 'pe_iv': pe_iv, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
                'gamma': ce_gamma, 'gex': net_gex
            })
            time.sleep(0.01)
        except Exception:
            continue
            
    df_quant = pd.DataFrame(records)
    if df_quant.empty:
        return None, T
        
    spline = UnivariateSpline(df_quant['strike'], df_quant['ce_price'], k=4, s=0.5)
    df_quant['iPDF'] = np.exp(RISK_FREE_RATE * T) * spline.derivative(n=2)(df_quant['strike'])
    df_quant['iPDF'] = np.maximum(df_quant['iPDF'], 0)
    
    return df_quant, T


# ==============================================================================
# 3. MATHEMATICAL OPTIMIZER ENGINE
# ==============================================================================
def find_optimal_iron_condor(df, spot, T):
    if df is None or df.empty:
        return None
        
    best_score = -np.inf
    optimal_condor = None
    strikes = df['strike'].values
    
    MIN_OTM_BUFFER = spot * 0.008
    
    for i in range(len(strikes) - 3):
        long_put_k = strikes[i]
        short_put_k = strikes[i + 1]
        
        if short_put_k >= spot or (spot - short_put_k) < MIN_OTM_BUFFER:
            continue
            
        for j in range(i + 2, len(strikes) - 1):
            short_call_k = strikes[j]
            long_call_k = strikes[j + 1]
            
            if short_call_k <= spot or (short_call_k - spot) < MIN_OTM_BUFFER:
                continue
            
            put_wing = short_put_k - long_put_k
            call_wing = long_call_k - short_call_k
            if put_wing != call_wing:
                continue
                
            wing_width = put_wing
            
            lp_price = df[df['strike'] == long_put_k]['pe_price'].values[0]
            sp_price = df[df['strike'] == short_put_k]['pe_price'].values[0]
            sc_price = df[df['strike'] == short_call_k]['ce_price'].values[0]
            lc_price = df[df['strike'] == long_call_k]['ce_price'].values[0]
            
            net_credit = (sp_price + sc_price) - (lp_price + lc_price)
            max_loss = wing_width - net_credit
            
            if net_credit <= 0 or max_loss <= 0:
                continue
                
            put_distance = spot - short_put_k
            call_distance = short_call_k - spot
            skew_ratio = min(put_distance, call_distance) / max(put_distance, call_distance)
            
            if skew_ratio < 0.50:
                continue
                
            margin_required = (wing_width - net_credit) * LOT_SIZE
            ipdf_subset = df[(df['strike'] >= short_put_k) & (df['strike'] <= short_call_k)]['iPDF']
            pop = np.sum(ipdf_subset) / np.sum(df['iPDF']) if np.sum(df['iPDF']) > 0 else 0.70
            
            short_put_gex = df[df['strike'] == short_put_k]['gex'].values[0]
            short_call_gex = df[df['strike'] == short_call_k]['gex'].values[0]
            gex_stability_score = short_put_gex + short_call_gex
            
            expected_value = (net_credit * pop) - (max_loss * (1 - pop))
            utility_score = (expected_value / margin_required) * (1 + gex_stability_score) * skew_ratio
            
            if utility_score > best_score:
                best_score = utility_score
                optimal_condor = {
                    'long_put': long_put_k,
                    'short_put': short_put_k,
                    'short_call': short_call_k,
                    'long_call': long_call_k,
                    'wing_width': wing_width,
                    'net_credit_pts': net_credit,
                    'net_credit_rupees': net_credit * LOT_SIZE,
                    'max_loss_rupees': max_loss * LOT_SIZE,
                    'margin_required': margin_required,
                    'pop': pop * 100,
                    'expected_value': expected_value * LOT_SIZE,
                    'return_on_margin': (net_credit * LOT_SIZE / margin_required) * 100
                }
                
    return optimal_condor


# ==============================================================================
# 4. STREAMLIT DASHBOARD
# ==============================================================================
st.title("⚡ Quantitative Iron Condor Engine (Live Stream)")

live_mode = st.sidebar.checkbox("Enable Live Refresh", value=True)

try:
    smartApi = authenticate()
    chain, expiry_dt = get_nifty_option_chain()
    
    # Safely fetch NIFTY Spot Price
    spot_res = smartApi.ltpData("NSE", "NIFTY", "99926000")
    
    if not isinstance(spot_res, dict) or not spot_res.get('status') or not isinstance(spot_res.get('data'), dict):
        err_msg = spot_res.get('message', 'Market Data Unavailable') if isinstance(spot_res, dict) else 'Invalid API Response'
        st.warning(f"⚠️ Spot Price Fetch Warning: {err_msg}. (Markets may be closed outside trading hours).")
        spot_price = 24383.60  # Default benchmark fallback outside market hours
    else:
        spot_price = float(spot_res['data']['ltp'])
    
    df_quant, T = run_quant_engine(smartApi, chain, spot_price, expiry_dt)
    result = find_optimal_iron_condor(df_quant, spot_price, T)

    st.subheader("📌 Live Market Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NIFTY Spot Price", f"₹{spot_price:,.2f}")
    m2.metric("Expiry Date", expiry_dt.strftime('%d-%b-%Y'))
    
    if result:
        m3.metric("Probability of Profit (PoP)", f"{result['pop']:.1f}%")
        m4.metric("Return on Margin", f"{result['return_on_margin']:.2f}%")

        st.markdown("---")
        st.subheader("🎯 Mathematically Optimal Option Legs")
        
        trade_df = pd.DataFrame([
            {"Leg": "[1] BUY PUT (Outer)", "Strike": f"{result['long_put']:.0f} PE", "Distance": f"-{spot_price - result['long_put']:.0f} pts"},
            {"Leg": "[2] SELL PUT (Inner)", "Strike": f"{result['short_put']:.0f} PE", "Distance": f"-{spot_price - result['short_put']:.0f} pts"},
            {"Leg": "[3] SELL CALL (Inner)", "Strike": f"{result['short_call']:.0f} CE", "Distance": f"+{result['short_call'] - spot_price:.0f} pts"},
            {"Leg": "[4] BUY CALL (Outer)", "Strike": f"{result['long_call']:.0f} CE", "Distance": f"+{result['long_call'] - spot_price:.0f} pts"},
        ])
        st.table(trade_df)

        st.subheader("📊 Trade Financials")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Net Credit", f"₹{result['net_credit_rupees']:,.2f}", f"{result['net_credit_pts']:.2f} pts")
        f2.metric("Max Loss", f"₹{result['max_loss_rupees']:,.2f}")
        f3.metric("Required Margin", f"₹{result['margin_required']:,.2f}")
        f4.metric("Expected Value", f"₹{result['expected_value']:,.2f}")
    else:
        st.warning("No valid Iron Condor setup met positive EV criteria at this tick.")

except Exception as e:
    st.error(f"Execution Error: {str(e)}")

if live_mode:
    time.sleep(2.0)
    st.rerun()
