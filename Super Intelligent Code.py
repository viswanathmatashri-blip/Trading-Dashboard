import os
import time
import json
import requests
import numpy as np
import pandas as pd
import pyotp
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq
from SmartApi import SmartConnect

# ==============================================================================
# PURE SCIPY BLACK-SCHOLES & GREEKS ENGINE
# ==============================================================================
def bs_price(flag, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if flag == 'c' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'c':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_implied_volatility(price, S, K, T, r, flag):
    intrinsic = max(0.0, (S - K) if flag == 'c' else (K - S))
    if price <= intrinsic:
        return 0.10
    
    f = lambda sigma: bs_price(flag, S, K, T, r, sigma) - price
    
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-4)
    except Exception:
        return 0.15

def bs_greeks(flag, S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega_val = S * norm.pdf(d1) * np.sqrt(T)
    
    p1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if flag == 'c':
        p2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_val = p1 - p2
    else:
        p2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_val = p1 + p2
        
    return gamma_val, vega_val, theta_val


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

API_KEY = get_secret("API_KEY")
CLIENT_CODE = get_secret("CLIENT_CODE")
PIN = get_secret("PIN")
TOTP_SECRET = get_secret("TOTP_SECRET")

RISK_FREE_RATE = 0.068       
LOT_SIZE = 65                
MIN_OI_THRESHOLD = 2500      
TRADING_DAYS_PER_YEAR = 252.0

@st.cache_resource(ttl=3600)
def authenticate():
    if not API_KEY or not CLIENT_CODE:
        raise ValueError("🔑 Credentials missing! Configure Streamlit secrets or environment variables.")
    smartApi = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    session = smartApi.generateSession(CLIENT_CODE, PIN, totp)
    if not isinstance(session, dict) or not session.get('status'):
        msg = session.get('message', 'Authentication Failed') if isinstance(session, dict) else 'Invalid Auth Response'
        raise ConnectionError(f"SmartAPI Login Failed: {msg}")
    return smartApi


@st.cache_data(ttl=1800)
def get_nifty_option_chain():
    """Pure data fetching function without Streamlit UI elements to allow clean caching."""
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


def extract_oi_and_price(market_data_res):
    """Safely extract price and Open Interest from SmartAPI FULL market data payload structures."""
    if not isinstance(market_data_res, dict) or not market_data_res.get('status'):
        return 0.0, 0.0
    
    data = market_data_res.get('data', {})
    
    # Locate target item dictionary in data or fetched list
    if isinstance(data, dict):
        if 'fetched' in data and isinstance(data['fetched'], list) and len(data['fetched']) > 0:
            item = data['fetched'][0]
        else:
            item = data
    elif isinstance(data, list) and len(data) > 0:
        item = data[0]
    else:
        return 0.0, 0.0

    # Extract Price (ltp / lastPrice)
    price = float(item.get('ltp', item.get('lastPrice', 0.0)))
    
    # Extract Open Interest using all potential key variants returned by SmartAPI FULL mode
    raw_oi = (
        item.get('op') or 
        item.get('openInterest') or 
        item.get('opnInterest') or 
        item.get('openinterest') or 
        item.get('oi') or 
        0
    )
    
    try:
        oi = float(str(raw_oi).replace(',', '').strip())
    except (ValueError, TypeError):
        oi = 0.0

    return price, oi


# ==============================================================================
# 2. QUANT ENGINE USING FULL MARKET DATA API
# ==============================================================================
def run_quant_engine(smartApi, chain, spot_price, expiry_dt, log_placeholder):
    log_placeholder.info("⏳ Step 3/4: Processing individual strikes & calculating Greeks...")
    now = pd.Timestamp.now()
    expiry_end = expiry_dt + pd.Timedelta(hours=15, minutes=30)
    remaining_seconds = max((expiry_end - now).total_seconds(), 3600)
    T = remaining_seconds / (365.0 * 86400.0)
    
    records = []
    strikes = chain[(chain['strike'] >= spot_price * 0.92) & (chain['strike'] <= spot_price * 1.08)]['strike'].unique()
    strikes.sort()
    
    total_strikes = len(strikes)
    for idx, K in enumerate(strikes):
        log_placeholder.write(f"🔄 **Evaluating Strike {K:.0f}** ({idx+1}/{total_strikes})...")
        
        ce_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('CE'))]
        pe_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('PE'))]
        
        if ce_row.empty or pe_row.empty:
            log_placeholder.write(f"⚠️ Strike {K:.0f}: Skipped (Missing CE/PE contract rows in chain)")
            continue
            
        ce_symbol, ce_token = ce_row.iloc[0]['tradingsymbol'], str(ce_row.iloc[0]['token'])
        pe_symbol, pe_token = pe_row.iloc[0]['tradingsymbol'], str(pe_row.iloc[0]['token'])
        
        try:
            # PROPER API PARAMETER KEYS REQUIRED BY SMARTCONNECT
            ce_res = smartApi.getMarketData(mode="FULL", exchangeTokens={"NFO": [ce_token]})
            pe_res = smartApi.getMarketData(mode="FULL", exchangeTokens={"NFO": [pe_token]})

            ce_price, ce_oi = extract_oi_and_price(ce_res)
            pe_price, pe_oi = extract_oi_and_price(pe_res)

            log_placeholder.write(f"📊 Strike {K:.0f} Metrics -> CE Price: ₹{ce_price:.2f}, PE Price: ₹{pe_price:.2f} | CE OI: {ce_oi:.0f}, PE OI: {pe_oi:.0f}")

            # Inequality Filters
            reasons = []
            if ce_price <= 0:
                reasons.append(f"CE Price ({ce_price:.2f}) <= 0.00")
            if pe_price <= 0:
                reasons.append(f"PE Price ({pe_price:.2f}) <= 0.00")
            if ce_oi < MIN_OI_THRESHOLD:
                reasons.append(f"CE OI ({ce_oi:.0f}) < {MIN_OI_THRESHOLD}")
            if pe_oi < MIN_OI_THRESHOLD:
                reasons.append(f"PE OI ({pe_oi:.0f}) < {MIN_OI_THRESHOLD}")

            if reasons:
                log_placeholder.write(f"⚠️ Strike {K:.0f}: Skipped -> Reason: {', '.join(reasons)}")
                continue

            ce_iv = bs_implied_volatility(ce_price, spot_price, K, T, RISK_FREE_RATE, 'c')
            pe_iv = bs_implied_volatility(pe_price, spot_price, K, T, RISK_FREE_RATE, 'p')

            ce_g, ce_v, ce_t = bs_greeks('c', spot_price, K, T, RISK_FREE_RATE, ce_iv)
            pe_g, pe_v, pe_t = bs_greeks('p', spot_price, K, T, RISK_FREE_RATE, pe_iv)
            
            net_gex = ((pe_oi * pe_g) - (ce_oi * ce_g)) * (spot_price ** 2) * 0.01 / 1e6
            
            records.append({
                'strike': K, 'ce_price': ce_price, 'pe_price': pe_price,
                'ce_iv': ce_iv, 'pe_iv': pe_iv, 'ce_oi': ce_oi, 'pe_oi': pe_oi,
                'ce_gamma': ce_g, 'pe_gamma': pe_g,
                'ce_vega': ce_v, 'pe_vega': pe_v,
                'ce_theta': ce_t, 'pe_theta': pe_t,
                'gex': net_gex
            })
            
            log_placeholder.write(f"✅ Strike {K:.0f}: Passed all filters & quantified.")

        except Exception as e:
            log_placeholder.write(f"❌ Strike {K:.0f}: Error during calculation ({str(e)})")
            continue
            
    df_quant = pd.DataFrame(records)
    log_placeholder.success(f"✅ Quant Engine Complete: Retained {len(df_quant)} strike pairs matching criteria.")
    return df_quant, T


# ==============================================================================
# 3. OPTIMIZER WITH STRICT RISK CONTROLS & AUDIT
# ==============================================================================
def find_optimal_iron_condor(df, spot, T, log_placeholder):
    log_placeholder.info("⏳ Step 4/4: Evaluating Iron Condor combinations...")
    if df is None or df.empty:
        return None, pd.DataFrame()
        
    best_score = -np.inf
    optimal_condor = None
    strikes = np.sort(df['strike'].unique())
    
    avg_chain_iv = np.mean(df['ce_iv'].tolist() + df['pe_iv'].tolist()) if not df.empty else 0.12
    expected_1d_move = spot * (avg_chain_iv * np.sqrt(1 / 365.0))
    MIN_OTM_BUFFER = max(spot * 0.004, expected_1d_move * 0.5)
    
    audit_logs = []
    combo_count = 0

    for i in range(len(strikes)):
        long_put_k = strikes[i]
        
        for j in range(i + 1, len(strikes)):
            short_put_k = strikes[j]
            wing_width = short_put_k - long_put_k
            
            if short_put_k >= spot or (spot - short_put_k) < MIN_OTM_BUFFER:
                audit_logs.append({
                    'combo': f"{long_put_k}/{short_put_k}/X/X",
                    'status': 'Rejected',
                    'reason': f"Short Put ({short_put_k}) too close to spot (Buffer: {spot - short_put_k:.1f} < {MIN_OTM_BUFFER:.1f})"
                })
                continue
                
            for k in range(j + 1, len(strikes)):
                short_call_k = strikes[k]
                
                if short_call_k <= spot or (short_call_k - spot) < MIN_OTM_BUFFER:
                    audit_logs.append({
                        'combo': f"{long_put_k}/{short_put_k}/{short_call_k}/X",
                        'status': 'Rejected',
                        'reason': f"Short Call ({short_call_k}) too close to spot (Buffer: {short_call_k - spot:.1f} < {MIN_OTM_BUFFER:.1f})"
                    })
                    continue
                    
                long_call_k = short_call_k + wing_width
                if long_call_k not in strikes:
                    audit_logs.append({
                        'combo': f"{long_put_k}/{short_put_k}/{short_call_k}/{long_call_k}",
                        'status': 'Rejected',
                        'reason': f"Symmetric Long Call Strike ({long_call_k}) not in liquid strikes"
                    })
                    continue
                    
                combo_count += 1
                lp_row = df[df['strike'] == long_put_k].iloc[0]
                sp_row = df[df['strike'] == short_put_k].iloc[0]
                sc_row = df[df['strike'] == short_call_k].iloc[0]
                lc_row = df[df['strike'] == long_call_k].iloc[0]
                
                raw_credit = (sp_row['pe_price'] + sc_row['ce_price']) - (lp_row['pe_price'] + lc_row['ce_price'])
                net_credit = raw_credit * 0.96
                max_loss = wing_width - net_credit
                
                if net_credit <= 1.0 or max_loss <= 0:
                    audit_logs.append({
                        'combo': f"{long_put_k}/{short_put_k}/{short_call_k}/{long_call_k}",
                        'status': 'Rejected',
                        'reason': f"Credit ({net_credit:.2f}) <= 1.0 or Max Loss ({max_loss:.2f}) <= 0"
                    })
                    continue
                    
                net_gamma = (sp_row['pe_gamma'] + sc_row['ce_gamma']) - (lp_row['pe_gamma'] + lc_row['ce_gamma'])
                net_vega = (sp_row['pe_vega'] + sc_row['ce_vega']) - (lp_row['pe_vega'] + lc_row['ce_vega'])
                net_theta = (sp_row['pe_theta'] + sc_row['ce_theta']) - (lp_row['pe_theta'] + lc_row['ce_theta'])
                
                sigma_avg = np.mean([sp_row['pe_iv'], sc_row['ce_iv']])
                d1_upper = (np.log(spot / short_call_k) + (RISK_FREE_RATE + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
                d1_lower = (np.log(spot / short_put_k) + (RISK_FREE_RATE + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
                pop = norm.cdf(d1_upper) - norm.cdf(d1_lower)
                
                if pop < 0.52:
                    audit_logs.append({
                        'combo': f"{long_put_k}/{short_put_k}/{short_call_k}/{long_call_k}",
                        'status': 'Rejected',
                        'reason': f"PoP ({pop*100:.1f}%) < 52.0%"
                    })
                    continue
                    
                base_margin = max_loss * LOT_SIZE
                buffered_margin = base_margin * 1.30  
                expected_value = (net_credit * pop) - (max_loss * (1 - pop))
                
                if expected_value <= 0:
                    audit_logs.append({
                        'combo': f"{long_put_k}/{short_put_k}/{short_call_k}/{long_call_k}",
                        'status': 'Rejected',
                        'reason': f"Expected Value (EV: ₹{expected_value * LOT_SIZE:.2f}) <= 0"
                    })
                    continue
                    
                utility_score = (expected_value * LOT_SIZE / buffered_margin) / (1.0 + (net_gamma * 20))
                
                cand_details = {
                    'long_put': long_put_k,
                    'short_put': short_put_k,
                    'short_call': short_call_k,
                    'long_call': long_call_k,
                    'wing_width': wing_width,
                    'raw_credit_pts': raw_credit,
                    'net_credit_pts': net_credit,
                    'net_credit_rupees': net_credit * LOT_SIZE,
                    'max_loss_rupees': max_loss * LOT_SIZE,
                    'stop_loss_level': net_credit * 2.0 * LOT_SIZE,
                    'base_margin': base_margin,
                    'buffered_margin': buffered_margin,
                    'pop': pop * 100,
                    'expected_value': expected_value * LOT_SIZE,
                    'net_gamma': net_gamma,
                    'net_vega_shock_5pct': net_vega * 0.05 * LOT_SIZE,
                    'net_theta_daily': (net_theta / TRADING_DAYS_PER_YEAR) * LOT_SIZE,
                    'return_on_margin': (net_credit * LOT_SIZE / buffered_margin) * 100,
                    'utility_score': utility_score
                }

                if utility_score > best_score:
                    if optimal_condor is not None:
                        audit_logs.append({
                            'combo': f"{optimal_condor['long_put']:.0f}/{optimal_condor['short_put']:.0f}/{optimal_condor['short_call']:.0f}/{optimal_condor['long_call']:.0f}",
                            'status': 'Outranked',
                            'reason': f"Utility ({utility_score:.4f}) outranked former score ({best_score:.4f})"
                        })
                    best_score = utility_score
                    optimal_condor = cand_details
                    log_placeholder.write(f"🎯 New Best Condor Found: {long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f} (Utility: {utility_score:.4f})")
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
                        'status': 'Selected (Current Best)',
                        'reason': f"Passed all criteria; Utility ({utility_score:.4f})"
                    })
                else:
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
                        'status': 'Rejected',
                        'reason': f"Utility ({utility_score:.4f}) < Best Utility ({best_score:.4f})"
                    })

    df_audit = pd.DataFrame(audit_logs)
    log_placeholder.success(f"✅ Evaluated {combo_count} valid Iron Condor structures.")
    return optimal_condor, df_audit


# ==============================================================================
# 4. STREAMLIT DASHBOARD UI
# ==============================================================================
st.title("⚡ Dynamic Risk-Managed Iron Condor Engine")

live_mode = st.sidebar.checkbox("Enable Live Refresh (5s)", value=False)

with st.sidebar:
    st.markdown("---")
    st.subheader("🖥️ Execution Log Monitor")
    exec_status = st.status("Initializing Engine...", expanded=True)

try:
    exec_status.write("🔑 Authenticating with SmartAPI...")
    smartApi = authenticate()
    st.success(f"🟢 **API Connection Status:** Successfully connected to SmartAPI (Client ID: `{CLIENT_CODE}`)")
    exec_status.write("✅ Authentication Successful.")

    exec_status.info("⏳ Step 1/4: Fetching master NIFTY option chain file...")
    chain, expiry_dt = get_nifty_option_chain()
    exec_status.success(f"✅ Option chain received for expiry date: **{expiry_dt.strftime('%d-%b-%Y')}**")
    
    exec_status.info("⏳ Step 2/4: Fetching live NIFTY index spot price...")
    spot_res = smartApi.ltpData("NSE", "NIFTY", "99926000")
    if not isinstance(spot_res, dict) or not spot_res.get('status') or not isinstance(spot_res.get('data'), dict):
        exec_status.warning("⚠️ Spot Price Fetch Warning: Market closed or feed delayed. Using fallback benchmark.")
        spot_price = 24383.60
    else:
        spot_price = float(spot_res['data']['ltp'])
        exec_status.success(f"✅ NIFTY Spot Price Received: **₹{spot_price:,.2f}**")
    
    df_quant, T = run_quant_engine(smartApi, chain, spot_price, expiry_dt, exec_status)
    result, df_audit = find_optimal_iron_condor(df_quant, spot_price, T, exec_status)

    exec_status.update(label="🚀 Execution Complete!", state="complete", expanded=False)

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

    st.markdown("---")
    st.subheader("🔍 Iron Condor Selection Audit Trail")
    if not df_audit.empty:
        status_filter = st.multiselect(
            "Filter Evaluation Status", 
            options=df_audit['status'].unique(), 
            default=df_audit['status'].unique()
        )
        filtered_audit = df_audit[df_audit['status'].isin(status_filter)]
        st.dataframe(filtered_audit, use_container_width=True)
    else:
        st.info("No strike combinations evaluated.")

except Exception as e:
    exec_status.update(label="❌ Execution Failed", state="error", expanded=True)
    exec_status.error(f"Error: {str(e)}")
    st.error(f"🔴 **API Connection Status:** Connection Failed! Details: {str(e)}")

if live_mode:
    time.sleep(5.0)
    st.rerun()
