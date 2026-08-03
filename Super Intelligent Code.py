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
from vollib.black_scholes.exceptions import BelowIntrinsicException
from SmartApi import SmartConnect

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & SECURE AUTHENTICATION
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
MIN_OI_THRESHOLD = 2500      # Open Interest Threshold
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
def get_nifty_option_chain(_log_placeholder):
    _log_placeholder.info("⏳ Step 1/4: Fetching master NIFTY option chain file...")
    url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        _log_placeholder.warning("⚠️ Primary master URL failed, retrying fallback URL...")
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
    _log_placeholder.success(f"✅ Option chain received for expiry date: **{nearest_expiry.strftime('%d-%b-%Y')}**")
    return chain, nearest_expiry


# Safe IV Calculation Helper to avoid crashing on ITM pricing anomalies
def calculate_safe_iv(price, spot, K, T, rate, flag):
    try:
        # Check intrinsic lower bound manually
        intrinsic = max(0.0, (spot - K) if flag == 'c' else (K - spot))
        if price <= intrinsic:
            return None, f"Price (₹{price:.2f}) <= Intrinsic (₹{intrinsic:.2f})"
        
        iv = implied_volatility(price, spot, K, T, rate, flag)
        return iv, None
    except BelowIntrinsicException:
        return None, f"Price (₹{price:.2f}) below intrinsic value"
    except Exception as e:
        return None, str(e)


# ==============================================================================
# 2. QUANT ENGINE (SAFE IV CALCULATION & RESILIENT BATCH FETCHING)
# ==============================================================================
def run_quant_engine(smartApi, chain, spot_price, expiry_dt, log_placeholder):
    log_placeholder.info("⏳ Step 3/4: Batch-fetching market data & calculating Greeks...")
    now = pd.Timestamp.now()
    expiry_end = expiry_dt + pd.Timedelta(hours=15, minutes=30)
    remaining_seconds = max((expiry_end - now).total_seconds(), 3600)
    T = remaining_seconds / (365.0 * 86400.0)
    
    strikes = chain[(chain['strike'] >= spot_price * 0.92) & (chain['strike'] <= spot_price * 1.08)]['strike'].unique()
    strikes.sort()
    
    token_map = {}
    tokens_to_fetch = []
    
    for K in strikes:
        ce_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('CE'))]
        pe_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('PE'))]
        
        if not ce_row.empty and not pe_row.empty:
            ce_tok = str(ce_row.iloc[0]['token'])
            pe_tok = str(pe_row.iloc[0]['token'])
            token_map[K] = {'ce_token': ce_tok, 'pe_token': pe_tok}
            tokens_to_fetch.extend([ce_tok, pe_tok])

    market_data_lookup = {}
    chunk_size = 50
    for i in range(0, len(tokens_to_fetch), chunk_size):
        chunk = tokens_to_fetch[i:i + chunk_size]
        try:
            res = smartApi.getMarketData("FULL", {"NFO": chunk})
            if isinstance(res, dict) and res.get('status') and res.get('data', {}).get('fetched'):
                for item in res['data']['fetched']:
                    tok = str(item.get('symbolToken', ''))
                    if tok:
                        market_data_lookup[tok] = item
        except Exception as batch_err:
            log_placeholder.write(f"⚠️ Batch fetch warning (Chunk {i//chunk_size + 1}): {str(batch_err)}")

    records = []
    total_strikes = len(strikes)
    
    for idx, K in enumerate(strikes):
        if K not in token_map:
            log_placeholder.write(f"⚠️ Strike {K:.0f}: Skipped (Missing CE/PE contract rows)")
            continue
            
        ce_tok = token_map[K]['ce_token']
        pe_tok = token_map[K]['pe_token']
        
        ce_data = market_data_lookup.get(ce_tok, {})
        pe_data = market_data_lookup.get(pe_tok, {})
        
        ce_price = float(ce_data.get('ltp', 0))
        pe_price = float(pe_data.get('ltp', 0))
        ce_oi = float(ce_data.get('opnInterest', ce_data.get('openinterest', 0)))
        pe_oi = float(pe_data.get('opnInterest', pe_data.get('openinterest', 0)))

        log_placeholder.write(
            f"📊 Strike {K:.0f} ({idx+1}/{total_strikes}) -> "
            f"CE Price: ₹{ce_price:.2f}, PE Price: ₹{pe_price:.2f} | "
            f"CE OI: {ce_oi:.0f}, PE OI: {pe_oi:.0f}"
        )

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

        ce_iv, ce_err = calculate_safe_iv(ce_price, spot_price, K, T, RISK_FREE_RATE, 'c')
        pe_iv, pe_err = calculate_safe_iv(pe_price, spot_price, K, T, RISK_FREE_RATE, 'p')

        if ce_err or pe_err:
            err_msg = ce_err if ce_err else pe_err
            log_placeholder.write(f"⚠️ Strike {K:.0f}: Skipped -> Reason: {err_msg}")
            continue

        try:
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
            
            log_placeholder.write(f"✅ Strike {K:.0f}: Passed all filters & quantified.")

        except Exception as calc_err:
            log_placeholder.write(f"❌ Strike {K:.0f}: Error during pricing calculation ({str(calc_err)})")
            continue
            
    df_quant = pd.DataFrame(records)
    log_placeholder.success(f"✅ Quant Engine Complete: Retained {len(df_quant)} strike pairs matching criteria.")
    return df_quant, T


# ==============================================================================
# 3. OPTIMIZER WITH STRICT RISK CONTROLS & SELECTION AUDIT LOGGING
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
            
            # Enforce Short Put MUST be OTM (below spot) and outside minimum safety buffer
            if short_put_k >= spot:
                audit_logs.append({
                    'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/X/X",
                    'status': 'Rejected',
                    'reason': f"Short Put ({short_put_k:.0f}) is In-The-Money (>= Spot {spot:.1f})"
                })
                continue

            if (spot - short_put_k) < MIN_OTM_BUFFER:
                audit_logs.append({
                    'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/X/X",
                    'status': 'Rejected',
                    'reason': f"Short Put ({short_put_k:.0f}) too close to spot (Buffer: {spot - short_put_k:.1f} < {MIN_OTM_BUFFER:.1f})"
                })
                continue
                
            for k in range(j + 1, len(strikes)):
                short_call_k = strikes[k]
                
                # Enforce Short Call MUST be OTM (above spot) and outside minimum safety buffer
                if short_call_k <= spot:
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/X",
                        'status': 'Rejected',
                        'reason': f"Short Call ({short_call_k:.0f}) is In-The-Money (<= Spot {spot:.1f})"
                    })
                    continue

                if (short_call_k - spot) < MIN_OTM_BUFFER:
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/X",
                        'status': 'Rejected',
                        'reason': f"Short Call ({short_call_k:.0f}) too close to spot (Buffer: {short_call_k - spot:.1f} < {MIN_OTM_BUFFER:.1f})"
                    })
                    continue
                    
                long_call_k = short_call_k + wing_width
                if long_call_k not in strikes:
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
                        'status': 'Rejected',
                        'reason': f"Symmetric Long Call Strike ({long_call_k:.0f}) not in liquid strikes"
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
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
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
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
                        'status': 'Rejected',
                        'reason': f"PoP ({pop*100:.1f}%) < 52.0%"
                    })
                    continue
                    
                base_margin = max_loss * LOT_SIZE
                buffered_margin = base_margin * 1.30  
                expected_value = (net_credit * pop) - (max_loss * (1 - pop))
                
                if expected_value <= 0:
                    audit_logs.append({
                        'combo': f"{long_put_k:.0f}/{short_put_k:.0f}/{short_call_k:.0f}/{long_call_k:.0f}",
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

# TOP-RIGHT EXECUTION PROGRESS MONITOR
with st.sidebar:
    st.markdown("---")
    st.subheader("🖥️ Execution Log Monitor")
    exec_status = st.status("Initializing Engine...", expanded=True)

try:
    exec_status.write("🔑 Authenticating with SmartAPI...")
    smartApi = authenticate()
    st.success(f"🟢 **API Connection Status:** Successfully connected to SmartAPI (Client ID: `{CLIENT_CODE}`)")
    exec_status.write("✅ Authentication Successful.")

    chain, expiry_dt = get_nifty_option_chain(exec_status)
    
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

    # AUDIT TRAIL DISPLAY
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
