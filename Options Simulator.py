import os
import json
import math
import pyotp
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time, timezone, timedelta
from flask import Flask, render_template_string, jsonify, request
from SmartApi import SmartConnect
from scipy.stats import norm

app = Flask(__name__)

# Environment Configuration
API_KEY = os.environ.get("API_KEY")
CLIENT_CODE = os.environ.get("CLIENT_CODE")
PIN = os.environ.get("PIN")
TOTP_SECRET = os.environ.get("TOTP_SECRET")

IST = timezone(timedelta(hours=5, minutes=30))
RISK_FREE_RATE = 0.065

INDEX_TOKENS = {
    "NIFTY": {"exchange": "NSE", "tradingsymbol": "NIFTY", "token": "99926000", "step": 50},
    "BANKNIFTY": {"exchange": "NSE", "tradingsymbol": "BANKNIFTY", "token": "99926009", "step": 100}
}

INTERVAL_MAP = {
    "1": "ONE_MINUTE",
    "3": "THREE_MINUTE",
    "5": "FIVE_MINUTE",
    "15": "FIFTEEN_MINUTE"
}

smart_api_session = None
INSTRUMENT_DF = None
SCRIP_MASTER_STATUS = "Initializing Scrip Master..."
IS_DOWNLOADING_MASTER = False

def get_smart_api():
    global smart_api_session
    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        return None, "Missing Render Env Variables"

    try:
        if smart_api_session is None:
            obj = SmartConnect(api_key=API_KEY)
            totp = pyotp.TOTP(TOTP_SECRET).now()
            data = obj.generateSession(CLIENT_CODE, PIN, totp)
            
            if data and data.get('status') and data.get('data') and data['data'].get('jwtToken'):
                smart_api_session = obj
                return smart_api_session, "API connection success"
            else:
                msg = data.get('message', 'Authentication Failed') if data else 'No response from SmartAPI'
                return None, f"Couldnt log in: {msg}"
        else:
            return smart_api_session, "API connection success"
            
    except Exception as e:
        smart_api_session = None
        return None, f"Couldnt log in: {str(e)}"

def download_scrip_master_thread():
    global INSTRUMENT_DF, SCRIP_MASTER_STATUS, IS_DOWNLOADING_MASTER
    if INSTRUMENT_DF is not None or IS_DOWNLOADING_MASTER:
        return

    IS_DOWNLOADING_MASTER = True
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    try:
        SCRIP_MASTER_STATUS = "Downloading Scrip Master: 0%"
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunks = []
            
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    mb_dl = round(downloaded / (1024 * 1024), 1)
                    if total_size > 0:
                        mb_tot = round(total_size / (1024 * 1024), 1)
                        pct = int((downloaded / total_size) * 100)
                        SCRIP_MASTER_STATUS = f"Downloading Scrip Master: {mb_dl} MB / {mb_tot} MB ({pct}%)"
                    else:
                        SCRIP_MASTER_STATUS = f"Downloading Scrip Master: {mb_dl} MB"

            SCRIP_MASTER_STATUS = "Processing Scrip Master JSON..."
            content = b"".join(chunks)
            data = json.loads(content.decode('utf-8'))
            
            df = pd.DataFrame(data)
            df = df[(df['exch_seg'] == 'NFO') & (df['instrumenttype'].isin(['OPTIDX', 'OPTSTK']))].copy()
            df['strike_price'] = df['strike'].astype(float) / 100.0
            
            INSTRUMENT_DF = df
            SCRIP_MASTER_STATUS = "Scrip Master Loaded"
        else:
            SCRIP_MASTER_STATUS = f"Scrip Master HTTP Error {response.status_code}"
    except Exception as e:
        SCRIP_MASTER_STATUS = f"Scrip Master Download Failed: {str(e)}"
    finally:
        IS_DOWNLOADING_MASTER = False

def ensure_scrip_master_loading():
    if INSTRUMENT_DF is None and not IS_DOWNLOADING_MASTER:
        t = threading.Thread(target=download_scrip_master_thread, daemon=True)
        t.start()

def is_market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)

def get_last_trading_day_dates(num_days=5):
    now = datetime.now(IST)
    end_date = now.date()
    if not is_market_open() and now.time() < time(9, 15):
        end_date -= timedelta(days=1)
        
    while end_date.weekday() >= 5:
        end_date -= timedelta(days=1)

    start_date = end_date
    trading_days = 0
    while trading_days < num_days:
        start_date -= timedelta(days=1)
        if start_date.weekday() < 5:
            trading_days += 1

    from_str = start_date.strftime("%Y-%m-%d 09:15")
    to_str = now.strftime("%Y-%m-%d %H:%M") if is_market_open() else end_date.strftime("%Y-%m-%d 15:30")
    return from_str, to_str

def fetch_option_chain_data(smart_api, symbol, expiry):
    try:
        chain_params = {"name": symbol, "expirydate": expiry}
        res = smart_api.optionChain(chain_params)
        if res and res.get('status') and res.get('data'):
            return res['data']
    except Exception:
        pass
    return []

# --- PCR CALCULATION ENGINE ---
def calculate_pcr(option_chain_data, spot_price, step, strike_range_limit=None):
    if not option_chain_data:
        return "N/A"

    total_put_oi = 0
    total_call_oi = 0

    atm_strike = round(spot_price / step) * step if spot_price > 0 else None

    filtered_chain = option_chain_data
    if strike_range_limit is not None and strike_range_limit > 0 and atm_strike:
        lower_bound = atm_strike - (strike_range_limit * step)
        upper_bound = atm_strike + (strike_range_limit * step)
        filtered_chain = [
            item for item in option_chain_data
            if lower_bound <= float(item.get('strikePrice', 0) or 0) <= upper_bound
        ]

    for item in filtered_chain:
        opt_type = str(item.get('optionType', '')).upper()
        oi = float(item.get('opennterest', 0) or item.get('openInterest', 0) or item.get('oi', 0) or 0)
        
        if opt_type in ['PE', 'PUT']:
            total_put_oi += oi
        elif opt_type in ['CE', 'CALL']:
            total_call_oi += oi

    if total_call_oi == 0:
        return "N/A"

    pcr_val = total_put_oi / total_call_oi
    return round(pcr_val, 2)

# --- TECHNICAL INDICATORS & ACCURATE DIVERGENCE ENGINE ---
def calculate_chart_indicators(candles):
    if not candles or len(candles) < 26:
        return {}, "BB (20,2) : N/A", "MACD (12,26,9) : N/A", "RSI (14) Divergence : N/A", [], [], []

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)

    # 1. Bollinger Bands (20, 2)
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']

    recent_width = df['bb_width'].dropna().iloc[-1] if not df['bb_width'].dropna().empty else 0
    avg_width = df['bb_width'].dropna().tail(20).mean() if not df['bb_width'].dropna().empty else 0
    latest_close = df['close'].iloc[-1]
    
    if recent_width > 0 and recent_width < (avg_width * 0.75):
        bb_status = "BB (20,2) : Squeeze active"
    elif not df['bb_upper'].dropna().empty and latest_close >= df['bb_upper'].dropna().iloc[-1]:
        bb_status = "BB (20,2) : Upper Breakout"
    elif not df['bb_lower'].dropna().empty and latest_close <= df['bb_lower'].dropna().iloc[-1]:
        bb_status = "BB (20,2) : Lower Breakdown"
    else:
        bb_status = "BB (20,2) : Normal"

    # 2. MACD (12, 26, 9)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    curr_macd = df['macd'].iloc[-1]
    curr_signal = df['macd_signal'].iloc[-1]
    prev_macd = df['macd'].iloc[-2]
    prev_signal = df['macd_signal'].iloc[-2]

    if prev_macd <= prev_signal and curr_macd > curr_signal:
        macd_status = "MACD (12,26,9) : Bullish Crossover"
    elif prev_macd >= prev_signal and curr_macd < curr_signal:
        macd_status = "MACD (12,26,9) : Bearish Crossover"
    elif curr_macd > curr_signal:
        macd_status = "MACD (12,26,9) : Uptrend Continuation"
    else:
        macd_status = "MACD (12,26,9) : Downtrend Continuation"

    # 3. RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- REFINED DIVERGENCE DETECTION LOGIC ---
    price_divergence_points = [None] * len(df)
    macd_divergence_points = [None] * len(df)
    rsi_divergence_points = [None] * len(df)
    rsi_status = "RSI (14) Divergence : None"

    w = 3 # Swing window
    peaks = []
    troughs = []

    # Find genuine swing highs (peaks) and swing lows (troughs)
    for i in range(w, len(df) - w):
        if df['close'].iloc[i] == df['close'].iloc[i-w:i+w+1].max():
            peaks.append(i)
        if df['close'].iloc[i] == df['close'].iloc[i-w:i+w+1].min():
            troughs.append(i)

    # Detect Bearish Divergence (Higher High in Price, Lower High in RSI/MACD)
    for k in range(1, len(peaks)):
        curr_p = peaks[k]
        prev_p = peaks[k-1]

        # Ensure peaks are reasonable distance apart (between 5 and 35 bars)
        if 5 <= (curr_p - prev_p) <= 35:
            price_hh = df['close'].iloc[curr_p] > df['close'].iloc[prev_p]
            rsi_lh = df['rsi'].iloc[curr_p] < df['rsi'].iloc[prev_p]
            macd_lh = df['macd'].iloc[curr_p] < df['macd'].iloc[prev_p]

            if price_hh and rsi_lh:
                rsi_divergence_points[curr_p] = round(df['rsi'].iloc[curr_p], 2)
                price_divergence_points[curr_p] = round(df['close'].iloc[curr_p], 2)
                rsi_status = "RSI (14) Divergence : Bearish Divergence"

            if price_hh and macd_lh:
                macd_divergence_points[curr_p] = round(df['macd'].iloc[curr_p], 2)
                if price_divergence_points[curr_p] is None:
                    price_divergence_points[curr_p] = round(df['close'].iloc[curr_p], 2)

    # Detect Bullish Divergence (Lower Low in Price, Higher Low in RSI/MACD)
    for k in range(1, len(troughs)):
        curr_t = troughs[k]
        prev_t = troughs[k-1]

        if 5 <= (curr_t - prev_t) <= 35:
            price_ll = df['close'].iloc[curr_t] < df['close'].iloc[prev_t]
            rsi_hl = df['rsi'].iloc[curr_t] > df['rsi'].iloc[prev_t]
            macd_hl = df['macd'].iloc[curr_t] > df['macd'].iloc[prev_t]

            if price_ll and rsi_hl:
                rsi_divergence_points[curr_t] = round(df['rsi'].iloc[curr_t], 2)
                price_divergence_points[curr_t] = round(df['close'].iloc[curr_t], 2)
                rsi_status = "RSI (14) Divergence : Bullish Divergence"

            if price_ll and macd_hl:
                macd_divergence_points[curr_t] = round(df['macd'].iloc[curr_t], 2)
                if price_divergence_points[curr_t] is None:
                    price_divergence_points[curr_t] = round(df['close'].iloc[curr_t], 2)

    def clean_series(series):
        return [None if np.isnan(val) else round(float(val), 2) for val in series]

    indicator_series = {
        "bb_upper": clean_series(df['bb_upper']),
        "bb_middle": clean_series(df['sma20']),
        "bb_lower": clean_series(df['bb_lower']),
        "macd": clean_series(df['macd']),
        "macd_signal": clean_series(df['macd_signal']),
        "macd_hist": clean_series(df['macd_hist']),
        "rsi": clean_series(df['rsi'])
    }

    return indicator_series, bb_status, macd_status, rsi_status, price_divergence_points, macd_divergence_points, rsi_divergence_points

# --- BLACK-SCHOLES IV & GREEKS ENGINE ---
def calculate_implied_volatility(market_price, spot, strike, t_years, r, opt_type='CE'):
    if market_price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
        return None

    intrinsic = max(0, spot - strike) if opt_type.upper() in ['CE', 'CALL'] else max(0, strike - spot)
    if market_price <= intrinsic:
        return None

    sigma = 0.15
    for _ in range(20):
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
        d2 = d1 - sigma * math.sqrt(t_years)

        if opt_type.upper() in ['CE', 'CALL']:
            price = spot * norm.cdf(d1) - strike * math.exp(-r * t_years) * norm.cdf(d2)
        else:
            price = strike * math.exp(-r * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        vega = spot * norm.pdf(d1) * math.sqrt(t_years)
        diff = price - market_price
        
        if abs(diff) < 1e-4:
            return round(sigma * 100.0, 2)
            
        if vega < 1e-6:
            break
            
        sigma = sigma - diff / vega

    return round(max(sigma, 0.01) * 100.0, 2)

def calculate_black_scholes_greeks(spot, strike, t_years, iv_pct, opt_type='CE'):
    greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "expected_day_theta": 0.0}
    sigma = max(iv_pct, 1.0) / 100.0
    t = max(t_years, 0.0001)
    r = RISK_FREE_RATE

    if spot <= 0 or strike <= 0:
        return greeks

    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    
    gamma = pdf_d1 / (spot * sigma * math.sqrt(t))
    vega = (spot * pdf_d1 * math.sqrt(t)) / 100.0

    if opt_type.upper() in ['CE', 'CALL']:
        delta = norm.cdf(d1)
        theta_annual = - (spot * pdf_d1 * sigma) / (2 * math.sqrt(t)) - r * strike * math.exp(-r * t) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = - (spot * pdf_d1 * sigma) / (2 * math.sqrt(t)) + r * strike * math.exp(-r * t) * norm.cdf(-d2)

    theta_day = theta_annual / 365.0

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta_day, 4),
        "vega": round(vega, 4),
        "expected_day_theta": round(theta_day, 4)
    }

def get_atm_iv_and_leg_greeks(smart_api, option_chain, spot_price, step, strike, opt_type, expiry_str, symbol="NIFTY"):
    atm_strike = round(spot_price / step) * step
    now = datetime.now(IST)
    try:
        exp_dt = datetime.strptime(expiry_str, "%d%b%Y").replace(hour=15, minute=30, tzinfo=IST)
        time_diff = exp_dt - now
        days_remaining = max(time_diff.total_seconds() / (24 * 3600), 0.001)
    except Exception:
        days_remaining = 1.0
        
    t_years = days_remaining / 365.0
    atm_iv = None

    if option_chain:
        for item in option_chain:
            item_strike = float(item.get('strikePrice', 0) or 0)
            if abs(item_strike - atm_strike) < 0.01:
                iv_val = float(item.get('impliedVolatility', 0) or 0)
                if iv_val > 0:
                    atm_iv = iv_val
                    break

    if atm_iv is None and smart_api and INSTRUMENT_DF is not None:
        try:
            match_ce = INSTRUMENT_DF[
                (INSTRUMENT_DF['name'] == symbol) & 
                (INSTRUMENT_DF['expiry'] == expiry_str) & 
                (INSTRUMENT_DF['strike_price'] == atm_strike) & 
                (INSTRUMENT_DF['symbol'].str.endswith('CE'))
            ]
            match_pe = INSTRUMENT_DF[
                (INSTRUMENT_DF['name'] == symbol) & 
                (INSTRUMENT_DF['expiry'] == expiry_str) & 
                (INSTRUMENT_DF['strike_price'] == atm_strike) & 
                (INSTRUMENT_DF['symbol'].str.endswith('PE'))
            ]

            ce_ltp, pe_ltp = None, None
            if not match_ce.empty:
                res_ce = smart_api.ltpData("NFO", match_ce.iloc[0]['symbol'], str(match_ce.iloc[0]['token']))
                if res_ce and res_ce.get('data'):
                    ce_ltp = float(res_ce['data']['ltp'])

            if not match_pe.empty:
                res_pe = smart_api.ltpData("NFO", match_pe.iloc[0]['symbol'], str(match_pe.iloc[0]['token']))
                if res_pe and res_pe.get('data'):
                    pe_ltp = float(res_pe['data']['ltp'])

            iv_ce = calculate_implied_volatility(ce_ltp, spot_price, atm_strike, t_years, RISK_FREE_RATE, 'CE') if ce_ltp else None
            iv_pe = calculate_implied_volatility(pe_ltp, spot_price, atm_strike, t_years, RISK_FREE_RATE, 'PE') if pe_ltp else None

            if iv_ce and iv_pe:
                atm_iv = (iv_ce + iv_pe) / 2.0
            elif iv_ce:
                atm_iv = iv_ce
            elif iv_pe:
                atm_iv = iv_pe
        except Exception:
            pass

    if atm_iv is None and smart_api:
        try:
            vix_res = smart_api.ltpData("NSE", "INDIA VIX", "26009")
            if vix_res and vix_res.get('data'):
                atm_iv = float(vix_res['data']['ltp'])
        except Exception:
            pass

    atm_iv = atm_iv if atm_iv is not None else 10.0
    leg_greeks = calculate_black_scholes_greeks(spot_price, strike, t_years, atm_iv, opt_type)
    return round(atm_iv, 2), leg_greeks

def get_past_price_within_market_hours(candles, duration_mins):
    if not candles:
        return None, None

    market_open_time = time(9, 15)
    market_close_time = time(15, 30)

    valid_candles = []
    for c in candles:
        try:
            dt_str = c[0].split('T')[1][:5]
            c_time = datetime.strptime(dt_str, "%H:%M").time()
            if market_open_time <= c_time <= market_close_time:
                valid_candles.append((dt_str, float(c[4])))
        except Exception:
            continue

    if not valid_candles:
        return None, None

    latest_ts, latest_price = valid_candles[-1]
    try:
        latest_dt = datetime.strptime(latest_ts, "%H:%M")
        target_dt = latest_dt - timedelta(minutes=duration_mins)
        market_open_dt = datetime.strptime("09:15", "%H:%M")
        if target_dt < market_open_dt:
            target_dt = market_open_dt
        target_str = target_dt.strftime("%H:%M")
    except Exception:
        target_str = valid_candles[0][0]

    best_price = valid_candles[0][1]
    for ts, price in valid_candles:
        if ts <= target_str:
            best_price = price
        else:
            break

    return latest_price, best_price

def calculate_basket_net_greeks(legs):
    net_delta, net_gamma, net_theta, net_vega = 0.0, 0.0, 0.0, 0.0
    for leg in legs:
        qty = int(leg.get('qty', 65))
        direction = 1 if leg.get('action') == 'BUY' else -1
        g = leg.get('greeks', {})
        
        net_delta += (g.get('delta', 0.0) * qty * direction)
        net_gamma += (g.get('gamma', 0.0) * qty * direction)
        net_theta += (g.get('theta', 0.0) * qty * direction)
        net_vega += (g.get('vega', 0.0) * qty * direction)

    return {
        "delta": round(net_delta, 2),
        "gamma": round(net_gamma, 4),
        "theta": round(net_theta, 2),
        "vega": round(net_vega, 2),
        "expected_day_theta": round(net_theta, 2)
    }

def calculate_max_profit_loss(legs):
    if not legs:
        return "₹0", "₹0"

    total_net_credit = 0.0
    short_calls, long_calls = [], []
    short_puts, long_puts = [], []

    for leg in legs:
        qty = int(leg.get('qty', 65))
        entry_price = float(leg.get('entry_price', 0.0) or 0.0)
        action = leg.get('action', 'BUY')
        opt_type = leg.get('option_type', 'CE')
        strike = float(leg.get('strike', 0))

        if action == 'SELL':
            total_net_credit += (entry_price * qty)
            if opt_type in ['CE', 'C']:
                short_calls.append(strike)
            else:
                short_puts.append(strike)
        elif action == 'BUY':
            total_net_credit -= (entry_price * qty)
            if opt_type in ['CE', 'C']:
                long_calls.append(strike)
            else:
                long_puts.append(strike)

    call_hedged = False
    call_width = 0
    if short_calls:
        if long_calls and min(long_calls) > max(short_calls):
            call_hedged = True
            call_width = min(long_calls) - max(short_calls)

    put_hedged = False
    put_width = 0
    if short_puts:
        if long_puts and max(long_puts) < min(short_puts):
            put_hedged = True
            put_width = min(short_puts) - max(long_puts)

    if short_calls and short_puts and call_hedged and put_hedged:
        max_width = max(call_width, put_width) * qty
        max_profit = total_net_credit
        max_loss = max_width - total_net_credit
        return f"₹{round(max_profit, 2)}", f"₹{round(max_loss, 2)}"

    if short_calls and call_hedged and not short_puts:
        max_width = call_width * qty
        max_profit = total_net_credit
        max_loss = max_width - total_net_credit
        return f"₹{round(max_profit, 2)}", f"₹{round(max_loss, 2)}"

    if short_puts and put_hedged and not short_calls:
        max_width = put_width * qty
        max_profit = total_net_credit
        max_loss = max_width - total_net_credit
        return f"₹{round(max_profit, 2)}", f"₹{round(max_loss, 2)}"

    max_profit_str = "Uncapped" if (long_calls or long_puts) and not (short_calls or short_puts) else f"₹{round(total_net_credit, 2)}"
    max_loss_str = "Uncapped" if (short_calls and not call_hedged) or (short_puts and not put_hedged) else "Defined"

    return max_profit_str, max_loss_str

def identify_strategy(legs):
    if not legs:
        return "Custom Strategy"
    
    n_legs = len(legs)
    buys = [l for l in legs if l['action'] == 'BUY']
    sells = [l for l in legs if l['action'] == 'SELL']

    if n_legs == 1:
        leg = legs[0]
        return f"Long {leg['option_type']}" if leg['action'] == 'BUY' else f"Short {leg['option_type']}"

    if n_legs == 2:
        if len(buys) == 1 and len(sells) == 1:
            b_leg, s_leg = buys[0], sells[0]
            if b_leg['option_type'] == 'CE' and s_leg['option_type'] == 'CE':
                return "Bull Call Spread" if b_leg['strike'] < s_leg['strike'] else "Bear Call Spread"
            if b_leg['option_type'] == 'PE' and s_leg['option_type'] == 'PE':
                return "Bear Put Spread" if b_leg['strike'] > s_leg['strike'] else "Bull Put Spread"
        elif len(buys) == 2:
            if set(l['option_type'] for l in buys) == {'CE', 'PE'}:
                strikes = set(l['strike'] for l in buys)
                return "Long Straddle" if len(strikes) == 1 else "Long Strangle"
        elif len(sells) == 2:
            if set(l['option_type'] for l in sells) == {'CE', 'PE'}:
                strikes = set(l['strike'] for l in sells)
                return "Short Straddle" if len(strikes) == 1 else "Short Strangle"

    if n_legs == 4 and len(buys) == 2 and len(sells) == 2:
        types = set(l['option_type'] for l in legs)
        if types == {'CE', 'PE'}:
            return "Iron Condor"

    return "Custom Multi-Leg"

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live SmartAPI Options Position Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 20px; padding-top: 35px; }
        
        #apiStatusPill {
            position: fixed; top: 10px; left: 15px; z-index: 9999;
            background: rgba(20, 20, 20, 0.95); border: 1px solid #00bcd4;
            color: #00ff88; padding: 6px 14px; border-radius: 20px;
            font-size: 11px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            max-width: 90vw; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        #refreshToggleOverlay {
            position: fixed; top: 10px; right: 15px; z-index: 9999;
            background: rgba(20, 20, 20, 0.95); border: 1px solid #444;
            color: #fff; padding: 8px 16px; border-radius: 20px;
            font-size: 11px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            display: flex; align-items: center; gap: 8px; cursor: pointer;
            white-space: nowrap; width: auto; max-width: none;
        }

        .pill-err { border-color: #ff5252 !important; color: #ff5252 !important; }
        .pill-warn { border-color: #ffca28 !important; color: #ffca28 !important; }

        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-top: 15px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 15px; }
        .chart-container-relative { position: relative; }

        .indicator-ribbon {
            position: absolute;
            top: 12px;
            right: 15px;
            background: rgba(18, 18, 18, 0.88);
            border: 1px solid #00bcd4;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 11px;
            font-family: monospace;
            z-index: 10;
            box-shadow: 0 2px 6px rgba(0,0,0,0.6);
            line-height: 1.4;
        }
        .ribbon-item { display: block; color: #00ff88; }

        .pcr-control-box {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid #00bcd4;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            color: #fff;
        }

        .pcr-input {
            width: 50px !important;
            padding: 2px 4px !important;
            margin: 0 !important;
            height: 22px;
            font-size: 11px;
            text-align: center;
            background: #121212;
            border: 1px solid #555;
            color: #00ff88;
        }

        .pcr-refresh-btn {
            background: transparent;
            border: 1px solid #00bcd4;
            color: #00bcd4;
            border-radius: 3px;
            padding: 2px 6px;
            cursor: pointer;
            font-size: 11px;
            margin: 0;
            width: auto;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .pcr-refresh-btn:hover { background: rgba(0, 188, 212, 0.2); }

        h2, h3 { margin-top: 0; color: #00bcd4; }
        
        .header-flex { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .header-flex h3 { margin: 0; }
        .chart-select { width: auto !important; padding: 4px 8px !important; margin: 0 !important; font-size: 12px; }

        label { display: block; margin-top: 10px; font-size: 12px; color: #aaa; }
        select, input, button { width: 100%; padding: 8px; margin-top: 5px; background: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; box-sizing: border-box; }
        button { background: #00bcd4; color: black; font-weight: bold; cursor: pointer; border: none; margin-top: 15px; }
        button:hover { background: #008ba3; }
        
        .btn-delete { background: #ff5252; color: white; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; width: auto; margin: 0; display: inline-flex; align-items: center; gap: 4px; }
        .btn-delete:hover { background: #d32f2f; }
        
        .pending-leg-item { display: flex; justify-content: space-between; align-items: center; background: #2a2a2a; padding: 6px; border-radius: 4px; margin-top: 5px; font-size: 12px; }

        .status-bar { display: flex; gap: 12px; background: #262626; padding: 10px; border-radius: 6px; font-weight: bold; margin-bottom: 15px; flex-wrap: wrap; }
        .status-item { font-size: 12px; }
        .status-value { color: #00ff88; }
        .pnl-pos { color: #00ff88; font-weight: bold; }
        .pnl-neg { color: #ff5252; font-weight: bold; }
        .pnl-err { color: #ff9800; font-weight: bold; }
        .greek-tag { font-family: monospace; background: #2d2d2d; padding: 3px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; display: inline-block; margin-top: 4px; }
        .basket-summary-tag { font-family: monospace; background: #1a3835; border: 1px solid #00bcd4; padding: 3px 7px; border-radius: 4px; font-size: 11px; color: #00e5ff; margin-left: 6px; display: inline-block; margin-top: 4px; }
        .strategy-badge { background: #00bcd4; color: #000; font-weight: bold; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 8px; }

        .chart-checkbox-container {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 11px;
            color: #00ff88;
            margin-right: 12px;
            cursor: pointer;
            user-select: none;
        }

        .header-spinner {
            width: 12px;
            height: 12px;
            border: 2px solid rgba(0, 188, 212, 0.3);
            border-radius: 50%;
            border-top-color: #00bcd4;
            animation: spin 0.8s linear infinite;
            display: inline-block;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .basket-status-msg {
            font-size: 11px;
            color: #00bcd4;
            font-weight: normal;
            margin-left: 8px;
            display: inline-flex;
            align-items: center;
        }

        .subchart-title {
            font-size: 11px;
            color: #aaa;
            margin: 12px 0 2px 0;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <div id="apiStatusPill">Status: Logging in API...</div>

    <div id="refreshToggleOverlay">
        <input type="checkbox" id="chkAutoRefresh" checked onchange="toggleRefreshInterval()" style="margin:0; width:auto;">
        <label for="chkAutoRefresh" style="margin:0; cursor:pointer; color:#fff; white-space:nowrap; display:inline;">Enable Refresh 5 sec</label>
    </div>

    <h2>Real-Market Options Strategy Tracker (SmartAPI Live)</h2>
    <div class="grid">
        <div>
            <div class="card">
                <h3>1. Select Underlying & Expiry</h3>
                <label>Index Symbol</label>
                <select id="symbol" onchange="loadExpiries()">
                    <option value="NIFTY">NIFTY 50</option>
                    <option value="BANKNIFTY">BANKNIFTY</option>
                </select>

                <label>Expiry Date</label>
                <select id="expirySelect" onchange="loadChain()"></select>

                <label>Exchange Segment</label>
                <select id="exchange">
                    <option value="NFO">NFO</option>
                </select>
            </div>

            <div class="card">
                <h3>2. Build Strategy Basket</h3>
                <label>Strike Price</label>
                <select id="strikeSelect"></select>

                <label>Option Type</label>
                <select id="optType">
                    <option value="CE">CE (Call)</option>
                    <option value="PE">PE (Put)</option>
                </select>

                <label>Trade Action</label>
                <select id="action">
                    <option value="BUY">BUY</option>
                    <option value="SELL" selected>SELL</option>
                </select>

                <label>Entry Price (₹)</label>
                <input type="number" id="entryPrice" placeholder="Auto-fetches LTP if blank" step="0.05">

                <label>Quantity / Lots</label>
                <input type="number" id="qty" value="65">

                <button onclick="addLegToPending()">+ Add Position Leg</button>
                
                <div id="pendingContainer" style="margin-top: 12px;"></div>
                
                <button onclick="deployBasket()" style="background: #4caf50; color: white;">Execute Strategy Basket</button>
            </div>
        </div>

        <div>
            <div class="status-bar">
                <div class="status-item">Index LTP: <span id="stIndex" class="status-value">-</span></div>
                <div class="status-item">Dynamic ATM IV: <span id="stIv" class="status-value">-</span></div>
                <div class="status-item">Expected 1-Day Move: <span id="stMove" class="status-value">-</span></div>
                <div class="status-item">Expected Expiry Move: <span id="stExpiryMove" class="status-value">-</span></div>
                <div class="status-item">PCR: <span id="stPcr" class="status-value">-</span></div>
                <div class="status-item">Market Status: <span id="stMarketStatus" class="status-value">-</span></div>
            </div>

            <div class="card chart-container-relative">
                <div id="indicatorRibbon" class="indicator-ribbon">
                    <span id="ribbonBB" class="ribbon-item">BB (20,2) : -</span>
                    <span id="ribbonMACD" class="ribbon-item">MACD (12,26,9) : -</span>
                    <span id="ribbonRSI" class="ribbon-item">RSI (14) Divergence : -</span>
                </div>

                <div class="header-flex">
                    <h3>Index Price & Bollinger Bands (20, 2)</h3>
                    <div style="display: flex; gap: 10px; align-items: center; margin-right: 250px;">
                        <div class="pcr-control-box" title="Blank or 0 = Entire Option Chain">
                            <span>PCR &plusmn; Range:</span>
                            <input type="number" id="pcrStrikeRange" class="pcr-input" placeholder="All" min="1" max="50">
                            <button class="pcr-refresh-btn" onclick="refreshPcrValue()" title="Recalculate PCR">
                                <span id="pcrSpinner" class="header-spinner" style="display:none;"></span>
                                🔄
                            </button>
                        </div>
                        <div>
                            <label style="display:inline; color:#aaa; font-size:11px; margin-right:3px;">Candle Timeframe:</label>
                            <select id="timeframeSelect" class="chart-select" onchange="updateDashboard()">
                                <option value="1">1 min</option>
                                <option value="3">3 mins</option>
                                <option value="5">5 mins</option>
                                <option value="15" selected>15 mins</option>
                            </select>
                        </div>
                    </div>
                </div>
                <canvas id="mainChart" height="90"></canvas>

                <!-- Subcharts for MACD and RSI -->
                <div class="subchart-title">MACD (12, 26, 9 EMA) Indicator with Divergence Points</div>
                <canvas id="macdChart" height="40"></canvas>

                <div class="subchart-title">RSI (14) Indicator with Divergence Points</div>
                <canvas id="rsiChart" height="70"></canvas>
            </div>

            <div class="card">
                <div style="display: flex; align-items: center;">
                    <h3 style="margin: 0;">Active Basket Positions & Real-Time Option LTP</h3>
                    <span id="basketHeaderStatus"></span>
                </div>
                <div id="basketsContainer" style="margin-top: 15px;"></div>
            </div>

            <div class="card">
                <div class="header-flex">
                    <h3>Basket Legs Real-Time Premium Chart & P&L Zones</h3>
                    <div>
                        <label style="display:inline; color:#aaa; font-size:12px; margin-right:5px;">Basket Candle Timeframe:</label>
                        <select id="basketTimeframeSelect" class="chart-select" onchange="updateDashboard()">
                            <option value="1">1 min</option>
                            <option value="3">3 mins</option>
                            <option value="5">5 mins</option>
                            <option value="15" selected>15 mins</option>
                        </select>
                    </div>
                </div>
                <canvas id="legsPremiumChart" height="120"></canvas>
            </div>
        </div>
    </div>

    <script>
        let pendingLegs = [];
        let activeBaskets = [];
        let selectedChartBasketId = null;
        let showStrikesPerBasket = {};
        let deletedBasketIds = new Set();
        let mainChart, macdChart, rsiChart, legsChart;
        let refreshTimer = null;

        const CHART_COLORS = ['#00bcd4', '#ff9800', '#e91e63', '#4caf50', '#9c27b0', '#ffeb3b'];

        function updateStatus(text, type='info') {
            const pill = document.getElementById('apiStatusPill');
            pill.innerText = 'Status: ' + text;
            pill.className = '';
            if (type === 'error') pill.classList.add('pill-err');
            if (type === 'warn') pill.classList.add('pill-warn');
        }

        // 1. Main Price & Bollinger Bands Chart
        const ctx = document.getElementById('mainChart').getContext('2d');
        mainChart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'Index Spot Price', data: [], borderColor: '#00bcd4', borderWidth: 1.5, pointRadius: 0, tension: 0.1, spanGaps: true },
                    { label: 'BB Upper (20,2)', data: [], borderColor: 'rgba(255, 82, 82, 0.8)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, spanGaps: true },
                    { label: 'BB Middle SMA (20)', data: [], borderColor: 'rgba(255, 202, 40, 0.8)', borderWidth: 1, pointRadius: 0, spanGaps: true },
                    { label: 'BB Lower (20,2)', data: [], borderColor: 'rgba(76, 175, 80, 0.8)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, spanGaps: true },
                    { label: 'Divergence Marker', data: [], borderColor: '#ff0055', backgroundColor: '#ff0055', pointRadius: 5, pointStyle: 'rectRot', showLine: false }
                ] 
            },
            options: { 
                animation: false,
                responsive: true,
                scales: { 
                    y: { display: true, position: 'left', grid: { color: '#2a2a2a' } },
                    x: { grid: { color: '#2a2a2a' } }
                }
            }
        });

        // 2. MACD Sub-Chart
        const ctxMACD = document.getElementById('macdChart').getContext('2d');
        macdChart = new Chart(ctxMACD, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    { type: 'line', label: 'MACD (12, 26)', data: [], borderColor: '#2196f3', borderWidth: 1.2, pointRadius: 0, spanGaps: true },
                    { type: 'line', label: 'Signal (9 EMA)', data: [], borderColor: '#ff9800', borderWidth: 1.2, pointRadius: 0, spanGaps: true },
                    { type: 'bar', label: 'MACD Hist', data: [], backgroundColor: 'rgba(0, 188, 212, 0.4)' },
                    { type: 'line', label: 'MACD Divergence', data: [], borderColor: '#ff0055', backgroundColor: '#ff0055', pointRadius: 5, pointStyle: 'rectRot', showLine: false }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                scales: {
                    y: { grid: { color: '#2a2a2a' } },
                    x: { display: false }
                }
            }
        });

        // 3. RSI Sub-Chart (Taller Height & Better Scaling)
        const ctxRSI = document.getElementById('rsiChart').getContext('2d');
        rsiChart = new Chart(ctxRSI, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'RSI (14)', data: [], borderColor: '#e91e63', borderWidth: 1.8, pointRadius: 0, tension: 0.1, spanGaps: true },
                    { label: 'RSI Divergence', data: [], borderColor: '#ff0055', backgroundColor: '#ff0055', pointRadius: 5, pointStyle: 'rectRot', showLine: false }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        suggestedMin: 10, 
                        suggestedMax: 90, 
                        grid: { color: '#2a2a2a' }, 
                        ticks: { stepSize: 20, color: '#aaa' } 
                    },
                    x: { grid: { color: '#2a2a2a' }, ticks: { color: '#888' } }
                }
            }
        });

        // 4. Basket Option Premiums Chart
        const ctxLegs = document.getElementById('legsPremiumChart').getContext('2d');
        legsChart = new Chart(ctxLegs, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                animation: false,
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { grid: { color: '#2a2a2a' } },
                    y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Option Premium (₹)', color: '#00bcd4' }, grid: { color: '#2a2a2a' } }
                }
            }
        });

        function toggleRefreshInterval() {
            const isChecked = document.getElementById('chkAutoRefresh').checked;
            if (refreshTimer) clearInterval(refreshTimer);

            if (isChecked) {
                refreshTimer = setInterval(updateDashboard, 5000);
            }
        }

        async function refreshPcrValue() {
            const spinner = document.getElementById('pcrSpinner');
            spinner.style.display = 'inline-block';
            await updateDashboard();
            spinner.style.display = 'none';
        }

        async function loadExpiries() {
            updateStatus("Initializing...", "warn");
            const symbol = document.getElementById('symbol').value;
            try {
                const res = await fetch('/api/fetch-expiries', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol })
                });
                const data = await res.json();
                
                if (data.status) updateStatus(data.status, data.status.includes("Downloading") ? "warn" : "info");

                const select = document.getElementById('expirySelect');
                select.innerHTML = '';
                if(data.expiries && data.expiries.length > 0) {
                    data.expiries.forEach((exp, idx) => {
                        let opt = document.createElement('option');
                        opt.value = exp; opt.textContent = exp;
                        if(idx === 0) opt.selected = true; // Auto Select Nearest Expiry
                        select.appendChild(opt);
                    });
                    loadChain();
                } else if (data.status && data.status.includes("Downloading")) {
                    setTimeout(loadExpiries, 2000);
                }
            } catch(e) { updateStatus("Couldnt log in: Network Error", "error"); }
        }

        async function loadChain() {
            const symbol = document.getElementById('symbol').value;
            const expiry = document.getElementById('expirySelect').value;
            if (!expiry) return;

            try {
                const res = await fetch('/api/fetch-chain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, expiry })
                });
                const data = await res.json();

                const select = document.getElementById('strikeSelect');
                select.innerHTML = '';
                if(data.strikes && data.strikes.length > 0) {
                    data.strikes.forEach(s => {
                        let opt = document.createElement('option');
                        opt.value = s; opt.textContent = s;
                        if(s === data.atm) opt.selected = true;
                        select.appendChild(opt);
                    });
                }
            } catch(e) { updateStatus("Couldnt fetch live price from API", "error"); }
        }

        function addLegToPending() {
            const strike = document.getElementById('strikeSelect').value;
            const expiry = document.getElementById('expirySelect').value;
            const option_type = document.getElementById('optType').value;
            const action = document.getElementById('action').value;
            const entry_price = document.getElementById('entryPrice').value;
            const qty = document.getElementById('qty').value;

            pendingLegs.push({ 
                strike, expiry, option_type, action, 
                entry_price: entry_price ? parseFloat(entry_price) : null, 
                qty: parseInt(qty) 
            });
            renderPendingLegs();
        }

        function removePendingLeg(index) {
            pendingLegs.splice(index, 1);
            renderPendingLegs();
        }

        function renderPendingLegs() {
            const container = document.getElementById('pendingContainer');
            if (pendingLegs.length === 0) { container.innerHTML = ''; return; }

            let html = `<div style="font-size: 12px; color: #ffca28; margin-bottom: 5px; font-weight: bold;">
                Pending Basket (${pendingLegs.length} leg(s)):
            </div>`;

            pendingLegs.forEach((leg, idx) => {
                let entryText = leg.entry_price ? `₹${leg.entry_price}` : 'Auto LTP';
                html += `
                    <div class="pending-leg-item">
                        <span>${leg.strike} ${leg.option_type} (${leg.action}) | ${leg.qty} Qty | ${entryText}</span>
                        <button class="btn-delete" onclick="removePendingLeg(${idx})" title="Remove Leg">❌</button>
                    </div>`;
            });
            container.innerHTML = html;
        }

        function deployBasket() {
            if (pendingLegs.length === 0) return alert("Add position legs first.");
            const basketId = Date.now();
            const newBasketName = `Basket #${activeBaskets.length + 1}`;
            
            if (!selectedChartBasketId) selectedChartBasketId = basketId;
            showStrikesPerBasket[basketId] = false;

            activeBaskets.push({ id: basketId, name: newBasketName, impact_duration: 15, legs: [...pendingLegs] });
            pendingLegs = [];
            renderPendingLegs();

            const statusElem = document.getElementById('basketHeaderStatus');
            statusElem.innerHTML = `<span class="basket-status-msg"><span class="header-spinner"></span> Adding ${newBasketName}...</span>`;

            updateDashboard().then(() => {
                statusElem.innerHTML = '';
            });
        }

        function deleteBasket(basketId) {
            deletedBasketIds.add(basketId);
            activeBaskets = activeBaskets.filter(b => b.id !== basketId);
            delete showStrikesPerBasket[basketId];
            
            if (selectedChartBasketId === basketId) {
                selectedChartBasketId = activeBaskets.length > 0 ? activeBaskets[0].id : null;
            }
            
            updateDashboard();
        }

        function toggleBasketChart(basketId) {
            selectedChartBasketId = selectedChartBasketId === basketId ? null : basketId;
            updateDashboard();
        }

        function toggleShowStrikes(basketId, checkbox) {
            showStrikesPerBasket[basketId] = checkbox.checked;
            updateDashboard();
        }

        function updateBasketImpactDuration(basketId, val) {
            const basket = activeBaskets.find(b => b.id === basketId);
            if (basket) {
                basket.impact_duration = parseInt(val);
                updateDashboard();
            }
        }

        async function updateDashboard() {
            const symbol = document.getElementById('symbol').value;
            const exchange = document.getElementById('exchange').value;
            const interval = document.getElementById('timeframeSelect').value;
            const basketInterval = document.getElementById('basketTimeframeSelect').value;
            const selectedExpiry = document.getElementById('expirySelect').value;
            const pcrRangeVal = document.getElementById('pcrStrikeRange').value;

            activeBaskets = activeBaskets.filter(b => !deletedBasketIds.has(b.id));

            try {
                const res = await fetch('/api/live-data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ 
                        symbol, exchange, interval, basket_interval: basketInterval, 
                        expiry: selectedExpiry, baskets: activeBaskets,
                        pcr_strike_range: pcrRangeVal ? parseInt(pcrRangeVal) : null
                    })
                });
                const data = await res.json();

                if (data.status) updateStatus(data.status, "info");
                if (data.error) { document.getElementById('stIndex').innerText = data.error; return; }

                document.getElementById('stIndex').innerText = data.underlying_price;
                document.getElementById('stIv').innerText = data.current_iv + "%";
                document.getElementById('stMove').innerText = "±" + data.expected_1day_move + " pts";
                document.getElementById('stExpiryMove').innerText = "±" + data.expected_expiry_move + " pts";
                document.getElementById('stPcr').innerText = data.pcr_value;
                document.getElementById('stMarketStatus').innerText = data.is_market_open ? "OPEN (Live)" : "CLOSED (Last Trading Day Data)";

                // Update Header Ribbon
                document.getElementById('ribbonBB').innerText = data.bb_status;
                document.getElementById('ribbonMACD').innerText = data.macd_status;
                document.getElementById('ribbonRSI').innerText = data.rsi_status;

                const validBaskets = (data.baskets || []).filter(b => !deletedBasketIds.has(b.id));

                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    macdChart.data.labels = data.chart_labels;
                    rsiChart.data.labels = data.chart_labels;

                    // Plot Main Price Chart & Bollinger Bands
                    mainChart.data.datasets[0].data = data.chart_prices;
                    if (data.indicators) {
                        mainChart.data.datasets[1].data = data.indicators.bb_upper || [];
                        mainChart.data.datasets[2].data = data.indicators.bb_middle || [];
                        mainChart.data.datasets[3].data = data.indicators.bb_lower || [];
                        mainChart.data.datasets[4].data = data.price_divergences || [];

                        // Plot MACD Subchart
                        macdChart.data.datasets[0].data = data.indicators.macd || [];
                        macdChart.data.datasets[1].data = data.indicators.macd_signal || [];
                        macdChart.data.datasets[2].data = data.indicators.macd_hist || [];
                        macdChart.data.datasets[3].data = data.macd_divergences || [];

                        // Plot RSI Subchart
                        rsiChart.data.datasets[0].data = data.indicators.rsi || [];
                        rsiChart.data.datasets[1].data = data.rsi_divergences || [];
                    }

                    const targetBasket = validBaskets.find(b => b.id === selectedChartBasketId);

                    let baseDatasets = mainChart.data.datasets.slice(0, 5);
                    if (targetBasket && showStrikesPerBasket[targetBasket.id]) {
                        targetBasket.legs.forEach((leg, lIdx) => {
                            let strikeVal = parseFloat(leg.strike);
                            let strikeDataset = {
                                label: `Strike: ${strikeVal} ${leg.option_type}`,
                                data: new Array(data.chart_labels.length).fill(strikeVal),
                                borderColor: CHART_COLORS[lIdx % CHART_COLORS.length],
                                borderDash: [4, 4],
                                borderWidth: 1.2,
                                pointRadius: 0,
                                fill: false,
                                spanGaps: true
                            };
                            baseDatasets.push(strikeDataset);
                        });
                    }
                    mainChart.data.datasets = baseDatasets;

                    mainChart.update('none');
                    macdChart.update('none');
                    rsiChart.update('none');
                }

                const container = document.getElementById('basketsContainer');
                container.innerHTML = '';

                validBaskets.forEach(b => {
                    let pnlDisplay = typeof b.basket_pnl === 'number' ? `₹${b.basket_pnl}` : b.basket_pnl;
                    let pnlClass = typeof b.basket_pnl === 'number' ? (b.basket_pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : 'pnl-err';
                    let isChecked = selectedChartBasketId === b.id;
                    let strikesChecked = showStrikesPerBasket[b.id] ? 'checked' : '';

                    let html = `<div style="border-bottom: 1px solid #333; padding: 10px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
                                <strong>${b.name}</strong>
                                <span class="strategy-badge">${b.strategy_type}</span>
                                <span class="${pnlClass}" style="margin-left: 10px; margin-right: 10px;">Live P&L: ${pnlDisplay}</span>
                                
                                <span class="basket-summary-tag">Net &Delta;: ₹${b.net_greeks.delta} /pt <small>(${b.basket_impact.delta_pct}%)</small></span>
                                <span class="basket-summary-tag">Net &Gamma;: ₹${b.net_greeks.gamma} /pt&sup2;</span>
                                <span class="basket-summary-tag">Net &Theta;: ₹${b.net_greeks.theta} /day <small>(${b.basket_impact.theta_pct}%)</small></span>
                                <span class="basket-summary-tag">Net &Nu;: ₹${b.net_greeks.vega} /1% IV <small>(${b.basket_impact.vega_pct}%)</small></span>
                                <span class="basket-summary-tag" style="color:#00ff88; border-color:#00ff88; background:#1b3821;">Tot Decay: ₹${b.total_decay_till_date}</span>
                                <span class="basket-summary-tag" style="color:#ffca28; border-color:#ffca28; background:#38321b;">Exp Day &Theta;: ₹${b.net_greeks.expected_day_theta}</span>
                            </div>
                            
                            <div style="display:flex; align-items:center; gap: 8px;">
                                <label class="chart-checkbox-container" title="Show leg strike lines on underlying index chart">
                                    <input type="checkbox" ${strikesChecked} onchange="toggleShowStrikes(${b.id}, this)" style="width:auto; margin:0;">
                                    <span>Show Strikes on Chart</span>
                                </label>
                                <label class="chart-checkbox-container">
                                    <input type="checkbox" ${isChecked ? 'checked' : ''} onchange="toggleBasketChart(${b.id})" style="width:auto; margin:0;">
                                    <span>Load Chart</span>
                                </label>
                                <select class="chart-select" onchange="updateBasketImpactDuration(${b.id}, this.value)" title="Duration of % Impact">
                                    <option value="3" ${b.impact_duration === 3 ? 'selected' : ''}>3 mins</option>
                                    <option value="5" ${b.impact_duration === 5 ? 'selected' : ''}>5 mins</option>
                                    <option value="15" ${b.impact_duration === 15 ? 'selected' : ''}>15 mins</option>
                                </select>
                                <button class="btn-delete" onclick="deleteBasket(${b.id})">🗑️ Delete Basket</button>
                            </div>
                        </div>
                        
                        <div style="margin-top: 6px; font-size: 12px; color: #aaa;">
                            <strong>Max Profit:</strong> <span style="color:#00ff88; margin-right:15px;">${b.max_profit}</span>
                            <strong>Max Loss:</strong> <span style="color:#ff5252;">${b.max_loss}</span>
                        </div>`;

                    b.legs.forEach((leg) => {
                        let ltpText = typeof leg.current_premium === 'number' ? `₹${leg.current_premium}` : leg.current_premium;
                        let entryText = typeof leg.entry_price === 'number' ? `₹${leg.entry_price}` : leg.entry_price;
                        let legPnlText = typeof leg.leg_pnl === 'number' ? `₹${leg.leg_pnl}` : leg.leg_pnl;

                        html += `
                            <div style="margin-top:6px; font-size:13px;">
                                <span>[${leg.expiry}] ${leg.strike} ${leg.option_type} (${leg.action}) | Entry: ${entryText} | Real LTP: ${ltpText} | P&L: ${legPnlText}</span>
                                <div>
                                    <span class="greek-tag">&Delta;: ${leg.greeks.delta} <small style="color:#00bcd4;">(${leg.impact.delta_pct}%)</small></span>
                                    <span class="greek-tag">&Gamma;: ${leg.greeks.gamma}</span>
                                    <span class="greek-tag">&Theta;: ${leg.greeks.theta} <small style="color:#00bcd4;">(${leg.impact.theta_pct}%)</small></span>
                                    <span class="greek-tag">&Nu;: ${leg.greeks.vega} <small style="color:#00bcd4;">(${leg.impact.vega_pct}%)</small></span>
                                    <span class="greek-tag" style="color:#00ff88;">Decay Till Date: ₹${leg.theta_decay_till_date}</span>
                                    <span class="greek-tag" style="color:#00bcd4;">Exp Day Theta: ${leg.greeks.expected_day_theta}</span>
                                </div>
                            </div>`;
                    });
                    html += `</div>`;
                    container.innerHTML += html;
                });

                const targetBasket = validBaskets.find(b => b.id === selectedChartBasketId);
                if (targetBasket) {
                    updateLegsHistoricalChart(targetBasket);
                } else {
                    legsChart.data.labels = [];
                    legsChart.data.datasets = [];
                    legsChart.update('none');
                }

            } catch(e) { updateStatus("Couldnt fetch live price from API", "error"); }
        }

        function updateLegsHistoricalChart(basket) {
            if (!basket || !basket.legs_historical) return;

            const hist = basket.legs_historical;
            if (hist.labels && hist.labels.length > 0) {
                legsChart.data.labels = hist.labels;
                let datasets = [];

                if (hist.leg_series) {
                    hist.leg_series.forEach((series, idx) => {
                        let isPositivePnl = basket.basket_pnl >= 0;
                        datasets.push({
                            label: series.label,
                            data: series.prices,
                            borderColor: CHART_COLORS[idx % CHART_COLORS.length],
                            borderWidth: 2,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            fill: { target: 'origin', above: isPositivePnl ? 'rgba(0,255,136,0.1)' : 'rgba(0,255,136,0.03)', below: !isPositivePnl ? 'rgba(255,82,82,0.1)' : 'rgba(255,82,82,0.03)' },
                            tension: 0.1,
                            spanGaps: true
                        });
                    });
                }

                legsChart.data.datasets = datasets;
                legsChart.update('none');
            }
        }

        loadExpiries();
        toggleRefreshInterval();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    ensure_scrip_master_loading()
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fetch-expiries', methods=['POST'])
def fetch_expiries():
    ensure_scrip_master_loading()
    if INSTRUMENT_DF is None:
        return jsonify({"expiries": [], "status": SCRIP_MASTER_STATUS})

    data = request.json or {}
    symbol = data.get('symbol', 'NIFTY')

    filtered = INSTRUMENT_DF[INSTRUMENT_DF['name'] == symbol]
    raw_expiries = filtered['expiry'].unique().tolist()

    # --- CHRONOLOGICAL EXPIRY SORTING ENGINE ---
    parsed_expiries = []
    now_date = datetime.now(IST).date()

    for exp in raw_expiries:
        try:
            exp_date = datetime.strptime(exp, "%d%b%Y").date()
            if exp_date >= now_date:
                parsed_expiries.append((exp_date, exp))
        except Exception:
            pass

    # Sort strictly by calendar date ascending (nearest date first)
    parsed_expiries.sort(key=lambda x: x[0])
    sorted_expiries = [item[1] for item in parsed_expiries]
    
    return jsonify({"expiries": sorted_expiries, "status": "API connection success"})

@app.route('/api/fetch-chain', methods=['POST'])
def fetch_chain():
    if INSTRUMENT_DF is None:
        return jsonify({"strikes": [], "atm": None})

    data = request.json or {}
    symbol = data.get('symbol', 'NIFTY')
    expiry = data.get('expiry')

    filtered = INSTRUMENT_DF[(INSTRUMENT_DF['name'] == symbol) & (INSTRUMENT_DF['expiry'] == expiry)]
    strikes = sorted(filtered['strike_price'].unique().tolist())

    smart_api, _ = get_smart_api()
    atm = None
    if smart_api and symbol in INDEX_TOKENS:
        try:
            tok_info = INDEX_TOKENS[symbol]
            ltp_resp = smart_api.ltpData(tok_info["exchange"], tok_info["tradingsymbol"], tok_info["token"])
            if ltp_resp and ltp_resp.get('status') and ltp_resp.get('data'):
                spot = float(ltp_resp['data']['ltp'])
                step = tok_info['step']
                atm = round(spot / step) * step
        except Exception:
            pass

    return jsonify({"strikes": strikes, "atm": atm})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    smart_api, conn_msg = get_smart_api()
    if not smart_api:
        return jsonify({"error": conn_msg, "status": conn_msg})

    req_data = request.json or {}
    symbol = req_data.get('symbol', 'NIFTY')
    baskets = req_data.get('baskets', [])
    interval = req_data.get('interval', '15')
    basket_interval = req_data.get('basket_interval', '15')
    selected_expiry = req_data.get('expiry')
    pcr_strike_range = req_data.get('pcr_strike_range')

    idx_info = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])
    
    try:
        ltp_resp = smart_api.ltpData(idx_info["exchange"], idx_info["tradingsymbol"], idx_info["token"])
        if not (ltp_resp and ltp_resp.get('status') and ltp_resp.get('data')):
            return jsonify({"error": "Failed to fetch Index LTP", "status": "LTP Fetch Failed"})
        
        spot_price = float(ltp_resp['data']['ltp'])
    except Exception as e:
        return jsonify({"error": str(e), "status": f"API Error: {str(e)}"})

    from_date, to_date = get_last_trading_day_dates(num_days=5)

    chart_labels, chart_prices = [], []
    index_candles = []
    try:
        hist_params = {
            "exchange": idx_info["exchange"],
            "symboltoken": idx_info["token"],
            "interval": INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE"),
            "fromdate": from_date,
            "todate": to_date
        }
        hist_resp = smart_api.getCandleData(hist_params)
        if hist_resp and hist_resp.get('status') and hist_resp.get('data'):
            index_candles = hist_resp['data']
            chart_labels = [c[0].replace('T', ' ')[:16] for c in index_candles]
            chart_prices = [float(c[4]) for c in index_candles]
    except Exception:
        pass

    indicators, bb_status, macd_status, rsi_status, price_divs, macd_divs, rsi_divs = calculate_chart_indicators(index_candles)

    option_chain_data = []
    if selected_expiry:
        option_chain_data = fetch_option_chain_data(smart_api, symbol, selected_expiry)

    pcr_value = calculate_pcr(option_chain_data, spot_price, idx_info['step'], pcr_strike_range)

    atm_iv_estimate, _ = get_atm_iv_and_leg_greeks(
        smart_api, option_chain_data, spot_price, idx_info['step'], 
        spot_price, 'CE', selected_expiry or "31DEC2026", symbol=symbol
    )
    
    expected_1day_move = round(spot_price * (atm_iv_estimate / 100.0) * math.sqrt(1 / 365.0), 2)
    
    now = datetime.now(IST)
    days_to_expiry = 7.0
    if selected_expiry:
        try:
            exp_dt = datetime.strptime(selected_expiry, "%d%b%Y").date()
            days_to_expiry = max((exp_dt - now.date()).days, 0.5)
        except Exception:
            pass
    expected_expiry_move = round(spot_price * (atm_iv_estimate / 100.0) * math.sqrt(days_to_expiry / 365.0), 2)

    processed_baskets = []
    for basket in baskets:
        basket_impact_duration = int(basket.get('impact_duration', 15))
        
        _, past_spot = get_past_price_within_market_hours(index_candles, basket_impact_duration)
        if past_spot is None:
            past_spot = spot_price
        delta_spot = spot_price - past_spot
        dt_days = basket_impact_duration / 1440.0

        basket_pnl = 0.0
        processed_legs = []
        leg_series_data = []
        master_timestamps = chart_labels or []

        basket_delta_contrib = 0.0
        basket_theta_contrib = 0.0
        basket_vega_contrib = 0.0
        total_basket_val_change = 0.0

        for leg in basket.get('legs', []):
            strike = float(leg['strike'])
            exp_date = leg['expiry']
            opt_type = leg['option_type']
            action = leg['action']
            qty = int(leg.get('qty', 65))
            entry_price = leg.get('entry_price')

            current_ltp = "N/A"
            token = None
            symbol_name = None
            
            if INSTRUMENT_DF is not None:
                match = INSTRUMENT_DF[
                    (INSTRUMENT_DF['name'] == symbol) & 
                    (INSTRUMENT_DF['expiry'] == exp_date) & 
                    (INSTRUMENT_DF['strike_price'] == strike) & 
                    (INSTRUMENT_DF['symbol'].str.endswith(opt_type))
                ]
                if not match.empty:
                    token = str(match.iloc[0]['token'])
                    symbol_name = match.iloc[0]['symbol']

            leg_prices_map = {}
            leg_candles = []
            if token and symbol_name:
                try:
                    opt_ltp_resp = smart_api.ltpData("NFO", symbol_name, token)
                    if opt_ltp_resp and opt_ltp_resp.get('status') and opt_ltp_resp.get('data'):
                        current_ltp = float(opt_ltp_resp['data']['ltp'])
                except Exception:
                    pass

                try:
                    leg_hist_params = {
                        "exchange": "NFO",
                        "symboltoken": token,
                        "interval": INTERVAL_MAP.get(basket_interval, "FIFTEEN_MINUTE"),
                        "fromdate": from_date,
                        "todate": to_date
                    }
                    leg_hist_resp = smart_api.getCandleData(leg_hist_params)
                    if leg_hist_resp and leg_hist_resp.get('status') and leg_hist_resp.get('data'):
                        leg_candles = leg_hist_resp['data']
                        for c in leg_candles:
                            ts = c[0].replace('T', ' ')[:16]
                            leg_prices_map[ts] = float(c[4])
                except Exception:
                    pass

            if entry_price is None or entry_price == 0:
                entry_price = current_ltp if isinstance(current_ltp, (int, float)) else 0.0

            leg_pnl = 0.0
            if isinstance(current_ltp, (int, float)) and isinstance(entry_price, (int, float)):
                if action == 'BUY':
                    leg_pnl = (current_ltp - entry_price) * qty
                else:
                    leg_pnl = (entry_price - current_ltp) * qty
                basket_pnl += leg_pnl

            _, greeks = get_atm_iv_and_leg_greeks(
                smart_api, option_chain_data, spot_price, idx_info['step'], 
                strike, opt_type, exp_date, symbol=symbol
            )
            
            decay_till_date = round((entry_price - (current_ltp if isinstance(current_ltp, (int, float)) else entry_price)) * qty, 2)

            latest_lp, past_lp = get_past_price_within_market_hours(leg_candles, basket_impact_duration)
            if latest_lp is not None and past_lp is not None:
                current_ltp = latest_lp
                past_leg_price = past_lp
            else:
                past_leg_price = current_ltp if isinstance(current_ltp, (int, float)) else entry_price

            direction_mult = 1 if action == 'BUY' else -1
            actual_leg_val_change = 0.0
            if isinstance(current_ltp, (int, float)) and isinstance(past_leg_price, (int, float)):
                actual_leg_val_change = (current_ltp - past_leg_price) * qty * direction_mult

            c_delta = greeks['delta'] * delta_spot * qty * direction_mult
            c_theta = greeks['theta'] * dt_days * qty * direction_mult
            c_vega = actual_leg_val_change - (c_delta + c_theta)

            if abs(actual_leg_val_change) > 1e-5:
                leg_delta_pct = round((c_delta / actual_leg_val_change) * 100.0, 1)
                leg_theta_pct = round((c_theta / actual_leg_val_change) * 100.0, 1)
                leg_vega_pct = round((c_vega / actual_leg_val_change) * 100.0, 1)
            else:
                leg_delta_pct, leg_theta_pct, leg_vega_pct = 0.0, 0.0, 0.0

            basket_delta_contrib += c_delta
            basket_theta_contrib += c_theta
            basket_vega_contrib += c_vega
            total_basket_val_change += actual_leg_val_change

            processed_legs.append({
                "strike": strike,
                "expiry": exp_date,
                "option_type": opt_type,
                "action": action,
                "qty": qty,
                "entry_price": entry_price,
                "current_premium": current_ltp,
                "leg_pnl": round(leg_pnl, 2) if isinstance(leg_pnl, float) else leg_pnl,
                "greeks": greeks,
                "theta_decay_till_date": decay_till_date,
                "impact": {
                    "delta_pct": leg_delta_pct,
                    "theta_pct": leg_theta_pct,
                    "vega_pct": leg_vega_pct
                }
            })

            aligned_series = [leg_prices_map.get(ts, None) for ts in master_timestamps]
            leg_series_data.append({
                "label": f"{strike} {opt_type} ({action})",
                "prices": aligned_series
            })

        strategy_type = identify_strategy(basket.get('legs', []))
        net_greeks = calculate_basket_net_greeks(processed_legs)
        max_prof, max_lss = calculate_max_profit_loss(basket.get('legs', []))

        if abs(total_basket_val_change) > 1e-5:
            b_delta_pct = round((basket_delta_contrib / total_basket_val_change) * 100.0, 1)
            b_theta_pct = round((basket_theta_contrib / total_basket_val_change) * 100.0, 1)
            b_vega_pct = round((basket_vega_contrib / total_basket_val_change) * 100.0, 1)
        else:
            b_delta_pct, b_theta_pct, b_vega_pct = 0.0, 0.0, 0.0

        basket_impact_metrics = {
            "delta_pct": b_delta_pct,
            "theta_pct": b_theta_pct,
            "vega_pct": b_vega_pct
        }

        processed_baskets.append({
            "id": basket['id'],
            "name": basket['name'],
            "strategy_type": strategy_type,
            "impact_duration": basket_impact_duration,
            "basket_pnl": round(basket_pnl, 2),
            "max_profit": max_prof,
            "max_loss": max_lss,
            "net_greeks": net_greeks,
            "basket_impact": basket_impact_metrics,
            "total_decay_till_date": round(basket_pnl, 2),
            "legs": processed_legs,
            "legs_historical": {
                "labels": master_timestamps,
                "leg_series": leg_series_data
            }
        })

    return jsonify({
        "status": conn_msg,
        "underlying_price": spot_price,
        "current_iv": atm_iv_estimate,
        "expected_1day_move": expected_1day_move,
        "expected_expiry_move": expected_expiry_move,
        "pcr_value": pcr_value,
        "is_market_open": is_market_open(),
        "chart_labels": chart_labels,
        "chart_prices": chart_prices,
        "indicators": indicators,
        "bb_status": bb_status,
        "macd_status": macd_status,
        "rsi_status": rsi_status,
        "price_divergences": price_divs,
        "macd_divergences": macd_divs,
        "rsi_divergences": rsi_divs,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
