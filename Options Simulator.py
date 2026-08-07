import os
import json
import math
import pyotp
import threading
import requests
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
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

INDIA_VIX_TOKEN = {"exchange": "NSE", "tradingsymbol": "INDIA VIX", "token": "99926017"}

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

# Memory Caching to Prevent Render 502 Bad Gateway Timeouts
API_CACHE = {}
CACHE_TTL_SECONDS = 10

def get_cached_data(key):
    if key in API_CACHE:
        val, ts = API_CACHE[key]
        if time_module.time() - ts < CACHE_TTL_SECONDS:
            return val
    return None

def set_cached_data(key, val):
    API_CACHE[key] = (val, time_module.time())

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
            keep_cols = ['exch_seg', 'instrumenttype', 'name', 'expiry', 'strike', 'symbol', 'token']
            df = df[[c for c in keep_cols if c in df.columns]]
            df = df[(df['exch_seg'] == 'NFO') & (df['instrumenttype'].isin(['OPTIDX', 'OPTSTK']))].copy()
            df['strike_price'] = pd.to_numeric(df['strike'], errors='coerce') / 100.0
            
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
    cache_key = f"chain_{symbol}_{expiry}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached

    try:
        chain_params = {"name": symbol, "expirydate": expiry}
        res = smart_api.optionChain(chain_params)
        if res and res.get('status') and res.get('data'):
            set_cached_data(cache_key, res['data'])
            return res['data']
    except Exception:
        pass
    return []

def get_nearest_expiry(symbol):
    if INSTRUMENT_DF is None:
        return None
    filtered = INSTRUMENT_DF[INSTRUMENT_DF['name'] == symbol]
    raw_expiries = filtered['expiry'].unique().tolist()
    
    parsed_expiries = []
    now_date = datetime.now(IST).date()

    for exp in raw_expiries:
        try:
            exp_date = datetime.strptime(str(exp), "%d%b%Y").date()
            if exp_date >= now_date:
                parsed_expiries.append((exp_date, str(exp)))
        except Exception:
            pass

    parsed_expiries.sort(key=lambda x: x[0])
    return parsed_expiries[0][1] if parsed_expiries else None

def fetch_vix(smart_api):
    cache_key = "india_vix"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached

    try:
        if smart_api:
            res = smart_api.ltpData(INDIA_VIX_TOKEN["exchange"], INDIA_VIX_TOKEN["tradingsymbol"], INDIA_VIX_TOKEN["token"])
            if res and res.get('status') and res.get('data'):
                vix_val = float(res['data']['ltp'])
                set_cached_data(cache_key, vix_val)
                return vix_val
    except Exception:
        pass
    return 13.50

def calculate_historical_volatility(candles):
    if not candles or len(candles) < 20:
        return 12.50

    try:
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        daily_std = df['log_ret'].std()
        
        annualized_hv = daily_std * np.sqrt(252 * 25) * 100.0
        if np.isnan(annualized_hv) or annualized_hv <= 0:
            return 12.50
        return round(float(annualized_hv), 2)
    except Exception:
        return 12.50

def calculate_pcr(smart_api, symbol, option_chain_data, spot_price, step, expiry=None, strike_range_limit=None):
    if spot_price <= 0:
        return "N/A"

    atm_strike = round(spot_price / step) * step

    if not expiry:
        expiry = get_nearest_expiry(symbol)

    chain = option_chain_data

    if not chain and INSTRUMENT_DF is not None and expiry:
        filtered_df = INSTRUMENT_DF[(INSTRUMENT_DF['name'] == symbol) & (INSTRUMENT_DF['expiry'] == expiry)]
        chain = []
        for _, row in filtered_df.iterrows():
            opt_type = 'CE' if str(row['symbol']).endswith('CE') else 'PE'
            chain.append({
                'strikePrice': row['strike_price'],
                'optionType': opt_type,
                'token': row['token'],
                'symbol': row['symbol'],
                'openInterest': 0
            })

    if not chain:
        return "N/A"

    if strike_range_limit is not None and strike_range_limit > 0:
        lower_bound = atm_strike - (strike_range_limit * step)
        upper_bound = atm_strike + (strike_range_limit * step)
        filtered_chain = []
        for item in chain:
            try:
                sp = float(item.get('strikePrice', 0) or item.get('strikeprice', 0) or 0)
                if lower_bound <= sp <= upper_bound:
                    filtered_chain.append(item)
            except Exception:
                continue
    else:
        filtered_chain = chain

    total_put_oi = 0.0
    total_call_oi = 0.0

    for item in filtered_chain:
        opt_type = str(item.get('optionType', '') or item.get('optiontype', '')).upper()
        
        oi = float(
            item.get('opennterest', 0) or 
            item.get('openInterest', 0) or 
            item.get('openinterest', 0) or 
            item.get('oi', 0) or 0
        )

        if opt_type in ['PE', 'PUT'] or str(item.get('symbol', '')).endswith('PE'):
            total_put_oi += oi
        elif opt_type in ['CE', 'CALL'] or str(item.get('symbol', '')).endswith('CE'):
            total_call_oi += oi

    if total_call_oi == 0:
        if total_put_oi > 0:
            return "> 5.0 (Extreme Bullish)"
        return "N/A"

    pcr_val = total_put_oi / total_call_oi
    return round(pcr_val, 2)

def calculate_chart_indicators(candles):
    if not candles or len(candles) < 26:
        return {}, "BB (20,2) : N/A", "MACD (12,26,9) : N/A", "RSI (14) : N/A"

    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # 1. Volume Weighted Average Price (VWAP)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['tp_v'] = df['typical_price'] * df['volume']
    
    # Calculate intraday cumulative sums reset per session
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['cum_tp_v'] = df.groupby('date')['tp_v'].cumsum()
    df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    
    # Handle zero volume cases cleanly
    df['vwap'] = np.where(df['cum_vol'] > 0, df['cum_tp_v'] / df['cum_vol'], df['close'])

    # 2. Bollinger Bands (20, 2)
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

    # 3. MACD (12, 26, 9)
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

    # 4. RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    curr_rsi = df['rsi'].dropna().iloc[-1] if not df['rsi'].dropna().empty else 50
    if curr_rsi >= 70:
        rsi_status = f"RSI (14) : Overbought ({round(curr_rsi, 1)})"
    elif curr_rsi <= 30:
        rsi_status = f"RSI (14) : Oversold ({round(curr_rsi, 1)})"
    else:
        rsi_status = f"RSI (14) : Neutral ({round(curr_rsi, 1)})"

    def clean_series(series):
        return [None if np.isnan(val) else round(float(val), 2) for val in series]

    indicator_series = {
        "vwap": clean_series(df['vwap']),
        "bb_upper": clean_series(df['bb_upper']),
        "bb_middle": clean_series(df['sma20']),
        "bb_lower": clean_series(df['bb_lower']),
        "macd": clean_series(df['macd']),
        "macd_signal": clean_series(df['macd_signal']),
        "macd_hist": clean_series(df['macd_hist']),
        "rsi": clean_series(df['rsi'])
    }

    return indicator_series, bb_status, macd_status, rsi_status

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

    atm_iv = atm_iv if atm_iv is not None else 13.5
    leg_greeks = calculate_black_scholes_greeks(spot_price, strike, t_years, atm_iv, opt_type)
    return round(atm_iv, 2), leg_greeks

def fetch_leg_market_data(smart_api, token, symbol_name, basket_interval, from_date, to_date):
    ltp_val = None
    leg_candles = []
    
    if not (smart_api and token and symbol_name):
        return ltp_val, leg_candles

    def _fetch_ltp():
        try:
            res = smart_api.ltpData("NFO", symbol_name, token)
            if res and res.get('status') and res.get('data'):
                return float(res['data']['ltp'])
        except Exception:
            pass
        return None

    def _fetch_candles():
        try:
            params = {
                "exchange": "NFO",
                "symboltoken": token,
                "interval": INTERVAL_MAP.get(basket_interval, "FIFTEEN_MINUTE"),
                "fromdate": from_date,
                "todate": to_date
            }
            res = smart_api.getCandleData(params)
            if res and res.get('status') and res.get('data'):
                return res['data']
        except Exception:
            pass
        return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_ltp = executor.submit(_fetch_ltp)
        f_cand = executor.submit(_fetch_candles)
        ltp_val = f_ltp.result()
        leg_candles = f_cand.result()

    return ltp_val, leg_candles

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
    qty = 65

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
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.0.1"></script>
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
            top: 8px;
            right: 5px;
            background: rgba(18, 18, 18, 0.92);
            border: 1px solid #00bcd4;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 10px;
            font-family: monospace;
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.7);
            line-height: 1.4;
            text-align: right;
            max-width: 300px;
        }
        .ribbon-item { display: block; color: #00ff88; }
        .ribbon-vix { color: #ffca28; font-weight: bold; }
        .ribbon-vol { color: #00e5ff; font-weight: bold; }
        .ribbon-rec { color: #ff4081; font-weight: bold; margin-top: 3px; border-top: 1px dashed #444; padding-top: 3px; }

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

        .rsi-wrapper {
            position: relative;
            height: 175px;
            width: 100%;
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
                    <span id="ribbonVix" class="ribbon-vix">VIX: -</span>
                    <span id="ribbonVol" class="ribbon-vol">IV: - | HV: -</span>
                    <span id="ribbonBB" class="ribbon-item">BB (20,2) : -</span>
                    <span id="ribbonMACD" class="ribbon-item">MACD (12,26,9) : -</span>
                    <span id="ribbonRSI" class="ribbon-item">RSI (14) : -</span>
                    <span id="ribbonRec" class="ribbon-rec">Trend Suggestion: -</span>
                </div>

                <div class="header-flex">
                    <h3>Index Price, VWAP & Bollinger Bands</h3>
                    <div style="display: flex; gap: 10px; align-items: center; margin-right: 320px;">
                        <div class="pcr-control-box" title="Number of Strikes +/- from ATM Strike Price">
                            <span>PCR &plusmn; Range:</span>
                            <input type="number" id="pcrStrikeRange" class="pcr-input" placeholder="All" min="1" max="50" value="5">
                            <button class="pcr-refresh-btn" onclick="refreshPcrValue()" title="Recalculate PCR">
                                <span id="pcrSpinner" class="header-spinner" style="display:none;"></span>
                                🔄
                            </button>
                        </div>
                        <div>
                            <label style="display:inline; color:#aaa; font-size:11px; margin-right:3px;">Timeframe:</label>
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

                <div class="subchart-title">MACD (12, 26, 9 EMA) Indicator</div>
                <canvas id="macdChart" height="65"></canvas>

                <div class="subchart-title">RSI (14) Indicator (0-30-70-100 Rescaled)</div>
                <div class="rsi-wrapper">
                    <canvas id="rsiChart"></canvas>
                </div>
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

        const ctx = document.getElementById('mainChart').getContext('2d');
        mainChart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'Index Spot Price', data: [], borderColor: '#00bcd4', borderWidth: 1.5, pointRadius: 0, tension: 0.1, spanGaps: true },
                    { label: 'VWAP', data: [], borderColor: '#ff00ff', borderWidth: 1.5, pointRadius: 0, tension: 0.1, spanGaps: true },
                    { label: 'BB Upper (20,2)', data: [], borderColor: 'rgba(255, 82, 82, 0.8)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, spanGaps: true },
                    { label: 'BB Middle SMA (20)', data: [], borderColor: 'rgba(255, 202, 40, 0.8)', borderWidth: 1, pointRadius: 0, spanGaps: true },
                    { label: 'BB Lower (20,2)', data: [], borderColor: 'rgba(76, 175, 80, 0.8)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, spanGaps: true }
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

        const ctxMACD = document.getElementById('macdChart').getContext('2d');
        macdChart = new Chart(ctxMACD, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    { type: 'line', label: 'MACD (12, 26)', data: [], borderColor: '#2196f3', borderWidth: 1.2, pointRadius: 0, spanGaps: true },
                    { type: 'line', label: 'Signal (9 EMA)', data: [], borderColor: '#ff9800', borderWidth: 1.2, pointRadius: 0, spanGaps: true },
                    { type: 'bar', label: 'MACD Hist', data: [], backgroundColor: 'rgba(0, 188, 212, 0.4)' }
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

        const ctxRSI = document.getElementById('rsiChart').getContext('2d');
        rsiChart = new Chart(ctxRSI, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'RSI (14)', data: [], borderColor: '#e91e63', borderWidth: 1.8, pointRadius: 0, tension: 0.1, spanGaps: true }
                ]
            },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        min: 0, 
                        max: 100, 
                        grid: { color: '#2a2a2a' }, 
                        ticks: { 
                            stepSize: 10,
                            color: '#aaa',
                            callback: function(val) {
                                if ([0, 30, 70, 100].includes(val)) return val;
                                return '';
                            }
                        } 
                    },
                    x: { grid: { color: '#2a2a2a' }, ticks: { color: '#888' } }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line70: {
                                type: 'line',
                                yMin: 70,
                                yMax: 70,
                                borderColor: 'rgba(255, 82, 82, 0.6)',
                                borderWidth: 1,
                                borderDash: [4, 4],
                                label: { display: true, content: 'Overbought (70)', color: '#ff5252', position: 'start', font: { size: 10 } }
                            },
                            line30: {
                                type: 'line',
                                yMin: 30,
                                yMax: 30,
                                borderColor: 'rgba(76, 175, 80, 0.6)',
                                borderWidth: 1,
                                borderDash: [4, 4],
                                label: { display: true, content: 'Oversold (30)', color: '#4caf50', position: 'start', font: { size: 10 } }
                            }
                        }
                    }
                }
            }
        });

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
                        if(idx === 0) opt.selected = true;
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

                document.getElementById('ribbonVix').innerText = "India VIX: " + data.vix_val;
                document.getElementById('ribbonVol').innerText = "IV: " + data.current_iv + "% | HV: " + data.hv_val + "% (" + data.vol_relation + ")";
                document.getElementById('ribbonBB').innerText = data.bb_status;
                document.getElementById('ribbonMACD').innerText = data.macd_status;
                document.getElementById('ribbonRSI').innerText = data.rsi_status;
                document.getElementById('ribbonRec').innerText = "Strategy Suggestion: " + data.trend_recommendation;

                const validBaskets = (data.baskets || []).filter(b => !deletedBasketIds.has(b.id));

                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    macdChart.data.labels = data.chart_labels;
                    rsiChart.data.labels = data.chart_labels;

                    mainChart.data.datasets[0].data = data.chart_prices;
                    if (data.indicators) {
                        mainChart.data.datasets[1].data = data.indicators.vwap || [];
                        mainChart.data.datasets[2].data = data.indicators.bb_upper || [];
                        mainChart.data.datasets[3].data = data.indicators.bb_middle || [];
                        mainChart.data.datasets[4].data = data.indicators.bb_lower || [];

                        macdChart.data.datasets[0].data = data.indicators.macd || [];
                        macdChart.data.datasets[1].data = data.indicators.macd_signal || [];
                        macdChart.data.datasets[2].data = data.indicators.macd_hist || [];

                        rsiChart.data.datasets[0].data = data.indicators.rsi || [];
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

    parsed_expiries = []
    now_date = datetime.now(IST).date()

    for exp in raw_expiries:
        try:
            exp_date = datetime.strptime(str(exp), "%d%b%Y").date()
            if exp_date >= now_date:
                parsed_expiries.append((exp_date, str(exp)))
        except Exception:
            pass

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
    strikes = sorted(filtered['strike_price'].dropna().unique().tolist())

    smart_api, _ = get_smart_api()
    atm = None
    if smart_api and symbol in INDEX_TOKENS:
        try:
            tok_info = INDEX_TOKENS[symbol]
            
            cache_key = f"ltp_{symbol}"
            spot = get_cached_data(cache_key)
            if spot is None:
                ltp_resp = smart_api.ltpData(tok_info["exchange"], tok_info["tradingsymbol"], tok_info["token"])
                if ltp_resp and ltp_resp.get('status') and ltp_resp.get('data'):
                    spot = float(ltp_resp['data']['ltp'])
                    set_cached_data(cache_key, spot)

            if spot:
                step = tok_info['step']
                atm = round(spot / step) * step
        except Exception:
            pass

    return jsonify({"strikes": strikes, "atm": atm})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    smart_api, status_msg = get_smart_api()
    if not smart_api:
        return jsonify({"error": status_msg})

    req = request.json or {}
    symbol = req.get('symbol', 'NIFTY')
    interval = req.get('interval', '15')
    basket_interval = req.get('basket_interval', '15')
    selected_expiry = req.get('expiry')
    baskets = req.get('baskets', [])
    pcr_strike_range = req.get('pcr_strike_range')

    tok_info = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])
    step = tok_info['step']

    # 1. Fetch Index Spot
    cache_key = f"ltp_{symbol}"
    spot_price = get_cached_data(cache_key)
    if spot_price is None:
        try:
            ltp_resp = smart_api.ltpData(tok_info["exchange"], tok_info["tradingsymbol"], tok_info["token"])
            if ltp_resp and ltp_resp.get('status') and ltp_resp.get('data'):
                spot_price = float(ltp_resp['data']['ltp'])
                set_cached_data(cache_key, spot_price)
        except Exception:
            pass

    if not spot_price:
        return jsonify({"error": "Unable to fetch index price"})

    # 2. Fetch Index Chart Candle Data
    from_date, to_date = get_last_trading_day_dates(num_days=5)
    
    candle_cache_key = f"candles_{symbol}_{interval}"
    candles = get_cached_data(candle_cache_key)
    if candles is None:
        try:
            candle_params = {
                "exchange": tok_info["exchange"],
                "symboltoken": tok_info["token"],
                "interval": INTERVAL_MAP.get(interval, "FIFTEEN_MINUTE"),
                "fromdate": from_date,
                "todate": to_date
            }
            c_res = smart_api.getCandleData(candle_params)
            if c_res and c_res.get('status') and c_res.get('data'):
                candles = c_res['data']
                set_cached_data(candle_cache_key, candles)
        except Exception:
            candles = []

    chart_labels = [c[0].split('T')[1][:5] for c in candles] if candles else []
    chart_prices = [float(c[4]) for c in candles] if candles else []

    # 3. Calculate Technical Indicators, HV & VIX
    indicator_series, bb_status, macd_status, rsi_status = calculate_chart_indicators(candles)
    hv_val = calculate_historical_volatility(candles)
    vix_val = fetch_vix(smart_api)

    # 4. Fetch Option Chain & PCR
    expiry_str = selected_expiry or get_nearest_expiry(symbol)
    chain_data = fetch_option_chain_data(smart_api, symbol, expiry_str) if expiry_str else []
    
    pcr_val = calculate_pcr(
        smart_api=smart_api, 
        symbol=symbol, 
        option_chain_data=chain_data, 
        spot_price=spot_price, 
        step=step, 
        expiry=expiry_str, 
        strike_range_limit=pcr_strike_range
    )

    current_iv, sample_greeks = get_atm_iv_and_leg_greeks(
        smart_api=smart_api,
        option_chain=chain_data,
        spot_price=spot_price,
        step=step,
        strike=spot_price,
        opt_type='CE',
        expiry_str=expiry_str or '01JAN2026',
        symbol=symbol
    )

    vol_relation = "IV > HV" if current_iv >= hv_val else "IV < HV"
    if current_iv > hv_val:
        trend_recommendation = "IV High (Premiums Rich) -> Prefer Selling Strategies (Credit Spreads, Condors)"
    else:
        trend_recommendation = "IV Low (Premiums Discounted) -> Prefer Buying Strategies (Debit Spreads, Straddles)"

    expected_1day_move = round(spot_price * (current_iv / 100.0) / math.sqrt(365), 2)
    expected_expiry_move = round(spot_price * (current_iv / 100.0) * math.sqrt(7 / 365), 2)

    # 5. Process Active Baskets
    processed_baskets = []
    for basket in baskets:
        basket_id = basket.get('id')
        basket_name = basket.get('name', 'Basket')
        impact_duration = int(basket.get('impact_duration', 15))
        legs = basket.get('legs', [])

        processed_legs = []
        basket_pnl = 0.0
        pnl_valid = True

        hist_labels = chart_labels
        hist_leg_series = []

        for leg in legs:
            l_strike = float(leg.get('strike'))
            l_opt_type = leg.get('option_type')
            l_action = leg.get('action')
            l_expiry = leg.get('expiry')
            l_qty = int(leg.get('qty', 65))
            l_entry = leg.get('entry_price')

            token, symbol_name = None, None
            if INSTRUMENT_DF is not None:
                match = INSTRUMENT_DF[
                    (INSTRUMENT_DF['name'] == symbol) & 
                    (INSTRUMENT_DF['expiry'] == l_expiry) & 
                    (INSTRUMENT_DF['strike_price'] == l_strike) & 
                    (INSTRUMENT_DF['symbol'].str.endswith(l_opt_type))
                ]
                if not match.empty:
                    token = match.iloc[0]['token']
                    symbol_name = match.iloc[0]['symbol']

            ltp, leg_candles = fetch_leg_market_data(smart_api, token, symbol_name, basket_interval, from_date, to_date)
            
            entry_price = l_entry if l_entry is not None else (ltp if ltp else 0.0)

            if ltp is not None and entry_price > 0:
                direction = 1 if l_action == 'BUY' else -1
                leg_pnl = (ltp - entry_price) * l_qty * direction
                basket_pnl += leg_pnl
                leg_pnl_str = round(leg_pnl, 2)
            else:
                pnl_valid = False
                leg_pnl_str = "N/A"

            leg_iv, leg_greeks = get_atm_iv_and_leg_greeks(
                smart_api=smart_api,
                option_chain=chain_data,
                spot_price=spot_price,
                step=step,
                strike=l_strike,
                opt_type=l_opt_type,
                expiry_str=l_expiry,
                symbol=symbol
            )

            delta_impact = abs(leg_greeks['delta'] * spot_price)
            theta_impact = abs(leg_greeks['theta'])
            vega_impact = abs(leg_greeks['vega'])
            tot_impact = delta_impact + theta_impact + vega_impact
            
            leg_impact = {
                "delta_pct": round((delta_impact / tot_impact * 100), 1) if tot_impact > 0 else 0,
                "theta_pct": round((theta_impact / tot_impact * 100), 1) if tot_impact > 0 else 0,
                "vega_pct": round((vega_impact / tot_impact * 100), 1) if tot_impact > 0 else 0
            }

            processed_legs.append({
                "strike": l_strike,
                "option_type": l_opt_type,
                "action": l_action,
                "expiry": l_expiry,
                "qty": l_qty,
                "entry_price": round(entry_price, 2) if entry_price else "N/A",
                "current_premium": round(ltp, 2) if ltp else "N/A",
                "leg_pnl": leg_pnl_str,
                "greeks": leg_greeks,
                "impact": leg_impact,
                "theta_decay_till_date": round(leg_greeks['theta'] * 2, 2)
            })

            if leg_candles:
                c_prices = [float(c[4]) for c in leg_candles]
                hist_leg_series.append({
                    "label": f"{l_strike} {l_opt_type} ({l_action})",
                    "prices": c_prices
                })

        net_greeks = calculate_basket_net_greeks(processed_legs)
        max_prof, max_loss = calculate_max_profit_loss(processed_legs)
        strat_type = identify_strategy(processed_legs)

        b_delta_imp = abs(net_greeks['delta'] * spot_price)
        b_theta_imp = abs(net_greeks['theta'])
        b_vega_imp = abs(net_greeks['vega'])
        b_tot_imp = b_delta_imp + b_theta_imp + b_vega_imp

        basket_impact = {
            "delta_pct": round((b_delta_imp / b_tot_imp * 100), 1) if b_tot_imp > 0 else 0,
            "theta_pct": round((b_theta_imp / b_tot_imp * 100), 1) if b_tot_imp > 0 else 0,
            "vega_pct": round((b_vega_imp / b_tot_imp * 100), 1) if b_tot_imp > 0 else 0
        }

        processed_baskets.append({
            "id": basket_id,
            "name": basket_name,
            "impact_duration": impact_duration,
            "strategy_type": strat_type,
            "basket_pnl": round(basket_pnl, 2) if pnl_valid else "N/A",
            "max_profit": max_prof,
            "max_loss": max_loss,
            "net_greeks": net_greeks,
            "basket_impact": basket_impact,
            "total_decay_till_date": round(net_greeks['theta'] * 2, 2),
            "legs": processed_legs,
            "legs_historical": {
                "labels": hist_labels,
                "leg_series": hist_leg_series
            }
        })

    return jsonify({
        "status": "API connection success",
        "underlying_price": spot_price,
        "current_iv": current_iv,
        "vix_val": vix_val,
        "hv_val": hv_val,
        "vol_relation": vol_relation,
        "trend_recommendation": trend_recommendation,
        "expected_1day_move": expected_1day_move,
        "expected_expiry_move": expected_expiry_move,
        "pcr_value": pcr_val,
        "is_market_open": is_market_open(),
        "bb_status": bb_status,
        "macd_status": macd_status,
        "rsi_status": rsi_status,
        "chart_labels": chart_labels,
        "chart_prices": chart_prices,
        "indicators": indicator_series,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    ensure_scrip_master_loading()
    app.run(host='0.0.0.0', port=5000, debug=True)
