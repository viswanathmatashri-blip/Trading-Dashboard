import os
import math
import json
import sqlite3
import urllib.request
import pyotp
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from dash import Dash, dcc, html, Input, Output, State
from SmartApi import SmartConnect

def get_secret(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val:
        return env_val
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

SCRIP_MASTER_FILE = "/tmp/scrip_master.json"
POSITIONS_DB_FILE = "/tmp/positions.db"
SELECTED_SYMBOL = "NIFTY 50"

STD_PARAMS = {
    "VWAP_SD": 1.5,
    "RSI_PERIOD": 14, "RSI_OVERSOLD": 35, "RSI_OVERBOUGHT": 65,
    "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
    "BB_PERIOD": 20, "BB_STD": 2.0,
    "SL_PCT": 0.002,
    "RR_RATIO": 1.5,
    "LOT_SIZE": 65,
    "STRIKE_STEP": 50,
    "BACKTEST_DAYS": 20,

    # ---- Iron Condor Settings ----
    "IC_TARGET_DELTA": 0.15,
    "IC_PUT_IV_SKEW": 1.12,
    "IC_CALL_IV_SKEW": 0.96,
    "IC_RISK_FREE_RATE": 0.065,
    "IC_WING_STEP_MULTIPLES": 20,
    "IC_MIN_WING": 100,
    "IC_MIN_CREDIT_TO_MAXLOSS": 0.30,
    "IC_SL_FRACTION_OF_MAXLOSS": 0.50,
    "IC_MIN_DTE_FOR_NEW_ENTRY": 1,
    "IC_AVOID_DATES": [],

    # ---- Cost model ----
    "EST_ROUND_TRIP_COST_PER_LOT": 45.0,
    "IC_NUM_LEGS": 4,

    "REFRESH_MS": int(os.environ.get("REFRESH_MS", 30000)),
    "BACKTEST_CACHE_TTL_SEC": int(os.environ.get("BACKTEST_CACHE_TTL_SEC", 900)),
}

MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

# ==================== SCRIP MASTER / TOKEN RESOLUTION ====================

def download_scrip_master():
    need_download = True
    if os.path.exists(SCRIP_MASTER_FILE):
        file_date = datetime.fromtimestamp(os.path.getmtime(SCRIP_MASTER_FILE), tz=IST).date()
        if file_date == now_ist().date():
            need_download = False
    if need_download:
        print("Downloading Scrip Master...")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        urllib.request.urlretrieve(url, SCRIP_MASTER_FILE)
        print("Scrip Master saved.")

def load_scrip_master_df():
    download_scrip_master()
    filtered_rows = []
    
    with open(SCRIP_MASTER_FILE, "r") as f:
        data = json.load(f)
        for row in data:
            name = str(row.get('name', '')).upper()
            symbol = str(row.get('symbol', '')).upper()
            exch = str(row.get('exch_seg', '')).upper()
            
            if name in ["NIFTY 50", "INDIA VIX"] or (name == "NIFTY" and exch == "NFO"):
                filtered_rows.append(row)
                
    return pd.DataFrame(filtered_rows)

def resolve_nifty_token(df_master):
    try:
        matched = df_master[
            (df_master['name'].str.upper() == "NIFTY 50") &
            (df_master['exch_seg'].str.upper().isin(["NSE", "INDICES", "NSE_IND"]))
        ]
        if not matched.empty:
            return str(matched.iloc[0]['token']), matched.iloc[0]['exch_seg']

        matched = df_master[
            (df_master['symbol'].str.contains("NIFTY 50", case=False, na=False)) |
            (df_master['name'].str.contains("NIFTY 50", case=False, na=False))
        ]
        if not matched.empty:
            return str(matched.iloc[0]['token']), matched.iloc[0]['exch_seg']

    except Exception as e:
        print(f"Nifty token resolve error: {e}")

    return "99926000", "NSE"

def resolve_india_vix_token(df_master):
    try:
        matched = df_master[
            (df_master['name'].str.upper() == "INDIA VIX") |
            (df_master['symbol'].str.upper() == "INDIA VIX")
        ]
        if not matched.empty:
            return str(matched.iloc[0]['token']), matched.iloc[0]['exch_seg']
    except Exception as e:
        print(f"VIX token resolve error: {e}")
    return None, None

def build_nifty_option_chain(df_master):
    try:
        opts = df_master[
            (df_master['name'].str.upper() == "NIFTY") &
            (df_master['instrumenttype'].str.upper() == "OPTIDX") &
            (df_master['exch_seg'].str.upper() == "NFO")
        ].copy()
        opts['expiry_dt'] = pd.to_datetime(opts['expiry'], format="%d%b%Y", errors='coerce')
        opts['strike_val'] = pd.to_numeric(opts['strike'], errors='coerce') / 100.0
        today = now_ist().date()
        future_expiries = sorted(opts.loc[opts['expiry_dt'].dt.date >= today, 'expiry_dt'].dt.date.unique())
        if not future_expiries:
            return opts.iloc[0:0], None
        nearest_expiry = future_expiries[0]
        opts = opts[opts['expiry_dt'].dt.date == nearest_expiry]
        return opts, nearest_expiry
    except Exception as e:
        print(f"Option chain build error: {e}")
        return df_master.iloc[0:0], None

def get_option_token(options_df, strike, opt_type):
    try:
        row = options_df[
            (options_df['strike_val'] == strike) &
            (options_df['symbol'].str.upper().str.endswith(opt_type))
        ]
        if not row.empty:
            r = row.iloc[0]
            return str(r['token']), r['symbol'], r['exch_seg']
    except Exception:
        pass
    return None, None, None

def nearest_available_strike(options_df, strike, opt_type, step):
    try:
        side = options_df[options_df['symbol'].str.upper().str.endswith(opt_type)]
        if side.empty:
            return strike
        candidates = side['strike_val'].dropna().unique()
        if len(candidates) == 0:
            return strike
        return float(min(candidates, key=lambda s: abs(s - strike)))
    except Exception:
        return strike

_df_master = load_scrip_master_df()
token, exchange = resolve_nifty_token(_df_master)
vix_token, vix_exchange = resolve_india_vix_token(_df_master)
nifty_options_df, current_expiry = build_nifty_option_chain(_df_master)

# ==================== SMARTAPI SESSION ====================

smart_api = SmartConnect(api_key=API_KEY)
session_active = False

def login_smartapi():
    global session_active
    if not (CLIENT_CODE and PASSWORD and TOTP_SECRET and API_KEY):
        print("Credentials missing -- set SMARTAPI_* environment variables on Render.")
        session_active = False
        return False
    try:
        totp_code = pyotp.TOTP(TOTP_SECRET).now()
        smart_api.generateSession(CLIENT_CODE, PASSWORD, totp_code)
        session_active = True
        print("SmartAPI session established.")
        return True
    except Exception as e:
        print(f"Session generation failed: {e}")
        session_active = False
        return False

login_smartapi()

def api_call(fn, *args, retry_on_auth_fail=True, **kwargs):
    global session_active
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict) and not result.get('status', True) and retry_on_auth_fail:
            msg = str(result.get('message', '')).lower()
            if 'token' in msg or 'session' in msg or 'auth' in msg:
                if login_smartapi():
                    return fn(*args, **kwargs)
        return result
    except Exception as e:
        print(f"API call error ({fn}): {e}")
        return None

def fetch_ltp(exch_seg, tradingsymbol, sym_token):
    if not session_active or not sym_token:
        return None
    resp = api_call(smart_api.ltpData, exch_seg, tradingsymbol, sym_token)
    try:
        if isinstance(resp, dict) and resp.get('status') and resp.get('data'):
            return float(resp['data']['ltp'])
    except Exception:
        pass
    return None

def fetch_ltp_batch(exch_seg, tokens):
    tokens = [str(t) for t in tokens if t]
    if not session_active or not tokens:
        return {}
    try:
        resp = api_call(smart_api.getMarketData, "LTP", {"exchangeTokens": {exch_seg: tokens}})
        out = {}
        if isinstance(resp, dict) and resp.get('status') and resp.get('data', {}).get('fetched'):
            for row in resp['data']['fetched']:
                tok = str(row.get('symbolToken') or row.get('symboltoken') or row.get('token') or "")
                ltp = row.get('ltp')
                if tok and ltp is not None:
                    out[tok] = float(ltp)
        return out
    except Exception as e:
        print(f"Batch LTP fetch failed: {e}")
        return {}

def safe_get_candle_data(params):
    resp = api_call(smart_api.getCandleData, params)
    if isinstance(resp, dict) and resp.get('status') and resp.get('data'):
        return resp['data']
    return None

# ==================== BLACK-SCHOLES & GREEKS ENGINE ====================

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if opt_type == "CE" else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        if opt_type == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if opt_type == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0

def bs_greeks(S, K, T, r, sigma, opt_type, position="BUY"):
    """Calculates Delta, Gamma, and Vega for a given position type."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    # Raw option Greeks (Long Position)
    if opt_type == "CE":
        delta = _norm_cdf(d1)
    else:
        delta = _norm_cdf(d1) - 1.0
        
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * _norm_pdf(d1) * math.sqrt(T)) / 100.0  # Normalized for 1% change in IV

    # Adjust signs based on position side (BUY vs SELL)
    multiplier = 1.0 if position == "BUY" else -1.0
    return {
        "delta": round(delta * multiplier, 3),
        "gamma": round(gamma * multiplier, 5),
        "vega": round(vega * multiplier, 3)
    }

def strike_for_target_delta(S, T, r, sigma, target_abs_delta, opt_type, step):
    lo, hi = S * 0.80, S * 1.20
    for _ in range(60):
        mid = (lo + hi) / 2.0
        d = bs_delta(S, mid, T, r, sigma, opt_type)
        mag = abs(d)
        if opt_type == "CE":
            if mag > target_abs_delta:
                lo = mid
            else:
                hi = mid
        else:
            if mag > target_abs_delta:
                hi = mid
            else:
                lo = mid
    result = (lo + hi) / 2.0
    return round(result / step) * step

# ==================== INDICATORS ====================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    df = df.copy()
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def calculate_adx(df, period=14):
    df = df.copy()
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                          np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                          np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
    tr_smooth = df['TR'].ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1 / period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (df['-DM'].ewm(alpha=1 / period, adjust=False).mean() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    return dx.ewm(alpha=1 / period, adjust=False).mean()

def calculate_indicators(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    if (df['Volume'] == 0).all():
        df['Volume'] = (df['High'] - df['Low']).replace(0, 1)
    df['PV'] = df['Typical_Price'] * df['Volume']
    df['Cum_PV'] = df.groupby('Date')['PV'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cum_Vol']
    df['VWAP_Std'] = df.groupby('Date')['Close'].transform(lambda x: x.rolling(15, min_periods=1).std()).fillna(0)
    df['VWAP_Upper'] = df['VWAP'] + STD_PARAMS['VWAP_SD'] * df['VWAP_Std']
    df['VWAP_Lower'] = df['VWAP'] - STD_PARAMS['VWAP_SD'] * df['VWAP_Std']

    df['RSI'] = calculate_rsi(df['Close'], STD_PARAMS['RSI_PERIOD'])
    df['RSI_SMA'] = df['RSI'].rolling(9).mean()

    ema_fast = df['Close'].ewm(span=STD_PARAMS['MACD_FAST'], adjust=False).mean()
    ema_slow = df['Close'].ewm(span=STD_PARAMS['MACD_SLOW'], adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=STD_PARAMS['MACD_SIGNAL'], adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['BB_Mid'] = df['Close'].rolling(STD_PARAMS['BB_PERIOD']).mean()
    df['BB_Std_Val'] = df['Close'].rolling(STD_PARAMS['BB_PERIOD']).std()
    df['BB_Upper'] = df['BB_Mid'] + STD_PARAMS['BB_STD'] * df['BB_Std_Val']
    df['BB_Lower'] = df['BB_Mid'] - STD_PARAMS['BB_STD'] * df['BB_Std_Val']

    df['ADX'] = calculate_adx(df)
    df['ATR'] = calculate_atr(df)
    return df

def generate_individual_signals(df):
    df = calculate_indicators(df).reset_index(drop=True)
    df['Combined_Sig'] = 0
    df['Regime'] = "SIDEWAYS"

    for i in range(2, len(df)):
        rsi_curr = df.loc[i, 'RSI']
        close_curr = df.loc[i, 'Close']
        close_prev = df.loc[i - 1, 'Close']
        adx_val = df.loc[i, 'ADX']
        vwap_val = df.loc[i, 'VWAP']

        if pd.isna(rsi_curr) or pd.isna(adx_val):
            continue

        if adx_val > 22 and close_curr > vwap_val:
            regime = "BULL"
        elif adx_val > 22 and close_curr < vwap_val:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"
        df.loc[i, 'Regime'] = regime

        if regime == "BULL":
            macd_buy = df.loc[i, 'MACD'] > df.loc[i, 'MACD_Signal'] and df.loc[i - 1, 'MACD'] <= df.loc[i - 1, 'MACD_Signal']
            vwap_bounce = close_prev <= df.loc[i - 1, 'VWAP'] and close_curr > vwap_val
            if (macd_buy or vwap_bounce) and rsi_curr < 68:
                df.loc[i, 'Combined_Sig'] = 1
        elif regime == "BEAR":
            macd_sell = df.loc[i, 'MACD'] < df.loc[i, 'MACD_Signal'] and df.loc[i - 1, 'MACD'] >= df.loc[i - 1, 'MACD_Signal']
            vwap_rejection = close_prev >= df.loc[i - 1, 'VWAP'] and close_curr < vwap_val
            if (macd_sell or vwap_rejection) and rsi_curr > 32:
                df.loc[i, 'Combined_Sig'] = -1
        else:
            bb_buy = close_prev <= df.loc[i - 1, 'BB_Lower'] and close_curr > df.loc[i, 'BB_Lower']
            bb_sell = close_prev >= df.loc[i - 1, 'BB_Upper'] and close_curr < df.loc[i, 'BB_Upper']
            if bb_buy and rsi_curr < STD_PARAMS['RSI_OVERSOLD']:
                df.loc[i, 'Combined_Sig'] = 1
            elif bb_sell and rsi_curr > STD_PARAMS['RSI_OVERBOUGHT']:
                df.loc[i, 'Combined_Sig'] = -1
    return df

def backtest_signal_col(df, col_name, sl_pct=0.002, rr_ratio=1.5):
    trades = []
    in_pos, entry, pos_type, tp, sl = False, 0, 0, 0, 0
    for i in range(len(df)):
        close, high, low = df.iloc[i]['Close'], df.iloc[i]['High'], df.iloc[i]['Low']
        sig = df.iloc[i][col_name]
        if in_pos:
            if pos_type == 1:
                if high >= tp:
                    trades.append(((tp - entry) / entry) * 100); in_pos = False
                elif low <= sl:
                    trades.append(((sl - entry) / entry) * 100); in_pos = False
            elif pos_type == -1:
                if low <= tp:
                    trades.append(((entry - tp) / entry) * 100); in_pos = False
                elif high >= sl:
                    trades.append(((entry - sl) / entry) * 100); in_pos = False
        if not in_pos and sig != 0:
            in_pos, entry, pos_type = True, close, sig
            dist = entry * sl_pct
            if pos_type == 1:
                sl, tp = entry - dist, entry + dist * rr_ratio
            else:
                sl, tp = entry + dist, entry - dist * rr_ratio
    return (sum(trades) if trades else 0.0), len(trades)

# ==================== DATA FETCH ====================

def _get_candles_for_range(from_dt, to_dt, sym_token=None, seg=None):
    sym_token = sym_token or token
    candle_exchange = seg or ("NSE" if exchange in ["NSE", "INDICES", "NSE_IND", None] else exchange)
    params = {
        "exchange": candle_exchange,
        "symboltoken": sym_token,
        "interval": "FIVE_MINUTE",
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
    }
    data = safe_get_candle_data(params)
    if not data:
        return None
    df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return df

def fetch_today_or_previous():
    now = now_ist()
    today_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    df = _get_candles_for_range(today_open, now)
    if df is not None and not df.empty:
        return generate_individual_signals(df), False

    for d in range(1, 7):
        day = now - timedelta(days=d)
        day_open = day.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
        day_close = day.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
        df = _get_candles_for_range(day_open, day_close)
        if df is not None and not df.empty:
            return generate_individual_signals(df), True
    return pd.DataFrame(), False

def fetch_recent_days_raw(n_days, sym_token=None, seg=None):
    now = now_ist()
    frames = []
    d = 0
    while len(frames) < n_days and d < n_days * 3:
        d += 1
        day = now - timedelta(days=d)
        if day.weekday() >= 5:
            continue
        day_open = day.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
        day_close = day.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
        df = _get_candles_for_range(day_open, day_close, sym_token=sym_token, seg=seg)
        if df is not None and not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            frames.append(df)
    frames.reverse()
    return frames

def fetch_recent_days(n_days=5):
    frames = fetch_recent_days_raw(n_days)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('Timestamp').reset_index(drop=True)
    return generate_individual_signals(combined)

# ==================== VOLATILITY-BASED EXPECTED MOVE ====================

def get_days_to_expiry(as_of=None):
    if current_expiry is None:
        return 1
    ref = as_of or now_ist().date()
    delta = (current_expiry - ref).days
    return max(delta, 0) + 1

def compute_expected_move(spot, df_recent, vix_val=None, dte=None):
    dte = dte if dte is not None else get_days_to_expiry()
    if vix_val is None and vix_token:
        vix_val = fetch_ltp(vix_exchange, "INDIA VIX", vix_token)

    if vix_val:
        em = spot * (vix_val / 100.0) * math.sqrt(dte / 365.0)
        return em, vix_val, "VIX"

    if df_recent is not None and not df_recent.empty and 'ATR' in df_recent.columns:
        atr = df_recent['ATR'].dropna().iloc[-1] if df_recent['ATR'].notna().any() else spot * 0.002
        bars_per_day = 75
        daily_move = atr * math.sqrt(bars_per_day)
        em = daily_move * math.sqrt(dte)
        return em, None, "ATR"

    return spot * 0.006 * math.sqrt(dte), None, "STATIC"

# ==================== POSITION PERSISTENCE ====================

def _db():
    conn = sqlite3.connect(POSITIONS_DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ic_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strikes TEXT NOT NULL,
            entry_premiums TEXT NOT NULL,
            entry_credit REAL NOT NULL,
            entry_time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            exit_premiums TEXT,
            exit_time TEXT,
            close_reason TEXT,
            realized_pnl_points REAL,
            realized_pnl_rupees REAL
        )
    """)
    return conn

def get_open_position():
    conn = _db()
    row = conn.execute("SELECT * FROM ic_positions WHERE status='OPEN' ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return None
    cols = ["id", "strikes", "entry_premiums", "entry_credit", "entry_time", "status",
            "exit_premiums", "exit_time", "close_reason", "realized_pnl_points", "realized_pnl_rupees"]
    rec = dict(zip(cols, row))
    rec["strikes"] = json.loads(rec["strikes"])
    rec["entry_premiums"] = json.loads(rec["entry_premiums"])
    return rec

def open_position(strikes_key, entry_premiums, entry_credit):
    conn = _db()
    conn.execute(
        "INSERT INTO ic_positions (strikes, entry_premiums, entry_credit, entry_time, status) VALUES (?,?,?,?,?)",
        (json.dumps(list(strikes_key)), json.dumps(entry_premiums), entry_credit,
         now_ist().strftime("%Y-%m-%d %H:%M:%S"), "OPEN")
    )
    conn.commit()
    conn.close()

def close_position(pos_id, exit_premiums, pnl_points, pnl_rupees, reason):
    conn = _db()
    conn.execute(
        """UPDATE ic_positions SET status='CLOSED', exit_premiums=?, exit_time=?, close_reason=?,
           realized_pnl_points=?, realized_pnl_rupees=? WHERE id=?""",
        (json.dumps(exit_premiums), now_ist().strftime("%Y-%m-%d %H:%M:%S"), reason,
         pnl_points, pnl_rupees, pos_id)
    )
    conn.commit()
    conn.close()

# ==================== IRON CONDOR: DELTA/SKEW STRIKE SELECTION & GREEKS ====================

def round_to_step(value, step):
    return round(value / step) * step

def estimated_round_trip_cost_points():
    total_rupees = STD_PARAMS['EST_ROUND_TRIP_COST_PER_LOT'] * STD_PARAMS['IC_NUM_LEGS']
    return total_rupees / STD_PARAMS['LOT_SIZE']

def is_avoid_date(d):
    return d.isoformat() in set(STD_PARAMS.get("IC_AVOID_DATES", []))

def build_iron_condor(spot, df_recent, vix_val=None):
    step = STD_PARAMS['STRIKE_STEP']
    dte = get_days_to_expiry()
    expected_move, vix_val, em_source = compute_expected_move(spot, df_recent, vix_val, dte=dte)
    T = dte / 365.0
    r = STD_PARAMS['IC_RISK_FREE_RATE']

    base_iv = (vix_val / 100.0) if vix_val else max(expected_move / (spot * math.sqrt(T)), 0.08)
    call_iv = base_iv * STD_PARAMS['IC_CALL_IV_SKEW']
    put_iv = base_iv * STD_PARAMS['IC_PUT_IV_SKEW']

    target_delta = STD_PARAMS['IC_TARGET_DELTA']
    call_short_strike = strike_for_target_delta(spot, T, r, call_iv, target_delta, "CE", step)
    put_short_strike = strike_for_target_delta(spot, T, r, put_iv, target_delta, "PE", step)
    call_short_strike = nearest_available_strike(nifty_options_df, call_short_strike, "CE", step)
    put_short_strike = nearest_available_strike(nifty_options_df, put_short_strike, "PE", step)

    gate_reasons = []
    if dte < STD_PARAMS['IC_MIN_DTE_FOR_NEW_ENTRY']:
        gate_reasons.append(f"DTE {dte} below minimum {STD_PARAMS['IC_MIN_DTE_FOR_NEW_ENTRY']} -- gamma/pin risk too high")
    if is_avoid_date(now_ist().date()):
        gate_reasons.append("Today is on event-avoid list")

    tok_cs, tsym_cs, exch_cs = get_option_token(nifty_options_df, call_short_strike, "CE")
    tok_ps, tsym_ps, exch_ps = get_option_token(nifty_options_df, put_short_strike, "PE")
    call_short_prem = fetch_ltp(exch_cs, tsym_cs, tok_cs) if tok_cs else None
    put_short_prem = fetch_ltp(exch_ps, tsym_ps, tok_ps) if tok_ps else None

    def resolve_wing_candidates(short_strike, opt_type):
        candidates = []
        for mult in range(1, STD_PARAMS['IC_WING_STEP_MULTIPLES'] + 1):
            wing = max(STD_PARAMS['IC_MIN_WING'], mult * step)
            hedge_strike = short_strike + wing if opt_type == "CE" else short_strike - wing
            hedge_strike = nearest_available_strike(nifty_options_df, hedge_strike, opt_type, step)
            tok_h, tsym_h, exch_h = get_option_token(nifty_options_df, hedge_strike, opt_type)
            if tok_h:
                candidates.append((wing, hedge_strike, tok_h, exch_h))
        return candidates

    def size_wing(short_strike, opt_type, short_prem, candidates):
        if short_prem is None or not candidates:
            return max(STD_PARAMS['IC_MIN_WING'], round_to_step(0.35 * expected_move, step)), None, None, False
        exch_h = candidates[0][3]
        ltp_map = fetch_ltp_batch(exch_h, [c[2] for c in candidates])
        last = candidates[-1]
        for wing, hedge_strike, tok_h, _ in candidates:
            hedge_prem = ltp_map.get(str(tok_h))
            if hedge_prem is None:
                continue
            credit = short_prem - hedge_prem
            max_loss = wing - credit
            if max_loss <= 0:
                continue
            if credit / max_loss >= STD_PARAMS['IC_MIN_CREDIT_TO_MAXLOSS']:
                return wing, hedge_strike, hedge_prem, True
        return last[0], last[1], ltp_map.get(str(last[2])), False

    call_candidates = resolve_wing_candidates(call_short_strike, "CE")
    put_candidates = resolve_wing_candidates(put_short_strike, "PE")
    call_wing, call_hedge_strike, call_hedge_prem, call_meets_rr = size_wing(call_short_strike, "CE", call_short_prem, call_candidates)
    put_wing, put_hedge_strike, put_hedge_prem, put_meets_rr = size_wing(put_short_strike, "PE", put_short_prem, put_candidates)

    premiums = {
        "call_short": call_short_prem, "call_hedge": call_hedge_prem,
        "put_short": put_short_prem, "put_hedge": put_hedge_prem,
    }
    data_is_live = all(v is not None for v in premiums.values())
    cost_pts = estimated_round_trip_cost_points()

    # ---- CALCULATE GREEKS FOR THE 4 LEGS ----
    greeks = {
        "buy_put": bs_greeks(spot, put_hedge_strike, T, r, put_iv, "PE", "BUY"),
        "buy_call": bs_greeks(spot, call_hedge_strike, T, r, call_iv, "CE", "BUY"),
        "sell_put": bs_greeks(spot, put_short_strike, T, r, put_iv, "PE", "SELL"),
        "sell_call": bs_greeks(spot, call_short_strike, T, r, call_iv, "CE", "SELL"),
    }

    call_credit = (premiums['call_short'] - premiums['call_hedge']) if data_is_live else None
    put_credit = (premiums['put_short'] - premiums['put_hedge']) if data_is_live else None
    total_credit_gross = (call_credit + put_credit) if data_is_live else None
    total_credit_net = (total_credit_gross - cost_pts) if data_is_live else None

    result = {
        "expected_move": expected_move, "vix": vix_val, "em_source": em_source,
        "call_iv": call_iv, "put_iv": put_iv, "target_delta": target_delta,
        "call_wing": call_wing, "put_wing": put_wing,
        "expiry": current_expiry, "days_to_expiry": dte,
        "call_short_strike": call_short_strike, "call_hedge_strike": call_hedge_strike,
        "put_short_strike": put_short_strike, "put_hedge_strike": put_hedge_strike,
        "premiums": premiums, "data_is_live": data_is_live,
        "call_credit": call_credit, "put_credit": put_credit,
        "total_credit_gross": total_credit_gross, "total_credit_net": total_credit_net,
        "cost_pts": cost_pts,
        "call_meets_rr": call_meets_rr, "put_meets_rr": put_meets_rr,
        "gate_reasons": gate_reasons, "entry_allowed": (len(gate_reasons) == 0),
        "greeks": greeks
    }

    if not data_is_live:
        result.update({
            "call_max_loss": None, "put_max_loss": None,
            "call_sl_points": None, "put_sl_points": None,
            "call_breakeven": None, "put_breakeven": None,
        })
        return result

    call_max_loss = call_wing - call_credit
    put_max_loss = put_wing - put_credit
    call_sl_points = STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS'] * call_max_loss
    put_sl_points = STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS'] * put_max_loss

    call_breakeven = call_short_strike + call_credit
    put_breakeven = put_short_strike - put_credit

    result.update({
        "call_max_loss": call_max_loss, "put_max_loss": put_max_loss,
        "call_sl_points": call_sl_points, "put_sl_points": put_sl_points,
        "call_breakeven": call_breakeven, "put_breakeven": put_breakeven,
    })
    return result

def compute_ic_unrealized_pnl(ic):
    if not ic['data_is_live']:
        return {"pnl_points": None, "pnl_rupees": None, "status": "NO_LIVE_DATA", "just_opened": False}

    strikes_key = (ic['call_short_strike'], ic['call_hedge_strike'], ic['put_short_strike'], ic['put_hedge_strike'])
    open_pos = get_open_position()

    if open_pos is None or tuple(open_pos['strikes']) != strikes_key:
        if open_pos is not None:
            cur = ic['premiums']
            entry_prem = open_pos['entry_premiums']
            pnl_pts = (
                (entry_prem['call_short'] - cur['call_short']) + (cur['call_hedge'] - entry_prem['call_hedge']) +
                (entry_prem['put_short'] - cur['put_short']) + (cur['put_hedge'] - entry_prem['put_hedge'])
            ) - ic['cost_pts']
            close_position(open_pos['id'], cur, pnl_pts, pnl_pts * STD_PARAMS['LOT_SIZE'], "REBALANCED")
        if not ic['entry_allowed']:
            return {"pnl_points": None, "pnl_rupees": None, "status": "ENTRY_BLOCKED",
                    "just_opened": False, "block_reasons": ic['gate_reasons']}
        open_position(strikes_key, ic['premiums'], ic['total_credit_net'])
        return {"pnl_points": 0.0, "pnl_rupees": 0.0, "status": "JUST_OPENED", "just_opened": True}

    entry_prem = open_pos['entry_premiums']
    cur = ic['premiums']
    pnl_points = (
        (entry_prem['call_short'] - cur['call_short']) + (cur['call_hedge'] - entry_prem['call_hedge']) +
        (entry_prem['put_short'] - cur['put_short']) + (cur['put_hedge'] - entry_prem['put_hedge'])
    ) - ic['cost_pts']
    pnl_rupees = pnl_points * STD_PARAMS['LOT_SIZE']

    call_leg_loss = max(0.0, (cur['call_short'] - entry_prem['call_short']) - (cur['call_hedge'] - entry_prem['call_hedge']))
    put_leg_loss = max(0.0, (cur['put_short'] - entry_prem['put_short']) - (cur['put_hedge'] - entry_prem['put_hedge']))

    sl_hit, sl_side = False, None
    if ic['call_sl_points'] is not None and call_leg_loss >= ic['call_sl_points']:
        sl_hit, sl_side = True, "CALL"
    elif ic['put_sl_points'] is not None and put_leg_loss >= ic['put_sl_points']:
        sl_hit, sl_side = True, "PUT"

    if sl_hit:
        close_position(open_pos['id'], cur, pnl_points, pnl_rupees, f"SL_HIT_{sl_side}")
        return {"pnl_points": pnl_points, "pnl_rupees": pnl_rupees, "status": f"SL_HIT_{sl_side}", "just_opened": False}

    return {"pnl_points": pnl_points, "pnl_rupees": pnl_rupees, "status": "OPEN", "just_opened": False}

def estimate_delta(spot, strike, expected_move, opt_type):
    if expected_move <= 0:
        expected_move = spot * 0.005
    z = (spot - strike) / expected_move
    call_delta = 1 / (1 + math.exp(-1.7 * z))
    if opt_type == "CE":
        return min(max(call_delta, 0.03), 0.97)
    else:
        return min(max(1 - call_delta, 0.03), 0.97)

def build_atm_directional(spot, df, expected_move):
    atm_strike = round_to_step(spot, STD_PARAMS['STRIKE_STEP'])
    last_signal = df.iloc[-1]['Combined_Sig'] if not df.empty else 0

    legs = {}
    for opt_type in ("CE", "PE"):
        tok, tsym, exch = get_option_token(nifty_options_df, atm_strike, opt_type)
        premium = fetch_ltp(exch, tsym, tok) if tok else None
        delta = estimate_delta(spot, atm_strike, expected_move, opt_type)
        spot_sl_dist = spot * STD_PARAMS['SL_PCT']
        spot_tp_dist = spot_sl_dist * STD_PARAMS['RR_RATIO']
        opt_sl_dist = spot_sl_dist * delta
        opt_tp_dist = spot_tp_dist * delta
        if premium is not None:
            sl_prem = round(max(premium - opt_sl_dist, 0.05), 2)
            tp_prem = round(premium + opt_tp_dist, 2)
        else:
            sl_prem = tp_prem = None
        legs[opt_type] = {
            "strike": atm_strike, "premium": premium, "delta": round(delta, 2),
            "sl_premium": sl_prem, "tp_premium": tp_prem,
        }
    return atm_strike, last_signal, legs

# ==================== IRON CONDOR BACKTEST ====================

def backtest_iron_condor(n_days=None, day_frames=None, vix_frames=None):
    n_days = n_days or STD_PARAMS['BACKTEST_DAYS']
    if day_frames is None:
        day_frames = fetch_recent_days_raw(n_days)
    if vix_frames is None:
        vix_frames = fetch_recent_days_raw(n_days, sym_token=vix_token, seg=vix_exchange) if vix_token else []

    vix_by_date = {}
    for vf in vix_frames:
        d = vf['Timestamp'].dt.date.iloc[-1]
        vix_by_date[d] = float(vf['Close'].iloc[-1])

    step = STD_PARAMS['STRIKE_STEP']
    r = STD_PARAMS['IC_RISK_FREE_RATE']
    target_delta = STD_PARAMS['IC_TARGET_DELTA']
    cost_pts = estimated_round_trip_cost_points()

    trades = []
    for day_df in day_frames:
        if day_df.empty or len(day_df) < 5:
            continue
        day_date = day_df['Timestamp'].dt.date.iloc[0]
        spot_open = float(day_df['Close'].iloc[0])
        spot_close = float(day_df['Close'].iloc[-1])
        spot_high = float(day_df['High'].max())
        spot_low = float(day_df['Low'].min())

        dte = 5
        T = dte / 365.0
        vix_val = vix_by_date.get(day_date)
        base_iv = (vix_val / 100.0) if vix_val else 0.13
        call_iv = base_iv * STD_PARAMS['IC_CALL_IV_SKEW']
        put_iv = base_iv * STD_PARAMS['IC_PUT_IV_SKEW']

        call_short_k = strike_for_target_delta(spot_open, T, r, call_iv, target_delta, "CE", step)
        put_short_k = strike_for_target_delta(spot_open, T, r, put_iv, target_delta, "PE", step)
        wing = max(STD_PARAMS['IC_MIN_WING'], round_to_step(0.35 * spot_open * base_iv * math.sqrt(T), step))
        call_hedge_k = call_short_k + wing
        put_hedge_k = put_short_k - wing

        call_short_p0 = bs_price(spot_open, call_short_k, T, r, call_iv, "CE")
        call_hedge_p0 = bs_price(spot_open, call_hedge_k, T, r, call_iv, "CE")
        put_short_p0 = bs_price(spot_open, put_short_k, T, r, put_iv, "PE")
        put_hedge_p0 = bs_price(spot_open, put_hedge_k, T, r, put_iv, "PE")
        credit0 = (call_short_p0 - call_hedge_p0) + (put_short_p0 - put_hedge_p0) - cost_pts

        T_end = max(T - (1.0 / 365.0), 1e-4)
        call_short_p1 = bs_price(spot_high, call_short_k, T_end, r, call_iv, "CE")
        call_hedge_p1 = bs_price(spot_high, call_hedge_k, T_end, r, call_iv, "CE")
        put_short_p1 = bs_price(spot_low, put_short_k, T_end, r, put_iv, "PE")
        put_hedge_p1 = bs_price(spot_low, put_hedge_k, T_end, r, put_iv, "PE")

        call_leg_loss = max(0.0, (call_short_p1 - call_short_p0) - (call_hedge_p1 - call_hedge_p0))
        put_leg_loss = max(0.0, (put_short_p1 - put_short_p0) - (put_hedge_p1 - put_hedge_p0))
        call_max_loss = wing - (call_short_p0 - call_hedge_p0)
        put_max_loss = wing - (put_short_p0 - put_hedge_p0)
        call_sl = STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS'] * call_max_loss
        put_sl = STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS'] * put_max_loss

        if call_leg_loss >= call_sl or put_leg_loss >= put_sl:
            realized_loss = max(call_leg_loss, put_leg_loss)
            pnl = credit0 - realized_loss
        else:
            call_short_pc = bs_price(spot_close, call_short_k, T_end, r, call_iv, "CE")
            call_hedge_pc = bs_price(spot_close, call_hedge_k, T_end, r, call_iv, "CE")
            put_short_pc = bs_price(spot_close, put_short_k, T_end, r, put_iv, "PE")
            put_hedge_pc = bs_price(spot_close, put_hedge_k, T_end, r, put_iv, "PE")
            creditc = (call_short_pc - call_hedge_pc) + (put_short_pc - put_hedge_pc)
            pnl = credit0 - creditc

        trades.append(pnl)

    if not trades:
        return {"trades": 0, "win_rate": None, "total_pnl_pts": None, "avg_pnl_pts": None, "max_drawdown_pts": None}

    wins = sum(1 for t in trades if t > 0)
    cum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return {
        "trades": len(trades),
        "win_rate": 100.0 * wins / len(trades),
        "total_pnl_pts": float(sum(trades)),
        "avg_pnl_pts": float(np.mean(trades)),
        "max_drawdown_pts": float(drawdown.max()) if len(drawdown) else 0.0,
        "note": "Modeled via Black-Scholes on historical spot + VIX -- not real historical fills.",
    }

# ==================== CACHED BACKTEST LAYER ====================

_backtest_cache = {"computed_at": None, "directional_pnl": 0.0, "directional_trades": 0, "ic_result": None}

def get_cached_backtests():
    now = now_ist()
    ttl = STD_PARAMS['BACKTEST_CACHE_TTL_SEC']
    stale = (
        _backtest_cache["computed_at"] is None
        or (now - _backtest_cache["computed_at"]).total_seconds() > ttl
    )
    if not stale:
        return _backtest_cache

    n_days = STD_PARAMS['BACKTEST_DAYS']
    try:
        day_frames = fetch_recent_days_raw(n_days)
        vix_frames = fetch_recent_days_raw(n_days, sym_token=vix_token, seg=vix_exchange) if vix_token else []

        if day_frames:
            combined = pd.concat(day_frames, ignore_index=True).sort_values('Timestamp').reset_index(drop=True)
            combined = generate_individual_signals(combined)
            comb_pnl, comb_t = backtest_signal_col(combined, 'Combined_Sig', STD_PARAMS['SL_PCT'], STD_PARAMS['RR_RATIO'])
        else:
            comb_pnl, comb_t = 0.0, 0

        ic_result = backtest_iron_condor(n_days, day_frames=day_frames, vix_frames=vix_frames)

        _backtest_cache.update({
            "computed_at": now,
            "directional_pnl": comb_pnl,
            "directional_trades": comb_t,
            "ic_result": ic_result,
        })
    except Exception as e:
        print(f"Backtest cache refresh failed, keeping stale values: {e}")
        if _backtest_cache["computed_at"] is None:
            _backtest_cache.update({"computed_at": now, "directional_pnl": 0.0, "directional_trades": 0, "ic_result": None})

    return _backtest_cache

# ==================== MARKET INSIGHTS ====================

def _find_local_extrema(values, order=3):
    n = len(values)
    highs, lows = [], []
    for i in range(order, n - order):
        window = values[i - order:i + order + 1]
        if values[i] == max(window):
            highs.append(i)
        if values[i] == min(window):
            lows.append(i)
    return highs, lows

def detect_rsi_divergence(df, lookback=50, order=3):
    sub = df.tail(lookback).reset_index(drop=True)
    if len(sub) < order * 2 + 5 or sub['RSI'].isna().sum() > len(sub) * 0.5:
        return "Not enough data yet"

    closes = sub['Close'].values
    highs_idx, lows_idx = _find_local_extrema(closes, order)
    messages = []

    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        p1, p2 = sub['Close'].iloc[i1], sub['Close'].iloc[i2]
        r1, r2 = sub['RSI'].iloc[i1], sub['RSI'].iloc[i2]
        if pd.notna(r1) and pd.notna(r2) and p2 > p1 and r2 < r1:
            messages.append("BEARISH divergence (price higher-high, RSI lower-high)")

    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        p1, p2 = sub['Close'].iloc[i1], sub['Close'].iloc[i2]
        r1, r2 = sub['RSI'].iloc[i1], sub['RSI'].iloc[i2]
        if pd.notna(r1) and pd.notna(r2) and p2 < p1 and r2 > r1:
            messages.append("BULLISH divergence (price lower-low, RSI higher-low)")

    return "; ".join(messages) if messages else "No clear divergence"

def compute_market_insights(df, vix_val):
    latest = df.iloc[-1]
    price = latest['Close']
    regime = latest['Regime']
    adx = latest['ADX']

    vwap = latest['VWAP']
    vwap_dist_pct = ((price - vwap) / vwap * 100) if vwap else 0
    vwap_bias = "ABOVE VWAP (bullish bias)" if price > vwap else "BELOW VWAP (bearish bias)" if price < vwap else "AT VWAP"

    rsi = latest['RSI']
    if pd.isna(rsi):
        rsi_state = "N/A"
    elif rsi >= STD_PARAMS['RSI_OVERBOUGHT']:
        rsi_state = "OVERBOUGHT"
    elif rsi <= STD_PARAMS['RSI_OVERSOLD']:
        rsi_state = "OVERSOLD"
    else:
        rsi_state = "NEUTRAL"
    divergence = detect_rsi_divergence(df)

    bb_upper, bb_lower, bb_mid = latest['BB_Upper'], latest['BB_Lower'], latest['BB_Mid']
    df = df.copy()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    bb_width = df['BB_Width'].iloc[-1]
    hist_widths = df['BB_Width'].dropna().tail(40)
    pct_rank = (hist_widths <= bb_width).mean() * 100 if len(hist_widths) >= 5 else 50

    if pd.isna(bb_width):
        bb_state = "N/A"
    elif pct_rank <= 20:
        bb_state = "SQUEEZE FORMING (volatility contracting -- breakout building)"
    elif pct_rank >= 80:
        bb_state = "BANDS EXPANDED (volatility high / trending move underway)"
    else:
        bb_state = "NORMAL WIDTH"

    if pd.isna(bb_upper) or pd.isna(bb_lower):
        bb_position = "N/A"
    elif price >= bb_upper:
        bb_position = "AT/ABOVE UPPER BAND (walking the band -- overextended or strong trend)"
    elif price <= bb_lower:
        bb_position = "AT/BELOW LOWER BAND (walking the band -- overextended or strong trend)"
    else:
        pos_in_band = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        bb_position = f"{pos_in_band * 100:.0f}% up the band range (mean-reverting zone)"

    today = latest['Date'] if 'Date' in df.columns else None
    today_vol = df[df['Date'] == today]['Volume'].sum() if today is not None else df['Volume'].sum()
    recent_avg = df['Volume'].tail(6).mean()
    prior_avg = df['Volume'].tail(30).head(24).mean() if len(df) >= 30 else recent_avg
    if prior_avg and recent_avg > prior_avg * 1.15:
        vol_trend = "RISING (participation increasing)"
    elif prior_avg and recent_avg < prior_avg * 0.85:
        vol_trend = "FALLING (participation cooling)"
    else:
        vol_trend = "STEADY"

    if vix_val is None:
        vix_state = "unavailable"
    elif vix_val < 12:
        vix_state = "LOW (complacent, range-bound bias)"
    elif vix_val < 16:
        vix_state = "MODERATE"
    elif vix_val < 20:
        vix_state = "ELEVATED (caution, bigger swings likely)"
    else:
        vix_state = "HIGH (fear regime, expect large moves)"

    return {
        "price": price, "regime": regime, "adx": adx,
        "vwap": vwap, "vwap_dist_pct": vwap_dist_pct, "vwap_bias": vwap_bias,
        "rsi": rsi, "rsi_state": rsi_state, "divergence": divergence,
        "bb_state": bb_state, "bb_position": bb_position, "bb_width_pct": bb_width * 100 if pd.notna(bb_width) else None,
        "today_vol": today_vol, "vol_trend": vol_trend,
        "vix_val": vix_val, "vix_state": vix_state,
    }

def fmt(v, prefix="", suffix="", decimals=2):
    if v is None:
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def generate_market_insights_card(df, vix_val):
    if df.empty:
        return html.Div()
    ins = compute_market_insights(df, vix_val)
    regime_color = {"BULL": "#00e676", "BEAR": "#ff1744", "SIDEWAYS": "#ffea00"}.get(ins['regime'], "#aaa")

    def row(label, value, color="#ccc"):
        return html.Div([html.Span(f"{label}: ", style={'color': '#888'}), html.B(value, style={'color': color})],
                         style={'fontSize': '12px', 'marginBottom': '4px'})

    return html.Div(style={'backgroundColor': '#1e1e1e', 'border': '1px solid #29b6f6', 'borderRadius': '8px', 'padding': '12px', 'marginBottom': '15px'}, children=[
        html.H4("MARKET INSIGHTS -- FULL SITUATION READ", style={'color': '#29b6f6', 'marginTop': '0', 'textAlign': 'center', 'fontSize': '16px'}),
        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'justifyContent': 'center'}, children=[
            html.Div(style={'flex': '1 1 260px'}, children=[
                row("Trend Regime", f"{ins['regime']} (ADX {ins['adx']:.1f})" if pd.notna(ins['adx']) else ins['regime'], regime_color),
                row("VWAP", f"{ins['vwap']:.2f} -- {ins['vwap_bias']} ({ins['vwap_dist_pct']:+.2f}%)"),
                row("India VIX", f"{fmt(ins['vix_val'], decimals=2)} -- {ins['vix_state']}"),
                row("Volume (session, proxy)", f"{ins['today_vol']:,.0f} -- {ins['vol_trend']}"),
            ]),
            html.Div(style={'flex': '1 1 260px'}, children=[
                row("RSI (14)", f"{ins['rsi']:.1f} -- {ins['rsi_state']}" if pd.notna(ins['rsi']) else "N/A"),
                row("RSI Divergence", ins['divergence'], "#ff9800" if "divergence" in ins['divergence'].lower() and "No" not in ins['divergence'] else "#ccc"),
                row("Bollinger Bands", ins['bb_state'] + (f" ({ins['bb_width_pct']:.2f}% width)" if ins['bb_width_pct'] else "")),
                row("Price vs Bands", ins['bb_position']),
            ]),
        ]),
        html.Div("Volume is a High-Low range proxy -- the NIFTY 50 index itself carries no traded volume; for true volume, wire this to NIFTY futures data.",
                 style={'color': '#666', 'fontSize': '10px', 'textAlign': 'center', 'marginTop': '8px'})
    ])

# ==================== DASHBOARD CARDS ====================

def generate_options_dashboard_cards(df, vix_val=None):
    if df.empty:
        return html.Div("No Data Available", style={'color': 'white'})

    curr_spot = df.iloc[-1]['Close']
    curr_regime = df.iloc[-1]['Regime']

    ic = build_iron_condor(curr_spot, df, vix_val)
    atm_strike, last_signal, legs = build_atm_directional(curr_spot, df, ic['expected_move'])
    pnl = compute_ic_unrealized_pnl(ic)

    live_tag = "LIVE" if ic['data_is_live'] else "ESTIMATED (no live premium -- check API session)"
    live_color = "#00e676" if ic['data_is_live'] else "#ff9800"

    call_leg = legs["CE"]
    put_leg = legs["PE"]

    if last_signal == 1:
        trade_type, active_leg = "BUY CALL", call_leg
    elif last_signal == -1:
        trade_type, active_leg = "BUY PUT", put_leg
    else:
        trade_type, active_leg = "NO TRADE / WAIT", None

    gate_block = html.Div()
    if ic['gate_reasons']:
        gate_block = html.Div(
            "ENTRY BLOCKED: " + "; ".join(ic['gate_reasons']),
            style={'color': '#ff1744', 'fontSize': '11px', 'textAlign': 'center', 'marginBottom': '6px', 'fontWeight': 'bold'}
        )

    rr_warn = html.Div()
    if ic['data_is_live'] and (not ic['call_meets_rr'] or not ic['put_meets_rr']):
        rr_warn = html.Div(
            f"Risk/reward below the {STD_PARAMS['IC_MIN_CREDIT_TO_MAXLOSS']*100:.0f}% credit/max-loss threshold at every wing width tried -- consider skipping this entry.",
            style={'color': '#ff9800', 'fontSize': '11px', 'textAlign': 'center', 'marginBottom': '6px'}
        )

    pnl_line = "UNREALIZED P&L: N/A (live premiums unavailable)"
    pnl_color = '#ccc'
    if pnl['status'] == "JUST_OPENED":
        pnl_line = "Position just opened -- tracking unrealized P&L from here (net of est. costs)"
        pnl_color = '#ffd700'
    elif pnl['status'] == "ENTRY_BLOCKED":
        pnl_line = "No position opened -- entry blocked (see warning above)"
        pnl_color = '#ff1744'
    elif pnl['pnl_points'] is not None:
        pnl_line = f"UNREALIZED P&L: {pnl['pnl_points']:+.2f} pts ({pnl['pnl_rupees']:+.2f} \u20b9/lot, net of est. costs)"
        pnl_color = '#00e676' if pnl['pnl_points'] >= 0 else '#ff1744'
        if pnl['status'].startswith("SL_HIT"):
            pnl_line += f" -- STOP-LOSS TRIGGERED ({pnl['status']}), POSITION CLOSED"
            pnl_color = '#ff1744'

    greeks = ic['greeks']

    return html.Div(style={'backgroundColor': '#1e1e1e', 'border': '1px solid #ffd700', 'borderRadius': '8px', 'padding': '12px', 'marginBottom': '15px'}, children=[
        html.H4("LIVE OPTIONS TRADING SUGGESTIONS & STOP-LOSS TRACKER", style={'color': '#ffd700', 'marginTop': '0', 'textAlign': 'center', 'fontSize': '16px'}),
        html.Div(f"Data quality: {live_tag} | Expected move: {fmt(ic['expected_move'])} pts ({ic['em_source']}"
                 + (f", VIX {fmt(ic['vix'],decimals=2)}" if ic['vix'] else "") + f") | Expiry: {ic['expiry']} ({ic['days_to_expiry']}d)"
                 + f" | Target delta: {ic['target_delta']}",
                 style={'color': live_color, 'fontSize': '11px', 'textAlign': 'center', 'marginBottom': '4px'}),
        gate_block, rr_warn,

        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}, children=[

            html.Div(style={'flex': '1 1 320px', 'backgroundColor': '#2a2a2a', 'padding': '12px', 'borderRadius': '6px'}, children=[
                html.H5("Iron Condor -- Delta/Skew-Sized Strikes & Greeks", style={'color': '#00e676', 'marginTop': '0', 'fontSize': '14px'}),
                html.Div(f"Spot: {curr_spot:.2f}  |  Call wing: {ic['call_wing']} pts  |  Put wing: {ic['put_wing']} pts", style={'color': '#fff', 'fontSize': '12px'}),
                html.Hr(style={'borderColor': '#444'}),
                
                # Legs with output Delta, Gamma & Vega
                html.Ul(style={'color': '#ccc', 'fontSize': '11px', 'paddingLeft': '16px'}, children=[
                    html.Li([
                        f"BUY PUT: ", html.B(f"{ic['put_hedge_strike']:.0f} PE" if ic['put_hedge_strike'] else "N/A"), 
                        f" | Prem: {fmt(ic['premiums']['put_hedge'],'\u20b9')}",
                        html.Div(f"   -> Δ: {greeks['buy_put']['delta']}, Γ: {greeks['buy_put']['gamma']}, ν: {greeks['buy_put']['vega']}", style={'color': '#29b6f6'})
                    ]),
                    html.Li([
                        f"BUY CALL: ", html.B(f"{ic['call_hedge_strike']:.0f} CE" if ic['call_hedge_strike'] else "N/A"), 
                        f" | Prem: {fmt(ic['premiums']['call_hedge'],'\u20b9')}",
                        html.Div(f"   -> Δ: {greeks['buy_call']['delta']}, Γ: {greeks['buy_call']['gamma']}, ν: {greeks['buy_call']['vega']}", style={'color': '#29b6f6'})
                    ]),
                    html.Li([
                        f"SELL PUT: ", html.B(f"{ic['put_short_strike']:.0f} PE"), 
                        f" | Prem: {fmt(ic['premiums']['put_short'],'\u20b9')}",
                        html.Div(f"   -> Δ: {greeks['sell_put']['delta']}, Γ: {greeks['sell_put']['gamma']}, ν: {greeks['sell_put']['vega']}", style={'color': '#29b6f6'})
                    ]),
                    html.Li([
                        f"SELL CALL: ", html.B(f"{ic['call_short_strike']:.0f} CE"), 
                        f" | Prem: {fmt(ic['premiums']['call_short'],'\u20b9')}",
                        html.Div(f"   -> Δ: {greeks['sell_call']['delta']}, Γ: {greeks['sell_call']['gamma']}, ν: {greeks['sell_call']['vega']}", style={'color': '#29b6f6'})
                    ]),
                ]),
                
                html.Div(style={'backgroundColor': '#121212', 'padding': '8px', 'borderRadius': '4px', 'marginTop': '8px'}, children=[
                    html.Div(f"Call credit: {fmt(ic['call_credit'])} pts | Max loss: {fmt(ic['call_max_loss'])} pts | Breakeven: {fmt(ic['call_breakeven'])}", style={'color': '#ccc', 'fontSize': '11px'}),
                    html.Div(f"\U0001F6D1 CALL SL: lose {fmt(ic['call_sl_points'])} pts on that leg (={STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS']*100:.0f}% of max loss)",
                             style={'color': '#ff1744', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '4px 0'}),
                    html.Div(f"Put credit: {fmt(ic['put_credit'])} pts | Max loss: {fmt(ic['put_max_loss'])} pts | Breakeven: {fmt(ic['put_breakeven'])}", style={'color': '#ccc', 'fontSize': '11px'}),
                    html.Div(f"\U0001F6D1 PUT SL: lose {fmt(ic['put_sl_points'])} pts on that leg (={STD_PARAMS['IC_SL_FRACTION_OF_MAXLOSS']*100:.0f}% of max loss)",
                             style={'color': '#ff1744', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '4px 0'}),
                    html.Div(f"Gross credit: {fmt(ic['total_credit_gross'])} pts  |  Est. costs: -{fmt(ic['cost_pts'])} pts  |  Net credit: {fmt(ic['total_credit_net'])} pts",
                             style={'color': '#00e676', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '4px 0'}),
                    html.Div(pnl_line, style={'color': pnl_color, 'fontSize': '13px', 'fontWeight': 'bold'}),
                ])
            ]),

            html.Div(style={'flex': '1 1 320px', 'backgroundColor': '#2a2a2a', 'padding': '12px', 'borderRadius': '6px'}, children=[
                html.H5("Live Directional Signal (ATM Strategy, long-premium)", style={'color': '#29b6f6', 'marginTop': '0', 'fontSize': '14px'}),
                html.Div([f"Market Regime: ", html.B(curr_regime, style={'color': '#ffea00' if curr_regime == 'SIDEWAYS' else '#00e676'})], style={'fontSize': '13px'}),
                html.Div([f"Signal Action: ", html.B(trade_type, style={'color': '#00e676' if last_signal == 1 else '#ff1744' if last_signal == -1 else '#aaa'})], style={'fontSize': '13px'}),
                html.Hr(style={'borderColor': '#444'}),
                html.Div([
                    html.Div(f"{atm_strike:.0f} CE  |  Premium: {fmt(call_leg['premium'],'\u20b9')}  |  \u0394\u2248{call_leg['delta']}", style={'color': '#ccc', 'fontSize': '12px'}),
                    html.Div([f"\U0001F6D1 CALL STOP-LOSS: ", html.B(fmt(call_leg['sl_premium'], '\u20b9'), style={'color': '#ff1744'}),
                              f"   \U0001F3AF Target: ", html.B(fmt(call_leg['tp_premium'], '\u20b9'), style={'color': '#00e676'})], style={'fontSize': '12px'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '6px 0'}),
                    html.Div(f"{atm_strike:.0f} PE  |  Premium: {fmt(put_leg['premium'],'\u20b9')}  |  \u0394\u2248{put_leg['delta']}", style={'color': '#ccc', 'fontSize': '12px'}),
                    html.Div([f"\U0001F6D1 PUT STOP-LOSS: ", html.B(fmt(put_leg['sl_premium'], '\u20b9'), style={'color': '#ff1744'}),
                              f"   \U0001F3AF Target: ", html.B(fmt(put_leg['tp_premium'], '\u20b9'), style={'color': '#00e676'})], style={'fontSize': '12px'}),
                    html.Div(style={'backgroundColor': '#121212', 'padding': '8px', 'borderRadius': '4px', 'marginTop': '8px'}, children=[
                        html.Div(f"Active suggestion: {trade_type}" + (f" -- SL \u20b9{fmt(active_leg['sl_premium'])} / TP \u20b9{fmt(active_leg['tp_premium'])}" if active_leg else ""),
                                 style={'color': '#ffd700', 'fontSize': '12px', 'fontWeight': 'bold'})
                    ])
                ])
            ])
        ])
    ])

def generate_condor_backtest_card(bt):
    if not bt or bt.get("trades", 0) == 0:
        return html.Div(
            "Iron condor backtest: insufficient historical data to compute a sample.",
            style={'color': '#aaa', 'fontSize': '11px', 'textAlign': 'center', 'marginBottom': '10px'}
        )
    color = '#00e676' if bt['total_pnl_pts'] >= 0 else '#ff1744'
    return html.Div(style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'borderRadius': '6px',
                            'border': '2px solid #ab47bc', 'flex': '1 1 260px', 'textAlign': 'center'}, children=[
        html.Div("IRON CONDOR BACKTEST (modeled, not real fills)", style={'color': '#aaa', 'fontSize': '11px', 'fontWeight': 'bold'}),
        html.Div(f"{bt['total_pnl_pts']:+.1f} pts total", style={'color': color, 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '3px 0'}),
        html.Div(f"Win rate: {bt['win_rate']:.0f}%  |  Trades: {bt['trades']}  |  Avg: {bt['avg_pnl_pts']:+.2f} pts", style={'color': '#fff', 'fontSize': '10px'}),
        html.Div(f"Max drawdown: {bt['max_drawdown_pts']:.1f} pts", style={'color': '#ff9800', 'fontSize': '10px'}),
        html.Div(bt['note'], style={'color': '#666', 'fontSize': '9px', 'marginTop': '4px'}),
    ])

# ==================== DASH APP ====================

app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app.layout = html.Div(style={'backgroundColor': '#121212', 'padding': '10px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    
    # Top Section with Title and Request Information at Top Right
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '15px'}, children=[
        html.H3("Dynamic Regime Strategy Engine: NIFTY 50", style={'color': '#ffffff', 'margin': '0'}),
        
        # Top-Right Requested Order & Checks Info Box
        html.Div(style={
            'backgroundColor': '#1e1e1e', 
            'border': '1px solid #ff9800', 
            'borderRadius': '6px', 
            'padding': '10px 15px', 
            'textAlign': 'left', 
            'color': '#ffffff',
            'fontSize': '12px'
        }, children=[
            html.Div(html.B("BASKETING ORDER"), style={'color': '#ff9800', 'marginBottom': '4px'}),
            html.Div("1. BUY PUT"),
            html.Div("2. BUY CALL"),
            html.Div("3. SELL PUT"),
            html.Div("4. SELL CALL"),
            html.Hr(style={'borderColor': '#444', 'margin': '6px 0'}),
            html.Div(html.B("Option Selling Check"), style={'color': '#ffea00', 'marginBottom': '2px'}),
            html.Div("BUY CALL > SELL CALL > SELL PUT > BUY PUT", style={'color': '#00e676', 'fontWeight': 'bold'})
        ])
    ]),
    
    html.Div(id='session-warning'),
    html.Div(id='market-insights-panel'),
    html.Div(id='options-trading-banner'),
    html.Div(id='strategy-performance-cards', style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '15px'}),
    dcc.Graph(id='multi-indicator-graph', config={'responsive': True}),
    dcc.Interval(id='interval-component', interval=STD_PARAMS['REFRESH_MS'], n_intervals=0),
])

@app.callback(
    [Output('multi-indicator-graph', 'figure'),
     Output('strategy-performance-cards', 'children'),
     Output('options-trading-banner', 'children'),
     Output('session-warning', 'children'),
     Output('market-insights-panel', 'children')],
    Input('interval-component', 'n_intervals'),
)
def update_dashboard(n):
    warning = html.Div() if session_active else html.Div(
        "SmartAPI session inactive -- set SMARTAPI_KEY / SMARTAPI_CLIENT_CODE / SMARTAPI_PASSWORD / SMARTAPI_TOTP_SECRET as env vars.",
        style={'color': '#ff1744', 'textAlign': 'center', 'fontSize': '12px', 'marginBottom': '8px'}
    )

    df, is_fallback = fetch_today_or_previous()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", annotations=[{
            "text": "NIFTY 50 DATA NOT FOUND OR API DOWN", "showarrow": False, "font": {"size": 18, "color": "#ff1744"}
        }])
        return fig, [html.Div("Data Not Found", style={'color': '#ff1744', 'fontSize': '16px'})], html.Div(), warning, html.Div()

    cached = get_cached_backtests()
    comb_pnl, comb_t = cached["directional_pnl"], cached["directional_trades"]
    current_regime = df.iloc[-1]['Regime'] if 'Regime' in df.columns else "UNKNOWN"

    color = '#00e676' if comb_pnl >= 0 else '#ff1744'
    card_elements = [
        html.Div(style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'borderRadius': '6px',
                         'border': '2px solid #00e676', 'flex': '1 1 220px', 'textAlign': 'center'}, children=[
            html.Div("DYNAMIC REGIME STRATEGY (directional)", style={'color': '#aaa', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.Div(f"{comb_pnl:+.2f}%", style={'color': color, 'fontSize': '18px', 'fontWeight': 'bold', 'margin': '3px 0'}),
            html.Div(f"Trades: {comb_t}  (last {STD_PARAMS['BACKTEST_DAYS']} sessions)", style={'color': '#ffffff', 'fontSize': '10px'}),
            html.Div(f"State: {current_regime}", style={'color': '#ffea00', 'fontSize': '9px', 'fontWeight': 'bold'})
        ]),
        generate_condor_backtest_card(cached["ic_result"]),
    ]

    vix_val = fetch_ltp(vix_exchange, "INDIA VIX", vix_token) if vix_token else None
    options_banner = generate_options_dashboard_cards(df, vix_val)
    insights_panel = generate_market_insights_card(df, vix_val)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("NIFTY 50 Price, VWAP & BB", "RSI (14)", "MACD (12, 26, 9)")
    )
    fig.add_trace(go.Candlestick(x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['VWAP'], line=dict(color='#ffea00', width=1.5), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Upper'], line=dict(color='#29b6f6', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Lower'], line=dict(color='#29b6f6', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

    comb_buy = df[df['Combined_Sig'] == 1]
    comb_sell = df[df['Combined_Sig'] == -1]
    if not comb_buy.empty:
        fig.add_trace(go.Scatter(x=comb_buy['Timestamp'], y=comb_buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00e676'), name='BUY'), row=1, col=1)
    if not comb_sell.empty:
        fig.add_trace(go.Scatter(x=comb_sell['Timestamp'], y=comb_sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff1744'), name='SELL'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI'], line=dict(color='#ab47bc', width=1.5), name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI_SMA'], line=dict(color='#ffa726', width=1, dash='dot'), name='RSI SMA'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD'], line=dict(color='#29b6f6', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD_Signal'], line=dict(color='#ff9800', width=1.5), name='Signal'), row=3, col=1)
    colors = ['#00e676' if v >= 0 else '#ff1744' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df['Timestamp'], y=df['MACD_Hist'], marker_color=colors, name='Hist'), row=3, col=1)

    fig.update_layout(
        title=f"NIFTY 50 Engine (Regime: {current_regime}){' [fallback: previous session]' if is_fallback else ''}",
        template="plotly_dark", height=700, xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig, card_elements, options_banner, warning, insights_panel

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except AttributeError:
        app.run_server(host='0.0.0.0', port=port, debug=False)
