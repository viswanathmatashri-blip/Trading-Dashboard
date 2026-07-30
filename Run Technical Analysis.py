import os
import math
import json
import urllib.request
import pyotp
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dash import Dash, dcc, html, Input, Output
from SmartApi import SmartConnect

# ==================== CREDENTIALS (set these as Render environment variables) ====================
# On Render: Dashboard -> your service -> Environment -> add these keys.
# NEVER hardcode secrets in the file itself.
API_KEY     = os.environ.get("SMARTAPI_KEY", "o2b7s4Oo")
CLIENT_CODE = os.environ.get("SMARTAPI_CLIENT_CODE", "AACK311190")
PASSWORD    = os.environ.get("SMARTAPI_PASSWORD", "8547")
TOTP_SECRET = os.environ.get("SMARTAPI_TOTP_SECRET", "YCRQCDQ7NPUHKYH7RS73NXQ5VE")
# ===================================================================================================

IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return datetime.now(IST)


SCRIP_MASTER_FILE = "/tmp/scrip_master.json"
SELECTED_SYMBOL = "NIFTY 50"

STD_PARAMS = {
    "VWAP_SD": 1.5,
    "RSI_PERIOD": 14, "RSI_OVERSOLD": 35, "RSI_OVERBOUGHT": 65,
    "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
    "BB_PERIOD": 20, "BB_STD": 2.0,
    "SL_PCT": 0.002,          # spot-based SL distance for the directional ATM trade
    "RR_RATIO": 1.5,
    "LOT_SIZE": 65,
    "STRIKE_STEP": 50,
    "BACKTEST_DAYS": 5,       # more days = statistically sturdier backtest cards

    # ---- Iron Condor volatility sizing ----
    "IC_SD_MULTIPLIER": 0.85,   # short strikes placed at 0.85x the expected move (~ wider than 1SD => higher win rate, lower credit)
    "IC_WING_RATIO": 0.35,      # hedge width as a fraction of the expected move
    "IC_MIN_WING": 100,         # floor for hedge width so risk stays defined
    "IC_SL_MULTIPLIER": 1.5,    # exit a side when its loss reaches this multiple of the credit collected on that side

    "REFRESH_MS": int(os.environ.get("REFRESH_MS", 30000)),  # 30s default -- 10s will hit SmartAPI rate limits
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
    with open(SCRIP_MASTER_FILE, "r") as f:
        return pd.DataFrame(json.load(f))


def resolve_nifty_token(df_master):
    try:
        matched = df_master[
            (df_master['symbol'].str.upper() == "NIFTY") |
            (df_master['name'].str.upper() == "NIFTY 50") |
            (df_master['symbol'].str.upper() == "NIFTY 50")
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
    """Filter the scrip master down to NIFTY weekly/monthly index options and
    resolve the nearest upcoming expiry. Returns (options_df, expiry_date)."""
    try:
        opts = df_master[
            (df_master['name'].str.upper() == "NIFTY") &
            (df_master['instrumenttype'].str.upper() == "OPTIDX") &
            (df_master['exch_seg'].str.upper() == "NFO")
        ].copy()
        opts['expiry_dt'] = pd.to_datetime(opts['expiry'], format="%d%b%Y", errors='coerce')
        opts['strike_val'] = pd.to_numeric(opts['strike'], errors='coerce') / 100.0  # Angel stores strike in paise
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
    """opt_type: 'CE' or 'PE'. Returns (token, tradingsymbol, exch_seg) or (None, None, None)."""
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


# Resolve once at startup; scrip master itself refreshes daily inside download_scrip_master()
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
    """Wrap a SmartConnect call; re-login once if it looks like an auth failure."""
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


def safe_get_candle_data(params):
    resp = api_call(smart_api.getCandleData, params)
    if isinstance(resp, dict) and resp.get('status') and resp.get('data'):
        return resp['data']
    return None


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
        df['Volume'] = (df['High'] - df['Low']).replace(0, 1)  # NIFTY index has no real volume -- proxy only
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

def _get_candles_for_range(from_dt, to_dt):
    params = {
        "exchange": exchange or "NSE",
        "symboltoken": token,
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
    """Data for the chart: today if the market has printed candles, else the most recent session."""
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


def fetch_recent_days(n_days=5):
    """Larger sample for the backtest cards -- a single day (~75 candles) isn't enough to judge a strategy."""
    now = now_ist()
    frames = []
    d = 0
    collected = 0
    while collected < n_days and d < n_days * 3:
        d += 1
        day = now - timedelta(days=d)
        if day.weekday() >= 5:  # skip weekends
            continue
        day_open = day.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
        day_close = day.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
        df = _get_candles_for_range(day_open, day_close)
        if df is not None and not df.empty:
            frames.append(df)
            collected += 1
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined['Timestamp'] = pd.to_datetime(combined['Timestamp'])
    combined = combined.sort_values('Timestamp').reset_index(drop=True)
    return generate_individual_signals(combined)


# ==================== VOLATILITY-BASED EXPECTED MOVE ====================

def get_days_to_expiry():
    if current_expiry is None:
        return 1
    delta = (current_expiry - now_ist().date()).days
    return max(delta, 0) + 1  # at least same-day time value


def compute_expected_move(spot, df_recent):
    """
    Preferred: India VIX implied expected move = spot * (VIX/100) * sqrt(days_to_expiry/365)
    Fallback:  ATR-based realized-vol expected move if VIX isn't available.
    """
    dte = get_days_to_expiry()
    vix_val = fetch_ltp(vix_exchange, "INDIA VIX", vix_token) if vix_token else None

    if vix_val:
        em = spot * (vix_val / 100.0) * math.sqrt(dte / 365.0)
        return em, vix_val, "VIX"

    # ATR fallback: scale average 5-min ATR up to a full day, then to days-to-expiry
    if df_recent is not None and not df_recent.empty and 'ATR' in df_recent.columns:
        atr = df_recent['ATR'].dropna().iloc[-1] if df_recent['ATR'].notna().any() else spot * 0.002
        bars_per_day = 75  # 09:15-15:30 in 5-min bars
        daily_move = atr * math.sqrt(bars_per_day)
        em = daily_move * math.sqrt(dte)
        return em, None, "ATR"

    # last-resort static fallback (~0.6% of spot as a 1-day move)
    return spot * 0.006 * math.sqrt(dte), None, "STATIC"


# ==================== IRON CONDOR: DYNAMIC STRIKES, REAL PREMIUMS, STOP LOSSES ====================

def round_to_step(value, step):
    return round(value / step) * step


def build_iron_condor(spot, df_recent):
    step = STD_PARAMS['STRIKE_STEP']
    expected_move, vix_val, em_source = compute_expected_move(spot, df_recent)

    call_leg_dist = STD_PARAMS['IC_SD_MULTIPLIER'] * expected_move
    put_leg_dist = STD_PARAMS['IC_SD_MULTIPLIER'] * expected_move
    wing_width = max(STD_PARAMS['IC_MIN_WING'], round_to_step(STD_PARAMS['IC_WING_RATIO'] * expected_move, step))

    call_short_strike = round_to_step(spot + call_leg_dist, step)
    call_hedge_strike = call_short_strike + wing_width
    put_short_strike = round_to_step(spot - put_leg_dist, step)
    put_hedge_strike = put_short_strike - wing_width

    # --- fetch real premiums (None if session/data unavailable -- we do NOT fabricate numbers) ---
    premiums = {}
    for label, strike, opt_type in [
        ("call_short", call_short_strike, "CE"), ("call_hedge", call_hedge_strike, "CE"),
        ("put_short", put_short_strike, "PE"), ("put_hedge", put_hedge_strike, "PE"),
    ]:
        tok, tsym, exch = get_option_token(nifty_options_df, strike, opt_type)
        premiums[label] = fetch_ltp(exch, tsym, tok) if tok else None

    data_is_live = all(v is not None for v in premiums.values())

    call_credit = (premiums['call_short'] - premiums['call_hedge']) if data_is_live else None
    put_credit = (premiums['put_short'] - premiums['put_hedge']) if data_is_live else None
    total_credit = (call_credit + put_credit) if data_is_live else None

    result = {
        "expected_move": expected_move, "vix": vix_val, "em_source": em_source,
        "wing_width": wing_width, "expiry": current_expiry, "days_to_expiry": get_days_to_expiry(),
        "call_short_strike": call_short_strike, "call_hedge_strike": call_hedge_strike,
        "put_short_strike": put_short_strike, "put_hedge_strike": put_hedge_strike,
        "premiums": premiums, "data_is_live": data_is_live,
        "call_credit": call_credit, "put_credit": put_credit, "total_credit": total_credit,
    }

    if not data_is_live:
        result.update({
            "call_sl_points": None, "put_sl_points": None, "combined_sl_points": None,
            "call_sl_spot": None, "put_sl_spot": None,
            "call_breakeven": None, "put_breakeven": None,
        })
        return result

    sl_mult = STD_PARAMS['IC_SL_MULTIPLIER']
    call_max_loss = wing_width - call_credit
    put_max_loss = wing_width - put_credit
    call_sl_points = min(call_credit * sl_mult, call_max_loss)
    put_sl_points = min(put_credit * sl_mult, put_max_loss)

    call_breakeven = call_short_strike + call_credit
    put_breakeven = put_short_strike - put_credit

    # spot level at which the loss on that side reaches its SL threshold (expiry-value approximation)
    call_sl_spot = call_short_strike + call_credit + call_sl_points
    put_sl_spot = put_short_strike - put_credit - put_sl_points

    combined_sl_points = min(total_credit * sl_mult, call_max_loss + put_max_loss)

    result.update({
        "call_sl_points": call_sl_points, "put_sl_points": put_sl_points,
        "combined_sl_points": combined_sl_points,
        "call_sl_spot": call_sl_spot, "put_sl_spot": put_sl_spot,
        "call_breakeven": call_breakeven, "put_breakeven": put_breakeven,
        "call_max_loss": call_max_loss, "put_max_loss": put_max_loss,
    })
    return result


def estimate_delta(spot, strike, expected_move, opt_type):
    """Rough moneyness-based delta approximation (not a full Black-Scholes model,
    but far more realistic than a flat 0.50 for every strike)."""
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
            "spot_sl_dist": spot_sl_dist, "spot_tp_dist": spot_tp_dist,
        }
    return atm_strike, last_signal, legs


# ==================== DASHBOARD CARDS ====================

def fmt(v, prefix="", suffix="", decimals=2):
    if v is None:
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def generate_options_dashboard_cards(df):
    if df.empty:
        return html.Div("No Data Available", style={'color': 'white'})

    curr_spot = df.iloc[-1]['Close']
    curr_regime = df.iloc[-1]['Regime']

    ic = build_iron_condor(curr_spot, df)
    atm_strike, last_signal, legs = build_atm_directional(curr_spot, df, ic['expected_move'])

    live_tag = "LIVE" if ic['data_is_live'] else f"ESTIMATED (no live premium -- check API session)"
    live_color = "#00e676" if ic['data_is_live'] else "#ff9800"

    call_leg = legs["CE"]
    put_leg = legs["PE"]

    if last_signal == 1:
        trade_type, active_leg = "BUY CALL", call_leg
    elif last_signal == -1:
        trade_type, active_leg = "BUY PUT", put_leg
    else:
        trade_type, active_leg = "NO TRADE / WAIT", None

    return html.Div(style={'backgroundColor': '#1e1e1e', 'border': '1px solid #ffd700', 'borderRadius': '8px', 'padding': '12px', 'marginBottom': '15px'}, children=[
        html.H4("LIVE OPTIONS TRADING SUGGESTIONS & STOP-LOSS TRACKER", style={'color': '#ffd700', 'marginTop': '0', 'textAlign': 'center', 'fontSize': '16px'}),
        html.Div(f"Data quality: {live_tag} | Expected move: {fmt(ic['expected_move'])} pts ({ic['em_source']}"
                 + (f", VIX {fmt(ic['vix'],decimals=2)}" if ic['vix'] else "") + f") | Expiry: {ic['expiry']} ({ic['days_to_expiry']}d)",
                 style={'color': live_color, 'fontSize': '11px', 'textAlign': 'center', 'marginBottom': '8px'}),

        html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}, children=[

            # ---- Iron Condor card ----
            html.Div(style={'flex': '1 1 320px', 'backgroundColor': '#2a2a2a', 'padding': '12px', 'borderRadius': '6px'}, children=[
                html.H5("Iron Condor -- Volatility-Sized Strikes", style={'color': '#00e676', 'marginTop': '0', 'fontSize': '14px'}),
                html.Div(f"Spot: {curr_spot:.2f}  |  Wing width: {ic['wing_width']} pts (dynamic)", style={'color': '#fff', 'fontSize': '12px'}),
                html.Hr(style={'borderColor': '#444'}),
                html.Ul(style={'color': '#ccc', 'fontSize': '11px', 'paddingLeft': '16px'}, children=[
                    html.Li([f"SELL Call: ", html.B(f"{ic['call_short_strike']:.0f} CE"), f"  Premium: {fmt(ic['premiums']['call_short'],'\u20b9')}"]),
                    html.Li([f"BUY Call Hedge: ", html.B(f"{ic['call_hedge_strike']:.0f} CE"), f"  Premium: {fmt(ic['premiums']['call_hedge'],'\u20b9')}"]),
                    html.Li([f"SELL Put: ", html.B(f"{ic['put_short_strike']:.0f} PE"), f"  Premium: {fmt(ic['premiums']['put_short'],'\u20b9')}"]),
                    html.Li([f"BUY Put Hedge: ", html.B(f"{ic['put_hedge_strike']:.0f} PE"), f"  Premium: {fmt(ic['premiums']['put_hedge'],'\u20b9')}"]),
                ]),
                html.Div(style={'backgroundColor': '#121212', 'padding': '8px', 'borderRadius': '4px', 'marginTop': '8px'}, children=[
                    html.Div(f"Call side credit: {fmt(ic['call_credit'])} pts  |  Breakeven: {fmt(ic['call_breakeven'])}", style={'color': '#ccc', 'fontSize': '11px'}),
                    html.Div(f"\U0001F6D1 CALL SIDE STOP-LOSS: {fmt(ic['call_sl_points'])} pts loss  (exit if spot \u2265 {fmt(ic['call_sl_spot'])})",
                             style={'color': '#ff1744', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '4px 0'}),
                    html.Div(f"Put side credit: {fmt(ic['put_credit'])} pts  |  Breakeven: {fmt(ic['put_breakeven'])}", style={'color': '#ccc', 'fontSize': '11px'}),
                    html.Div(f"\U0001F6D1 PUT SIDE STOP-LOSS: {fmt(ic['put_sl_points'])} pts loss  (exit if spot \u2264 {fmt(ic['put_sl_spot'])})",
                             style={'color': '#ff1744', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '4px 0'}),
                    html.Div(f"Net credit (total): {fmt(ic['total_credit'])} pts (~\u20b9{fmt(ic['total_credit']*STD_PARAMS['LOT_SIZE']) if ic['total_credit'] else 'N/A'}/lot)",
                             style={'color': '#00e676', 'fontSize': '12px', 'fontWeight': 'bold'}),
                    html.Div(f"\U0001F6D1 COMBINED IC STOP-LOSS: {fmt(ic['combined_sl_points'])} pts (~\u20b9{fmt(ic['combined_sl_points']*STD_PARAMS['LOT_SIZE']) if ic['combined_sl_points'] else 'N/A'}/lot)",
                             style={'color': '#ff1744', 'fontSize': '13px', 'fontWeight': 'bold'}),
                ])
            ]),

            # ---- Directional ATM card ----
            html.Div(style={'flex': '1 1 320px', 'backgroundColor': '#2a2a2a', 'padding': '12px', 'borderRadius': '6px'}, children=[
                html.H5("Live Directional Signal (ATM Strategy)", style={'color': '#29b6f6', 'marginTop': '0', 'fontSize': '14px'}),
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


# ==================== DASH APP ====================

app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app.layout = html.Div(style={'backgroundColor': '#121212', 'padding': '10px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    html.H3("Dynamic Regime Strategy Engine: NIFTY 50", style={'color': '#ffffff', 'textAlign': 'center', 'margin': '10px 0'}),
    html.Div(id='session-warning'),
    html.Div(id='options-trading-banner'),
    html.Div(id='strategy-performance-cards', style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap', 'gap': '8px', 'marginBottom': '15px'}),
    dcc.Graph(id='multi-indicator-graph', config={'responsive': True}),
    dcc.Interval(id='interval-component', interval=STD_PARAMS['REFRESH_MS'], n_intervals=0)
])


@app.callback(
    [Output('multi-indicator-graph', 'figure'),
     Output('strategy-performance-cards', 'children'),
     Output('options-trading-banner', 'children'),
     Output('session-warning', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    warning = html.Div() if session_active else html.Div(
        "SmartAPI session inactive -- set SMARTAPI_KEY / SMARTAPI_CLIENT_CODE / SMARTAPI_PASSWORD / SMARTAPI_TOTP_SECRET as env vars on Render. Showing indicator chart only where data is cached.",
        style={'color': '#ff1744', 'textAlign': 'center', 'fontSize': '12px', 'marginBottom': '8px'}
    )

    df, is_fallback = fetch_today_or_previous()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", annotations=[{
            "text": "NIFTY 50 DATA NOT FOUND OR API DOWN", "showarrow": False, "font": {"size": 18, "color": "#ff1744"}
        }])
        return fig, [html.Div("Data Not Found", style={'color': '#ff1744', 'fontSize': '16px'})], html.Div(), warning

    backtest_df = fetch_recent_days(STD_PARAMS['BACKTEST_DAYS'])
    if backtest_df.empty:
        backtest_df = df

    comb_pnl, comb_t = backtest_signal_col(backtest_df, 'Combined_Sig', STD_PARAMS['SL_PCT'], STD_PARAMS['RR_RATIO'])
    current_regime = df.iloc[-1]['Regime'] if 'Regime' in df.columns else "UNKNOWN"

    color = '#00e676' if comb_pnl >= 0 else '#ff1744'
    card_elements = [
        html.Div(style={'backgroundColor': '#1e1e1e', 'padding': '10px', 'borderRadius': '6px',
                         'border': '2px solid #00e676', 'flex': '1 1 220px', 'textAlign': 'center'}, children=[
            html.Div("DYNAMIC REGIME STRATEGY", style={'color': '#aaa', 'fontSize': '11px', 'fontWeight': 'bold'}),
            html.Div(f"{comb_pnl:+.2f}%", style={'color': color, 'fontSize': '18px', 'fontWeight': 'bold', 'margin': '3px 0'}),
            html.Div(f"Trades: {comb_t}  (last {STD_PARAMS['BACKTEST_DAYS']} sessions)", style={'color': '#ffffff', 'fontSize': '10px'}),
            html.Div(f"State: {current_regime}", style={'color': '#ffea00', 'fontSize': '9px', 'fontWeight': 'bold'})
        ])
    ]

    options_banner = generate_options_dashboard_cards(df)

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
    return fig, card_elements, options_banner, warning


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except AttributeError:
        app.run_server(host='0.0.0.0', port=port, debug=False)
