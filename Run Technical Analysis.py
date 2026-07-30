import os
import sys
import pyotp
import json
import urllib.request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output
from SmartApi import SmartConnect

# ==================== YOUR CREDENTIALS ====================
API_KEY     = "o2b7s4Oo"
CLIENT_CODE = "AACK311190"
PASSWORD    = "8547"
TOTP_SECRET = "YCRQCDQ7NPUHKYH7RS73NXQ5VE"
SCRIP_MASTER_FILE = "scrip_master.json"
# ==========================================================

def download_scrip_master():
    need_download = True
    if os.path.exists(SCRIP_MASTER_FILE):
        file_time = datetime.fromtimestamp(os.path.getmtime(SCRIP_MASTER_FILE)).date()
        if file_time == datetime.now().date():
            need_download = False

    if need_download:
        print("📥 Cache missing or outdated. Downloading official Scrip Master file...")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        urllib.request.urlretrieve(url, SCRIP_MASTER_FILE)
        print("✅ Scrip Master file saved locally as 'scrip_master.json'.")


def resolve_token(symbol_name):
    download_scrip_master()
    symbol_clean = symbol_name.strip().upper()

    alias_map = {
        "NIFTY 50": "NIFTY", "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY", "NIFTY BANK": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY", "NIFTY FINANCIAL SERVICES": "FINNIFTY"
    }
    target = alias_map.get(symbol_clean, symbol_clean)

    try:
        with open(SCRIP_MASTER_FILE, "r") as f:
            master_data = json.load(f)

        df_master = pd.DataFrame(master_data)
        matched = df_master[
            (df_master['symbol'].str.upper() == target) |
            (df_master['name'].str.upper() == target) |
            (df_master['symbol'].str.upper() == f"{target}-EQ")
        ]

        if not matched.empty:
            token = str(matched.iloc[0]['token'])
            exch = matched.iloc[0]['exch_seg']
            return token, exch

    except Exception as e:
        print(f"❌ Error scanning master file: {e}")

    return None, None


selected_symbol = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
token, exchange = resolve_token(selected_symbol)

if not token:
    print(f"\n❌ CRITICAL ERROR: Index/Token for '{selected_symbol}' was NOT found in the master file!")
    sys.exit(1)

# SmartAPI Session Setup
totp_code = pyotp.TOTP(TOTP_SECRET).now() if TOTP_SECRET else ""
smart_api = SmartConnect(api_key=API_KEY)
if CLIENT_CODE and PASSWORD and TOTP_SECRET:
    session_data = smart_api.generateSession(CLIENT_CODE, PASSWORD, totp_code)

STD_PARAMS = {
    "VWAP_SD": 1.5,
    "RSI_PERIOD": 14, "RSI_OVERSOLD": 35, "RSI_OVERBOUGHT": 65,
    "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
    "BB_PERIOD": 20, "BB_STD": 2.0,
    "SL_PCT": 0.002,     # 0.2% Spot Stop Loss
    "RR_RATIO": 1.5,     # 1:1.5 Risk-Reward Ratio
    "LOT_SIZE": 65       # Standard NIFTY 50 Lot Size
}


def safe_get_candle_data(params):
    try:
        response = smart_api.getCandleData(params)
        if isinstance(response, dict) and response.get('status') and response.get('data'):
            return response['data']
    except Exception:
        pass
    return None


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


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

    tr_smooth = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (df['+DM'].ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (df['-DM'].ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    return dx.ewm(alpha=1/period, adjust=False).mean()


def calculate_indicators(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    
    # VWAP & Bands
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    if (df['Volume'] == 0).all():
        df['Volume'] = (df['High'] - df['Low']).replace(0, 1)
    df['PV'] = df['Typical_Price'] * df['Volume']
    df['Cum_PV'] = df.groupby('Date')['PV'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cum_Vol']
    df['VWAP_Std'] = df.groupby('Date')['Close'].transform(lambda x: x.rolling(window=15, min_periods=1).std()).fillna(0)
    df['VWAP_Upper'] = df['VWAP'] + (STD_PARAMS['VWAP_SD'] * df['VWAP_Std'])
    df['VWAP_Lower'] = df['VWAP'] - (STD_PARAMS['VWAP_SD'] * df['VWAP_Std'])

    # Indicators
    df['RSI'] = calculate_rsi(df['Close'], STD_PARAMS['RSI_PERIOD'])
    df['RSI_SMA'] = df['RSI'].rolling(window=9).mean()

    ema_fast = df['Close'].ewm(span=STD_PARAMS['MACD_FAST'], adjust=False).mean()
    ema_slow = df['Close'].ewm(span=STD_PARAMS['MACD_SLOW'], adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=STD_PARAMS['MACD_SIGNAL'], adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['BB_Mid'] = df['Close'].rolling(window=STD_PARAMS['BB_PERIOD']).mean()
    df['BB_Std_Val'] = df['Close'].rolling(window=STD_PARAMS['BB_PERIOD']).std()
    df['BB_Upper'] = df['BB_Mid'] + (STD_PARAMS['BB_STD'] * df['BB_Std_Val'])
    df['BB_Lower'] = df['BB_Mid'] - (STD_PARAMS['BB_STD'] * df['BB_Std_Val'])

    df['ADX'] = calculate_adx(df)
    return df


def generate_individual_signals(df):
    df = calculate_indicators(df)
    df = df.reset_index(drop=True)
    
    df['VWAP_Sig'] = 0
    df['RSI_Sig']  = 0
    df['MACD_Sig'] = 0
    df['BB_Sig']   = 0
    df['Combined_Sig'] = 0
    df['Regime'] = "SIDEWAYS"

    for i in range(2, len(df)):
        rsi_curr   = df.loc[i, 'RSI']
        close_curr = df.loc[i, 'Close']
        close_prev = df.loc[i-1, 'Close']
        adx_val    = df.loc[i, 'ADX']
        vwap_val   = df.loc[i, 'VWAP']

        if pd.isna(rsi_curr) or pd.isna(adx_val):
            continue

        # Dynamic Regime Classification
        if adx_val > 22 and close_curr > vwap_val:
            regime = "BULL"
        elif adx_val > 22 and close_curr < vwap_val:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        df.loc[i, 'Regime'] = regime

        # Adaptive Strategy Routing
        if regime == "BULL":
            macd_buy = df.loc[i, 'MACD'] > df.loc[i, 'MACD_Signal'] and df.loc[i-1, 'MACD'] <= df.loc[i-1, 'MACD_Signal']
            vwap_bounce = (close_prev <= df.loc[i-1, 'VWAP'] and close_curr > vwap_val)
            if (macd_buy or vwap_bounce) and rsi_curr < 68:
                df.loc[i, 'Combined_Sig'] = 1

        elif regime == "BEAR":
            macd_sell = df.loc[i, 'MACD'] < df.loc[i, 'MACD_Signal'] and df.loc[i-1, 'MACD'] >= df.loc[i-1, 'MACD_Signal']
            vwap_rejection = (close_prev >= df.loc[i-1, 'VWAP'] and close_curr < vwap_val)
            if (macd_sell or vwap_rejection) and rsi_curr > 32:
                df.loc[i, 'Combined_Sig'] = -1

        else: # SIDEWAYS
            bb_buy  = close_prev <= df.loc[i-1, 'BB_Lower'] and close_curr > df.loc[i, 'BB_Lower']
            bb_sell = close_prev >= df.loc[i-1, 'BB_Upper'] and close_curr < df.loc[i, 'BB_Upper']
            if bb_buy and rsi_curr < STD_PARAMS['RSI_OVERSOLD']:
                df.loc[i, 'Combined_Sig'] = 1
            elif bb_sell and rsi_curr > STD_PARAMS['RSI_OVERBOUGHT']:
                df.loc[i, 'Combined_Sig'] = -1

    return df


def backtest_signal_col(df, col_name, sl_pct=0.002, rr_ratio=1.5):
    trades = []
    in_pos, entry, pos_type, tp, sl = False, 0, 0, 0, 0

    for i in range(len(df)):
        close = df.iloc[i]['Close']
        high  = df.iloc[i]['High']
        low   = df.iloc[i]['Low']
        sig   = df.iloc[i][col_name]

        if in_pos:
            if pos_type == 1:
                if high >= tp:
                    trades.append(((tp - entry) / entry) * 100)
                    in_pos = False
                elif low <= sl:
                    trades.append(((sl - entry) / entry) * 100)
                    in_pos = False
            elif pos_type == -1:
                if low <= tp:
                    trades.append(((entry - tp) / entry) * 100)
                    in_pos = False
                elif high >= sl:
                    trades.append(((entry - sl) / entry) * 100)
                    in_pos = False

        if not in_pos and sig != 0:
            in_pos = True
            entry = close
            pos_type = sig
            target_dist = entry * sl_pct
            if pos_type == 1:
                sl = entry - target_dist
                tp = entry + (target_dist * rr_ratio)
            else:
                sl = entry + target_dist
                tp = entry - (target_dist * rr_ratio)

    return sum(trades) if trades else 0.0, len(trades)


def fetch_today_or_previous():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    
    params = {
        "exchange": exchange if exchange else "NSE",
        "symboltoken": token,
        "interval": "FIVE_MINUTE",
        "fromdate": f"{today_str} 09:15",
        "todate": now.strftime("%Y-%m-%d %H:%M")
    }
    
    data = safe_get_candle_data(params)
    if data:
        df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return generate_individual_signals(df), False

    for d in range(1, 7):
        prev_date = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        params['fromdate'] = f"{prev_date} 09:15"
        params['todate'] = f"{prev_date} 15:30"
        data = safe_get_candle_data(params)
        if data:
            df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            return generate_individual_signals(df), True

    return pd.DataFrame(), False


# ==================== NEW OPTIONS & P&L CALCULATOR MODULE ====================

def generate_options_dashboard_cards(df):
    if df.empty:
        return html.Div("No Data Available", style={'color': 'white'})

    curr_spot = df.iloc[-1]['Close']
    curr_regime = df.iloc[-1]['Regime']
    last_signal = df.iloc[-1]['Combined_Sig']

    # --- 1. IRON CONDOR LOGIC & P&L ---
    call_short_strike = round((curr_spot + 600) / 50) * 50
    call_hedge_strike = call_short_strike + 200
    put_short_strike  = round((curr_spot - 500) / 50) * 50
    put_hedge_strike  = put_short_strike - 200

    # Estimated premiums & Stop-Losses
    est_call_short_prem = 3.50
    est_call_hedge_prem = 1.70
    est_put_short_prem  = 19.00
    est_put_hedge_prem  = 9.25

    net_credit_pts = (est_call_short_prem - est_call_hedge_prem) + (est_put_short_prem - est_put_hedge_prem)
    net_credit_rupees = net_credit_pts * STD_PARAMS['LOT_SIZE']
    ic_sl_pts = net_credit_pts * 1.5  # Exit trade if loss hits 1.5x credit collected

    # Estimate Live P&L for Iron Condor
    if put_short_strike <= curr_spot <= call_short_strike:
        ic_pnl_pts = net_credit_pts * 0.85 # Decay capturing
        ic_status_color = "#00e676"
    else:
        # Distance breach penalty
        breach_dist = max(curr_spot - call_short_strike, put_short_strike - curr_spot)
        ic_pnl_pts = net_credit_pts - (breach_dist * 0.5)
        ic_status_color = "#ff1744" if ic_pnl_pts < 0 else "#00e676"

    ic_pnl_rupees = ic_pnl_pts * STD_PARAMS['LOT_SIZE']

    # --- 2. ATM DIRECTIONAL SIGNAL & OPTION SL/TP ---
    atm_strike = round(curr_spot / 50) * 50
    spot_sl_dist = curr_spot * STD_PARAMS['SL_PCT']           # ~0.2% Spot SL (~48 pts)
    spot_tp_dist = spot_sl_dist * STD_PARAMS['RR_RATIO']     # ~0.3% Spot TP (~72 pts)

    # Convert Spot SL/TP to Option Premium (Delta approx 0.50 for ATM)
    option_delta = 0.50
    est_atm_prem = 120.00  # Base estimated premium for ATM option
    option_sl_dist = spot_sl_dist * option_delta
    option_tp_dist = spot_tp_dist * option_delta

    if last_signal == 1:
        trade_type = "BUY CALL"
        opt_symbol = f"{atm_strike} CE"
        entry_prem = est_atm_prem
        sl_prem    = round(entry_prem - option_sl_dist, 2)
        tp_prem    = round(entry_prem + option_tp_dist, 2)
        
        # Calculate P&L relative to current spot move
        entry_spot = df[df['Combined_Sig'] == 1].iloc[-1]['Close'] if (df['Combined_Sig'] == 1).any() else curr_spot
        spot_change = curr_spot - entry_spot
        opt_pnl_pts = spot_change * option_delta
        opt_pnl_rupees = opt_pnl_pts * STD_PARAMS['LOT_SIZE']

    elif last_signal == -1:
        trade_type = "BUY PUT"
        opt_symbol = f"{atm_strike} PE"
        entry_prem = est_atm_prem
        sl_prem    = round(entry_prem - option_sl_dist, 2)
        tp_prem    = round(entry_prem + option_tp_dist, 2)

        entry_spot = df[df['Combined_Sig'] == -1].iloc[-1]['Close'] if (df['Combined_Sig'] == -1).any() else curr_spot
        spot_change = entry_spot - curr_spot
        opt_pnl_pts = spot_change * option_delta
        opt_pnl_rupees = opt_pnl_pts * STD_PARAMS['LOT_SIZE']

    else:
        trade_type = "NO TRADE / WAIT"
        opt_symbol = "N/A"
        entry_prem = 0.0
        sl_prem = 0.0
        tp_prem = 0.0
        opt_pnl_pts = 0.0
        opt_pnl_rupees = 0.0

    # --- UI LAYOUT FOR OPTIONS SUGGESTIONS ---
    return html.Div(style={'backgroundColor': '#1e1e1e', 'border': '1px solid #ffd700', 'borderRadius': '8px', 'padding': '15px', 'marginBottom': '20px'}, children=[
        html.H4("💡 LIVE OPTIONS TRADING SUGGESTIONS & P&L TRACKER", style={'color': '#ffd700', 'marginTop': '0', 'textAlign': 'center'}),
        
        html.Div(style={'display': 'flex', 'justify': 'space-between', 'flexWrap': 'wrap'}, children=[
            
            # Left Card: Iron Condor Position
            html.Div(style={'flex': '1', 'minWidth': '320px', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '6px', 'margin': '5px'}, children=[
                html.H5("🗓️ 1-Day Expiry Strategy (Iron Condor)", style={'color': '#00e676', 'marginTop': '0'}),
                html.Div(f"Current NIFTY Spot Price: {curr_spot:.2f}", style={'color': '#fff', 'fontWeight': 'bold'}),
                html.Hr(style={'borderColor': '#444'}),
                html.Ul(style={'color': '#ccc', 'fontSize': '12px', 'paddingLeft': '18px'}, children=[
                    html.Li([f"SELL Call Leg: ", html.B(f"{call_short_strike} CE"), f" | Premium: ~₹{est_call_short_prem:.2f} | ", html.Span(f"Leg SL: ₹{est_call_short_prem*2:.2f}", style={'color': '#ff1744'})]),
                    html.Li([f"BUY Call Hedge: ", html.B(f"{call_hedge_strike} CE"), f" | Premium: ~₹{est_call_hedge_prem:.2f}"]),
                    html.Li([f"SELL Put Leg: ", html.B(f"{put_short_strike} PE"), f" | Premium: ~₹{est_put_short_prem:.2f} | ", html.Span(f"Leg SL: ₹{est_put_put_prem*2:.2f}" if 'est_put_put_prem' in locals() else f"Leg SL: ₹{est_put_short_prem*2:.2f}", style={'color': '#ff1744'})]),
                    html.Li([f"BUY Put Hedge: ", html.B(f"{put_hedge_strike} PE"), f" | Premium: ~₹{est_put_hedge_prem:.2f}"]),
                ]),
                html.Div(style={'backgroundColor': '#121212', 'padding': '10px', 'borderRadius': '4px', 'marginTop': '10px'}, children=[
                    html.Div(f"Net Credit Collected: +{net_credit_pts:.2f} pts (~₹{net_credit_rupees:.2f}/lot)", style={'color': '#00e676', 'fontSize': '11px'}),
                    html.Div(f"Max Strategy Stop-Loss: Exit if loss exceeds -{ic_sl_pts:.2f} pts (~₹{ic_sl_pts*STD_PARAMS['LOT_SIZE']:.2f})", style={'color': '#ff1744', 'fontSize': '11px', 'fontWeight': 'bold'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
                    html.Div(f"UNREALIZED P&L: {ic_pnl_pts:+.2f} pts ({ic_pnl_rupees:+.2f} ₹/lot)", style={'color': ic_status_color, 'fontSize': '14px', 'fontWeight': 'bold'})
                ])
            ]),

            # Right Card: Directional Signal & Option SL/TP
            html.Div(style={'flex': '1', 'minWidth': '320px', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '6px', 'margin': '5px'}, children=[
                html.H5("⚡ Live Directional Signal (ATM Strategy)", style={'color': '#29b6f6', 'marginTop': '0'}),
                html.Div([f"Market Regime: ", html.B(curr_regime, style={'color': '#ffea00' if curr_regime == 'SIDEWAYS' else '#00e676'})]),
                html.Div([f"Signal Action: ", html.B(trade_type, style={'color': '#00e676' if last_signal == 1 else '#ff1744' if last_signal == -1 else '#aaa'})]),
                html.Hr(style={'borderColor': '#444'}),
                
                if_active := (last_signal != 0),
                html.Div(children=[
                    html.Div([f"Target ATM Option: ", html.B(opt_symbol, style={'color': '#29b6f6'})]),
                    html.Div([f"Est. Entry Premium: ", html.B(f"₹{entry_prem:.2f}")]),
                    html.Div([f"🛑 OPTION STOP-LOSS: ", html.B(f"₹{sl_prem:.2f} (Exit if Premium drops below this)", style={'color': '#ff1744'})]),
                    html.Div([f"🎯 OPTION TARGET (TP): ", html.B(f"₹{tp_prem:.2f} (Target level)", style={'color': '#00e676'})]),
                    
                    html.Div(style={'backgroundColor': '#121212', 'padding': '10px', 'borderRadius': '4px', 'marginTop': '10px'}, children=[
                        html.Div(f"Spot SL Distance: {spot_sl_dist:.2f} pts | Spot TP Distance: {spot_tp_dist:.2f} pts", style={'color': '#888', 'fontSize': '10px'}),
                        html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
                        html.Div(
                            f"ESTIMATED TRADE P&L: {opt_pnl_pts:+.2f} pts ({opt_pnl_rupees:+.2f} ₹/lot)" if if_active else "ESTIMATED TRADE P&L: ₹0.00 (WAITING FOR SIGNAL)",
                            style={'color': '#00e676' if opt_pnl_pts >= 0 else '#ff1744', 'fontSize': '14px', 'fontWeight': 'bold'}
                        )
                    ])
                ])
            ])
        ])
    ])


# Dash Web UI Layout
app = Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#121212', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    html.H2(f"Dynamic Regime Strategy Engine: {selected_symbol}", style={'color': '#ffffff', 'textAlign': 'center'}),
    html.Div(id='options-trading-banner'),  # <--- OPTIONS SUGGESTIONS AND P&L BANNER
    html.Div(id='strategy-performance-cards', style={'display': 'flex', 'justify': 'space-around', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
    dcc.Graph(id='multi-indicator-graph'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])


@app.callback(
    [Output('multi-indicator-graph', 'figure'),
     Output('strategy-performance-cards', 'children'),
     Output('options-trading-banner', 'children')],  # <--- NEW OUTPUT
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    df, is_fallback = fetch_today_or_previous()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            annotations=[{
                "text": f"❌ TOKEN / INDEX DATA NOT FOUND FOR: '{selected_symbol}'",
                "showarrow": False, "font": {"size": 22, "color": "#ff1744"}
            }]
        )
        return fig, [html.Div(f"Index or Token Not Found for '{selected_symbol}'", style={'color': '#ff1744', 'fontSize': '18px'})], html.Div()

    # Backtest PnL Strategy Cards
    vwap_pnl, vwap_t = backtest_signal_col(df, 'VWAP_Sig')
    rsi_pnl, rsi_t   = backtest_signal_col(df, 'RSI_Sig')
    macd_pnl, macd_t = backtest_signal_col(df, 'MACD_Sig')
    bb_pnl, bb_t     = backtest_signal_col(df, 'BB_Sig')
    comb_pnl, comb_t = backtest_signal_col(df, 'Combined_Sig')

    current_regime = df.iloc[-1]['Regime'] if 'Regime' in df.columns else "UNKNOWN"

    cards = [
        ("VWAP Strategy", vwap_pnl, vwap_t, f"Standard {STD_PARAMS['VWAP_SD']}σ"),
        ("RSI Strategy", rsi_pnl, rsi_t, f"Period {STD_PARAMS['RSI_PERIOD']} (MA Cross)"),
        ("MACD Strategy", macd_pnl, macd_t, f"Std ({STD_PARAMS['MACD_FAST']}/{STD_PARAMS['MACD_SLOW']}/{STD_PARAMS['MACD_SIGNAL']})"),
        ("Bollinger Bands", bb_pnl, bb_t, f"Std ({STD_PARAMS['BB_PERIOD']}, {STD_PARAMS['BB_STD']}σ)"),
        ("DYNAMIC REGIME ENGINE", comb_pnl, comb_t, f"Current State: {current_regime}")
    ]

    card_elements = []
    for title, pnl, trades, sub in cards:
        color = '#00e676' if pnl >= 0 else '#ff1744'
        is_main = title == "DYNAMIC REGIME ENGINE"
        card_elements.append(
            html.Div(style={
                'backgroundColor': '#1e1e1e', 'padding': '15px', 'borderRadius': '8px',
                'border': '2px solid #00e676' if is_main else '1px solid #333', 'minWidth': '180px', 'textAlign': 'center'
            }, children=[
                html.Div(title, style={'color': '#aaa', 'fontSize': '12px', 'fontWeight': 'bold'}),
                html.Div(f"{pnl:+.2f}%", style={'color': color, 'fontSize': '20px', 'fontWeight': 'bold', 'margin': '5px 0'}),
                html.Div(f"Trades: {trades}", style={'color': '#ffffff', 'fontSize': '12px'}),
                html.Div(sub, style={'color': '#ffea00' if is_main else '#888', 'fontSize': '10px', 'marginTop': '4px', 'fontWeight': 'bold' if is_main else 'normal'})
            ])
        )

    # Generate Options Banner Card
    options_banner = generate_options_dashboard_cards(df)

    # Create Charts
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Price, VWAP & Bollinger Bands", "Standard RSI (14)", "MACD (12, 26, 9) Signal & Histogram")
    )

    fig.add_trace(go.Candlestick(x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['VWAP'], line=dict(color='#ffea00', width=1.5), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Upper'], line=dict(color='#29b6f6', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['BB_Lower'], line=dict(color='#29b6f6', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

    comb_buy = df[df['Combined_Sig'] == 1]
    comb_sell = df[df['Combined_Sig'] == -1]
    if not comb_buy.empty:
        fig.add_trace(go.Scatter(x=comb_buy['Timestamp'], y=comb_buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676'), name='DYNAMIC BUY'), row=1, col=1)
    if not comb_sell.empty:
        fig.add_trace(go.Scatter(x=comb_sell['Timestamp'], y=comb_sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ff1744'), name='DYNAMIC SELL'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI'], line=dict(color='#ab47bc', width=1.5), name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI_SMA'], line=dict(color='#ffa726', width=1, dash='dot'), name='RSI SMA'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD'], line=dict(color='#29b6f6', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD_Signal'], line=dict(color='#ff9800', width=1.5), name='Signal'), row=3, col=1)
    
    colors = ['#00e676' if val >= 0 else '#ff1744' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df['Timestamp'], y=df['MACD_Hist'], marker_color=colors, name='Histogram'), row=3, col=1)

    chart_title = f"{selected_symbol} Dynamic Regime Engine (Active Regime: {current_regime})"
    fig.update_layout(title=chart_title, template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=True)
    
    return fig, card_elements, options_banner


if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:8050")
    app.run(port=8050, debug=False)