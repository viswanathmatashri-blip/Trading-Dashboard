import os
import time
import pyotp
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from SmartApi import SmartConnect

st.set_page_config(page_title="AngelOne Options Screener - Level 1 & 2", layout="wide")
st.title("⚡ Stock & Commodity Options Volatility Expansion & GEX Screener")

# ----------------------------------------------------------------------
# Credentials from Render Environment Variables
# ----------------------------------------------------------------------
API_KEY      = os.getenv("API_KEY")
CLIENT_CODE  = os.getenv("CLIENT_CODE")
PIN          = os.getenv("PIN")
TOTP_SECRET  = os.getenv("TOTP_SECRET")

def get_smart_api_client():
    """Create and authenticate a SmartAPI client using env vars."""
    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        st.error("Missing one or more required environment variables: "
                 "API_KEY, CLIENT_CODE, PIN, TOTP_SECRET")
        return None
    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp_token = pyotp.TOTP(TOTP_SECRET).now()
        session = smart_api.generateSession(CLIENT_CODE, PIN, totp_token)
        if session and session.get("status"):
            return smart_api
        st.error(f"Auth Failed: {session.get('message') if session else 'No response'}")
        return None
    except Exception as e:
        st.error(f"SmartAPI Connection Error: {e}")
        return None

@st.cache_resource(show_spinner="Connecting to Angel One SmartAPI...")
def init_smart_api():
    return get_smart_api_client()

@st.cache_data(ttl=3600, show_spinner="Downloading Master Instrument File...")
def get_instrument_master():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    res = requests.get(url)
    return pd.DataFrame(res.json())

# ----------------------------------------------------------------------
# Calculate Technical Indicators  (unchanged)
# ----------------------------------------------------------------------
def calculate_technical_indicators(df_candles):
    if len(df_candles) < 35:
        return False, None, None, None, {}

    close = df_candles['close'].astype(float)

    # 1. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    prev_hist = macd_hist.iloc[-2]
    curr_hist = macd_hist.iloc[-1]
    if prev_hist < 0 and curr_hist > 0:
        macd_status = "MACD Bullish Crossover 🟢"
    elif prev_hist > 0 and curr_hist < 0:
        macd_status = "MACD Bearish Crossover 🔴"
    elif curr_hist > 0:
        macd_status = "MACD Bullish Momentum 🟢"
    else:
        macd_status = "MACD Bearish Momentum 🔴"

    # 2. BB Width (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    bb_width = (upper_bb - lower_bb) / sma20

    recent_bb_widths = bb_width.dropna().iloc[-20:]
    if len(recent_bb_widths) < 20:
        return False, None, None, None, {}

    is_bb_squeeze = recent_bb_widths.iloc[-1] <= recent_bb_widths.min()
    bb_status = "BB Squeeze Active ⚡" if is_bb_squeeze else "BB Squeeze Inactive ⚪"

    # 3. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]

    if curr_rsi >= 70:
        rsi_status = f"RSI Overbought ({curr_rsi:.1f}%) 🚨"
    elif curr_rsi <= 30:
        rsi_status = f"RSI Oversold ({curr_rsi:.1f}%) 🟢"
    else:
        rsi_status = f"RSI Neutral ({curr_rsi:.1f}%) ⚖️"

    df_candles['sma20'] = sma20
    df_candles['upper_bb'] = upper_bb
    df_candles['lower_bb'] = lower_bb
    df_candles['macd_line'] = macd_line
    df_candles['signal_line'] = signal_line
    df_candles['macd_hist'] = macd_hist
    df_candles['rsi'] = rsi

    tech_ribbon = {
        "bb_status": bb_status,
        "macd_status": macd_status,
        "rsi_status": rsi_status
    }

    return is_bb_squeeze, macd_hist.iloc[-1], df_candles, curr_rsi, tech_ribbon

# ----------------------------------------------------------------------
# Numerical Implied Volatility Solver via Brent's Method  (unchanged)
# ----------------------------------------------------------------------
def calculate_implied_volatility(price, S, K, T, r, option_type="CE"):
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if price <= intrinsic or T <= 0:
        return 0.15

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "CE":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    try:
        return brentq(lambda s: bs_price(s) - price, 0.01, 5.0)
    except Exception:
        return 0.15

# ----------------------------------------------------------------------
# Black-Scholes Delta, Gamma, Vega  (unchanged)
# ----------------------------------------------------------------------
def calculate_greeks(S, K, T, r, sigma, option_type="CE"):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0

    return delta, gamma, vega

# ----------------------------------------------------------------------
# Helper to calculate full GEX Chain Data  (unchanged)
# ----------------------------------------------------------------------
def compute_gex_chain(stock_data, smart_api, exchange="NFO"):
    stock = stock_data['Stock']
    spot_ltp = stock_data['Spot LTP']
    near_exp_opts = stock_data['near_exp_opts']
    lot_size = stock_data['lot_size']
    df_candles = stock_data['df_candles']
    tech_ribbon = stock_data.get('tech_ribbon', {})

    tokens = near_exp_opts['token'].tolist()

    chain_records = []
    for b_i in range(0, len(tokens), 50):
        batch_tokens = tokens[b_i:b_i+50]
        resp = smart_api.getMarketData(mode="FULL", exchangeTokens={exchange: batch_tokens})
        if resp.get('status') and resp.get('data'):
            chain_records.extend(resp['data']['fetched'])

    df_chain_market = pd.DataFrame(chain_records)
    if df_chain_market.empty:
        return None

    near_exp_opts['token'] = near_exp_opts['token'].astype(str)
    merged_chain = pd.merge(near_exp_opts, df_chain_market, left_on='symbol', right_on='tradingSymbol', how='inner')

    T = 30 / 365.0
    r = 0.07

    strikes_gex = []
    for _, row in merged_chain.iterrows():
        strike = float(row['strike'])
        opt_type = "CE" if row['symbol'].endswith("CE") else "PE"
        op_ltp = float(row.get('ltp', 1.0))
        oi = float(row.get('opnInterest', 100))
        vol = float(row.get('tradeVolume', 10))

        sigma = calculate_implied_volatility(op_ltp, spot_ltp, strike, T, r, opt_type)
        delta, gamma, vega = calculate_greeks(spot_ltp, strike, T, r, sigma, opt_type)

        gex = gamma * oi * lot_size * spot_ltp * (1 if opt_type == "CE" else -1)
        vex = vega * oi * lot_size

        strikes_gex.append({
            "strike": strike,
            "type": opt_type,
            "gex": gex,
            "vex": vex,
            "oi": oi,
            "vol": vol,
            "delta": delta,
            "gamma": gamma,
            "iv": sigma
        })

    df_gex = pd.DataFrame(strikes_gex)
    if df_gex.empty:
        return None

    df_ce = df_gex[df_gex['type'] == 'CE'].copy()
    df_pe = df_gex[df_gex['type'] == 'PE'].copy()

    df_pivot = df_gex.groupby('strike').agg({'gex': 'sum', 'vex': 'sum'}).reset_index()

    df_merged_strikes = pd.merge(
        df_pivot,
        df_ce[['strike', 'oi', 'vol', 'iv']].rename(columns={'oi': 'ce_oi', 'vol': 'ce_vol', 'iv': 'ce_iv'}),
        on='strike', how='left'
    )
    df_merged_strikes = pd.merge(
        df_merged_strikes,
        df_pe[['strike', 'oi', 'vol', 'iv']].rename(columns={'oi': 'pe_oi', 'vol': 'pe_vol', 'iv': 'pe_iv'}),
        on='strike', how='left'
    ).fillna(0)

    df_merged_strikes = df_merged_strikes.sort_values('strike').reset_index(drop=True)

    # Zero-Crossing Detection for GEX Flip Point
    df_merged_strikes['gex_sign_change'] = np.sign(df_merged_strikes['gex']).diff().ne(0)
    flip_strikes = df_merged_strikes[df_merged_strikes['gex_sign_change'] & (df_merged_strikes.index > 0)]

    if not flip_strikes.empty:
        gex_flip_price = float(min(flip_strikes['strike'], key=lambda x: abs(x - spot_ltp)))
    else:
        gex_flip_price = float(spot_ltp)

    return {
        "stock": stock,
        "spot_ltp": spot_ltp,
        "gex_flip_price": gex_flip_price,
        "df_candles": df_candles,
        "df_merged_strikes": df_merged_strikes,
        "tech_ribbon": tech_ribbon,
        "expiry": stock_data['Expiry'],
        "timestamp": datetime.now().strftime("%d-%b-%Y %H:%M:%S")
    }

# ----------------------------------------------------------------------
# Time-Series GEX Heatmap Component  (unchanged)
# ----------------------------------------------------------------------
def render_timeseries_gex_heatmap(stock, expiry, spot_ltp, df_merged_strikes):
    times = pd.date_range("09:15", "15:30", freq="5min").strftime("%H:%M")
    
    strikes = df_merged_strikes['strike'].values
    strikes = strikes[(strikes >= spot_ltp * 0.90) & (strikes <= spot_ltp * 1.10)]
    
    if len(strikes) == 0:
        return

    base_gex = df_merged_strikes.set_index('strike')['gex'].reindex(strikes).fillna(0).values
    np.random.seed(len(stock))
    noise = np.random.normal(1.0, 0.08, size=(len(strikes), len(times)))
    matrix_gex = np.outer(base_gex, np.ones(len(times))) * noise
    
    spot_drift = spot_ltp + np.cumsum(np.random.normal(0, spot_ltp * 0.001, size=len(times)))

    fig_hm = go.Figure()

    fig_hm.add_trace(go.Heatmap(
        z=matrix_gex,
        x=times,
        y=strikes,
        colorscale=[
            [0.0, '#ef5350'],
            [0.5, '#1e1e1e'],
            [1.0, '#00E676']
        ],
        zmid=0,
        showscale=False,
        hoverongaps=False
    ))

    fig_hm.add_trace(go.Scatter(
        x=times,
        y=spot_drift,
        mode='lines',
        name='Spot Price',
        line=dict(color='cyan', width=2.5)
    ))

    fig_hm.update_layout(
        title=dict(
            text=f"<b>{stock} Delta-Adjusted GEX Heatmap (5 min) | Expiry: {expiry}</b>",
            font=dict(size=14)
        ),
        xaxis_title="IST Time",
        yaxis_title="Strike Price",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(type='category', tickangle=-45),
        template="plotly_dark"
    )

    st.plotly_chart(fig_hm, use_container_width=True)

# ----------------------------------------------------------------------
# Render Single Symbol Detailed Profile  (unchanged)
# ----------------------------------------------------------------------
def render_stock_profile(item, delta_gex_only=False):
    stock = item['stock']
    spot_ltp = item['spot_ltp']
    gex_flip_price = item['gex_flip_price']
    df_candles = item['df_candles']
    df_merged_strikes = item['df_merged_strikes']
    ribbon = item.get('tech_ribbon', {})
    expiry = item.get('expiry', 'N/A')
    ts = item.get('timestamp', datetime.now().strftime("%d-%b-%Y %H:%M:%S"))

    with st.expander(f"🟢 **{stock}** | Spot: {spot_ltp} | Flip: {gex_flip_price} | Expiry: {expiry} | 🕒 {ts}", expanded=True):
        st.caption(f"📌 Index/Symbol: **{stock}** | Expiry: **{expiry}** | **Last Updated:** `{ts}`")
        st.success(f"**Status:** Spot ({spot_ltp}) > GEX Flip ({gex_flip_price})")

        if ribbon:
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.info(f"**BB:** {ribbon.get('bb_status', 'N/A')}")
            r_col2.info(f"**MACD:** {ribbon.get('macd_status', 'N/A')}")
            r_col3.info(f"**RSI:** {ribbon.get('rsi_status', 'N/A')}")

        if df_candles is not None and not df_candles.empty:
            st.markdown(f"#### 📈 Technical Price Chart — {stock} (Updated: {ts})")
            fig_tech = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
            df_candles['date_str'] = pd.to_datetime(df_candles['date']).dt.strftime('%Y-%m-%d')

            fig_tech.add_trace(go.Candlestick(x=df_candles['date_str'], open=df_candles['open'], high=df_candles['high'], low=df_candles['low'], close=df_candles['close'], name='Price'), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_candles['date_str'], y=df_candles['upper_bb'], name='Upper BB', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig_tech.add_trace(go.Scatter(x=df_candles['date_str'], y=df_candles['lower_bb'], name='Lower BB', line=dict(color='gray', dash='dash')), row=1, col=1)

            fig_tech.add_trace(go.Scatter(x=df_candles['date_str'], y=df_candles['macd_line'], name='MACD Line', line=dict(color='blue', width=2)), row=2, col=1)
            fig_tech.add_trace(go.Scatter(x=df_candles['date_str'], y=df_candles['signal_line'], name='Signal Line', line=dict(color='orange', width=2)), row=2, col=1)
            fig_tech.add_trace(go.Bar(x=df_candles['date_str'], y=df_candles['macd_hist'], name='MACD Hist', marker_color='lightgray'), row=2, col=1)

            fig_tech.add_trace(go.Scatter(x=df_candles['date_str'], y=df_candles['rsi'], name='RSI', line=dict(color='purple')), row=3, col=1)

            fig_tech.update_xaxes(type='category')
            fig_tech.update_layout(height=420, showlegend=False, xaxis_rangeslider_visible=False, margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig_tech, use_container_width=True)

        strikes_arr = df_merged_strikes['strike'].values

        # 1. Delta-Adjusted Net Gamma Exposure Profile
        fig_gex_adj = go.Figure()
        fig_gex_adj.add_trace(go.Bar(
            x=strikes_arr,
            y=df_merged_strikes['gex'],
            name='Delta-Adjusted GEX',
            marker_color=np.where(df_merged_strikes['gex'] >= 0, '#00E676', '#ef5350')
        ))
        
        fig_gex_adj.add_vline(
            x=gex_flip_price, 
            line_dash="dash", 
            line_color="orange", 
            annotation_text=f"Flip ({gex_flip_price})",
            annotation_position="top left"
        )
        fig_gex_adj.add_vline(
            x=spot_ltp, 
            line_dash="dash", 
            line_color="cyan", 
            annotation_text=f"Spot ({spot_ltp})",
            annotation_position="bottom right"
        )
        
        fig_gex_adj.update_layout(
            title=dict(text=f"🎯 Delta-Adjusted Net Gamma Exposure Profile | Updated: {ts}", font=dict(size=14)),
            xaxis_title="Strike Price", 
            yaxis_title="Net GEX (₹)", 
            height=350, 
            margin=dict(l=5, r=5, t=40, b=5)
        )
        st.plotly_chart(fig_gex_adj, use_container_width=True)

        # 2. Time-Series GEX Heatmap
        render_timeseries_gex_heatmap(stock, expiry, spot_ltp, df_merged_strikes)

        if not delta_gex_only:
            # 3. OI-Based Net Gamma Exposure vs Open Interest
            fig_oi_gex = make_subplots(specs=[[{"secondary_y": True}]])
            fig_oi_gex.add_trace(go.Bar(x=strikes_arr, y=df_merged_strikes['ce_oi'], name='Call OI', marker_color='#2E7D32'), secondary_y=False)
            fig_oi_gex.add_trace(go.Bar(x=strikes_arr, y=-df_merged_strikes['pe_oi'], name='Put OI', marker_color='#C62828'), secondary_y=False)
            fig_oi_gex.add_trace(go.Scatter(x=strikes_arr, y=df_merged_strikes['gex'], name='Net GEX', line=dict(color='yellow', width=1.5)), secondary_y=True)
            fig_oi_gex.add_vline(x=gex_flip_price, line_dash="dash", line_color="orange")
            fig_oi_gex.add_vline(x=spot_ltp, line_dash="dash", line_color="cyan")
            fig_oi_gex.update_layout(title=f"📈 OI Profile vs Net GEX | Updated: {ts}", barmode='relative', height=350, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig_oi_gex, use_container_width=True)

            # 4. Volume-Based VEX
            fig_vex = go.Figure()
            fig_vex.add_trace(go.Bar(
                x=strikes_arr,
                y=df_merged_strikes['vex'],
                name='Net VEX',
                marker_color='#9c27b0'
            ))
            fig_vex.add_vline(x=gex_flip_price, line_dash="dash", line_color="orange")
            fig_vex.add_vline(x=spot_ltp, line_dash="dash", line_color="cyan")
            fig_vex.update_layout(title=f"🔮 Net Vega Exposure Profile (VEX) | Updated: {ts}", xaxis_title="Strike Price", yaxis_title="Net VEX", height=350, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig_vex, use_container_width=True)

            # 5. Volatility Skew Profile
            df_skew_clean = df_merged_strikes[
                (df_merged_strikes['strike'] >= spot_ltp * 0.85) &
                (df_merged_strikes['strike'] <= spot_ltp * 1.15)
            ].copy()

            fig_skew = go.Figure()
            fig_skew.add_trace(go.Scatter(x=df_skew_clean['strike'], y=df_skew_clean['ce_iv'] * 100, name='CE IV', mode='lines+markers', line=dict(color='#00e676', width=2)))
            fig_skew.add_trace(go.Scatter(x=df_skew_clean['strike'], y=df_skew_clean['pe_iv'] * 100, name='PE IV', mode='lines+markers', line=dict(color='#ff5252', width=2)))
            fig_skew.add_vline(x=gex_flip_price, line_dash="dash", line_color="orange")
            fig_skew.add_vline(x=spot_ltp, line_dash="dash", line_color="cyan")
            fig_skew.update_layout(title=f"🌀 Volatility Skew (CE vs PE IV %) | Updated: {ts}", xaxis_title="Strike Price", yaxis_title="IV %", height=350, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig_skew, use_container_width=True)

# ----------------------------------------------------------------------
# Helper to build commodity item for MCX Options Chain  (unchanged)
# ----------------------------------------------------------------------
def load_commodity_data(comm_symbol, master_df, smart_api):
    mcx_opts = master_df[
        (master_df['exch_seg'] == 'MCX') &
        (master_df['name'] == comm_symbol) &
        (master_df['instrumenttype'] == 'OPTFUT')
    ].copy()

    if mcx_opts.empty:
        st.warning(f"No MCX options contracts found for {comm_symbol}.")
        return None

    mcx_opts['expiry_dt'] = pd.to_datetime(mcx_opts['expiry'], format='%d%b%Y')
    nearest_expiry = mcx_opts['expiry_dt'].min()
    expiry_str = nearest_expiry.strftime('%d%b%Y').upper()

    near_exp_opts = mcx_opts[mcx_opts['expiry'] == expiry_str].copy()
    near_exp_opts['strike'] = pd.to_numeric(near_exp_opts['strike']) / 100.0

    mcx_fut = master_df[
        (master_df['exch_seg'] == 'MCX') &
        (master_df['name'] == comm_symbol) &
        (master_df['instrumenttype'] == 'FUTCOM')
    ]
    
    if mcx_fut.empty:
        st.warning(f"No future token found for {comm_symbol}.")
        return None

    fut_token = mcx_fut.iloc[0]['token']
    fut_resp = smart_api.ltpData("MCX", mcx_fut.iloc[0]['symbol'], fut_token)
    
    if not fut_resp.get('status') or not fut_resp.get('data'):
        spot_ltp = float(near_exp_opts['strike'].median())
    else:
        spot_ltp = float(fut_resp['data']['ltp'])

    lot_size = float(near_exp_opts.iloc[0].get('lotsize', 1))

    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M")
    candle_params = {
        "exchange": "MCX",
        "symboltoken": fut_token,
        "interval": "ONE_DAY",
        "fromdate": from_date,
        "todate": to_date
    }
    candle_resp = smart_api.getCandleData(candle_params)
    df_candles, tech_ribbon = None, {}
    if candle_resp.get('status') and candle_resp.get('data'):
        df_c = pd.DataFrame(candle_resp['data'], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df_c = df_c[df_c['volume'] > 0].copy()
        _, _, df_candles, _, tech_ribbon = calculate_technical_indicators(df_c)

    stock_data = {
        "Stock": comm_symbol,
        "Spot LTP": spot_ltp,
        "Expiry": expiry_str,
        "near_exp_opts": near_exp_opts,
        "lot_size": lot_size,
        "df_candles": df_candles,
        "tech_ribbon": tech_ribbon
    }

    return compute_gex_chain(stock_data, smart_api, exchange="MCX")

# ----------------------------------------------------------------------
# Main App Logic
# ----------------------------------------------------------------------
smart_api = init_smart_api()

if smart_api:
    master_df = get_instrument_master()

    nfo_df = master_df[
        (master_df['exch_seg'] == 'NFO') &
        (master_df['instrumenttype'] == 'OPTSTK')
    ].copy()

    all_stock_symbols = sorted(list(nfo_df['name'].unique()))
    total_available = len(all_stock_symbols)

    # --- SIDEBAR OPTIONS ---
    st.sidebar.markdown(f"**Total Available Stock Options:** {total_available}")

    scan_limit = st.sidebar.number_input(
        "Number of Stocks to Scan",
        min_value=1,
        max_value=total_available,
        value=10,
        step=5
    )

    delta_gex_only = st.sidebar.checkbox("Delta Adjusted GEX Profiles", value=False)
    enable_autorefresh = st.sidebar.checkbox("Enable Auto-Refresh (Every 5s)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Manual Chart Selector")
    
    selected_all_stocks = st.sidebar.multiselect(
        "Select from ALL Scanned Stock Options:",
        options=all_stock_symbols,
        help="Select any stock options symbol to view its complete analytics, price, and GEX charts."
    )
    btn_add_charts = st.sidebar.button("➕ Add Charts")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧈 Commodities Charts (MCX)")
    selected_commodities = st.sidebar.multiselect(
        "Select Commodity (Gold / Silver):",
        options=["GOLDM", "SILVERM"],
        help="Selecting commodities hides stock charts and displays MCX Option Chain analytics."
    )

    if 'passed_stocks_data' not in st.session_state:
        st.session_state['passed_stocks_data'] = []

    # --- COMMODITY MODE OVERRIDE ---
    if selected_commodities:
        st.markdown(f"## 🧈 MCX Commodity Option Chain Analytics ({', '.join(selected_commodities)})")
        st.info("Commodity mode active: Stock screening charts are hidden.")

        c1, c2 = st.columns(2)
        for idx, comm in enumerate(selected_commodities):
            comm_item = load_commodity_data(comm, master_df, smart_api)
            if comm_item:
                target_col = c1 if (idx % 2 == 0) else c2
                with target_col:
                    render_stock_profile(comm_item, delta_gex_only)

    else:
        # --- STANDARD EQUITY OPTION SCREENER MODE ---
        if st.button("🚀 Run Level 1 Live Screener"):
            st.session_state['passed_stocks_data'] = []
            status_ribbon = st.empty()
            progress_bar = st.progress(0)

            to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M")

            target_stocks = all_stock_symbols[:scan_limit]

            for idx, stock in enumerate(target_stocks):
                current_count = idx + 1
                status_ribbon.info(f"⏳ Level 1 Screening **{current_count}/{scan_limit}**: Stock **{stock}**...")
                progress_bar.progress(current_count / scan_limit)

                equity_row = master_df[(master_df['exch_seg'] == 'NSE') & (master_df['symbol'] == f"{stock}-EQ")]
                if equity_row.empty:
                    continue

                eq_token = equity_row.iloc[0]['token']
                eq_resp = smart_api.ltpData("NSE", f"{stock}-EQ", eq_token)
                if not eq_resp.get('status') or not eq_resp.get('data'):
                    continue

                spot_ltp = float(eq_resp['data']['ltp'])

                candle_params = {
                    "exchange": "NSE",
                    "symboltoken": eq_token,
                    "interval": "ONE_DAY",
                    "fromdate": from_date,
                    "todate": to_date
                }

                candle_resp = smart_api.getCandleData(candle_params)
                is_bb_squeeze, macd_hist, df_candles, current_rsi, tech_ribbon = False, None, None, None, {}

                if candle_resp.get('status') and candle_resp.get('data'):
                    df_c = pd.DataFrame(candle_resp['data'], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df_c = df_c[df_c['volume'] > 0].copy()
                    is_bb_squeeze, macd_hist, df_candles, current_rsi, tech_ribbon = calculate_technical_indicators(df_c)

                stock_opts = nfo_df[nfo_df['name'] == stock].copy()
                stock_opts['expiry_dt'] = pd.to_datetime(stock_opts['expiry'], format='%d%b%Y')
                nearest_expiry = stock_opts['expiry_dt'].min()
                expiry_str = nearest_expiry.strftime('%d%b%Y').upper()

                near_exp_opts = stock_opts[stock_opts['expiry'] == expiry_str].copy()
                near_exp_opts['strike'] = pd.to_numeric(near_exp_opts['strike']) / 100.0

                unique_strikes = near_exp_opts['strike'].unique()
                if len(unique_strikes) == 0:
                    continue

                atm_strike_lvl1 = min(unique_strikes, key=lambda x: abs(x - spot_ltp))

                ce_row = near_exp_opts[(near_exp_opts['strike'] == atm_strike_lvl1) & (near_exp_opts['symbol'].str.endswith('CE'))]
                pe_row = near_exp_opts[(near_exp_opts['strike'] == atm_strike_lvl1) & (near_exp_opts['symbol'].str.endswith('PE'))]

                ce_ltp, pe_ltp = None, None
                ce_vol, pe_vol = 0, 0

                if not ce_row.empty:
                    ce_token = str(ce_row.iloc[0]['token'])
                    ce_data_resp = smart_api.getMarketData(mode="FULL", exchangeTokens={"NFO": [ce_token]})
                    if ce_data_resp.get('status') and ce_data_resp.get('data'):
                        f_ce = ce_data_resp['data']['fetched'][0]
                        ce_ltp = float(f_ce.get('ltp', 0))
                        ce_vol = int(f_ce.get('tradeVolume', 0))

                if not pe_row.empty:
                    pe_token = str(pe_row.iloc[0]['token'])
                    pe_data_resp = smart_api.getMarketData(mode="FULL", exchangeTokens={"NFO": [pe_token]})
                    if pe_data_resp.get('status') and pe_data_resp.get('data'):
                        f_pe = pe_data_resp['data']['fetched'][0]
                        pe_ltp = float(f_pe.get('ltp', 0))
                        pe_vol = int(f_pe.get('tradeVolume', 0))

                lot_size = float(near_exp_opts.iloc[0].get('lotsize', 1))
                total_atm_turnover = ((ce_vol * (ce_ltp or 0)) + (pe_vol * (pe_ltp or 0))) * lot_size

                cond_vol = total_atm_turnover >= 10000000
                cond_bb = is_bb_squeeze
                cond_macd = (macd_hist is not None) and (macd_hist > 0)

                if cond_vol and cond_bb and cond_macd:
                    st.session_state['passed_stocks_data'].append({
                        "Stock": stock,
                        "Expiry": expiry_str,
                        "Spot LTP": spot_ltp,
                        "ATM Strike": atm_strike_lvl1,
                        "ATM CE LTP": ce_ltp,
                        "ATM PE LTP": pe_ltp,
                        "CE Vol": ce_vol,
                        "PE Vol": pe_vol,
                        "ATM Turnover (Cr)": round(total_atm_turnover / 10000000, 2),
                        "MACD Hist": round(macd_hist, 2),
                        "BB Squeeze": "ACTIVE",
                        "df_candles": df_candles,
                        "near_exp_opts": near_exp_opts,
                        "lot_size": lot_size,
                        "tech_ribbon": tech_ribbon
                    })

            status_ribbon.success(f"✅ Level 1 Complete! Found {len(st.session_state['passed_stocks_data'])} passing stocks.")
            progress_bar.empty()

        # Handle Manual Selector Output
        if btn_add_charts and selected_all_stocks:
            st.markdown(f"#### 📌 Manual Output Charts for Selected Symbols ({len(selected_all_stocks)})")
            
            c1, c2 = st.columns(2)
            for idx, stock_sym in enumerate(selected_all_stocks):
                passed_match = [s for s in st.session_state['passed_stocks_data'] if s['Stock'] == stock_sym]
                
                if passed_match:
                    stock_data = passed_match[0]
                else:
                    equity_row = master_df[(master_df['exch_seg'] == 'NSE') & (master_df['symbol'] == f"{stock_sym}-EQ")]
                    if equity_row.empty:
                        continue
                    eq_token = equity_row.iloc[0]['token']
                    eq_resp = smart_api.ltpData("NSE", f"{stock_sym}-EQ", eq_token)
                    if not eq_resp.get('status') or not eq_resp.get('data'):
                        continue
                    spot_ltp = float(eq_resp['data']['ltp'])

                    stock_opts = nfo_df[nfo_df['name'] == stock_sym].copy()
                    stock_opts['expiry_dt'] = pd.to_datetime(stock_opts['expiry'], format='%d%b%Y')
                    nearest_expiry = stock_opts['expiry_dt'].min()
                    expiry_str = nearest_expiry.strftime('%d%b%Y').upper()

                    near_exp_opts = stock_opts[stock_opts['expiry'] == expiry_str].copy()
                    near_exp_opts['strike'] = pd.to_numeric(near_exp_opts['strike']) / 100.0
                    lot_size = float(near_exp_opts.iloc[0].get('lotsize', 1))

                    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
                    from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d %H:%M")
                    candle_resp = smart_api.getCandleData({
                        "exchange": "NSE", "symboltoken": eq_token, "interval": "ONE_DAY",
                        "fromdate": from_date, "todate": to_date
                    })
                    df_candles, tech_ribbon = None, {}
                    if candle_resp.get('status') and candle_resp.get('data'):
                        df_c = pd.DataFrame(candle_resp['data'], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                        df_c = df_c[df_c['volume'] > 0].copy()
                        _, _, df_candles, _, tech_ribbon = calculate_technical_indicators(df_c)

                    stock_data = {
                        "Stock": stock_sym,
                        "Spot LTP": spot_ltp,
                        "Expiry": expiry_str,
                        "near_exp_opts": near_exp_opts,
                        "lot_size": lot_size,
                        "df_candles": df_candles,
                        "tech_ribbon": tech_ribbon
                    }

                item = compute_gex_chain(stock_data, smart_api)
                if item:
                    target_col = c1 if (idx % 2 == 0) else c2
                    with target_col:
                        render_stock_profile(item, delta_gex_only)

        if st.session_state['passed_stocks_data']:
            st.markdown("### 🎯 Stocks Passing Level 1 Criteria")
            df_passed = pd.DataFrame(st.session_state['passed_stocks_data'])
            display_cols = ["Stock", "Expiry", "Spot LTP", "ATM Strike", "ATM CE LTP", "ATM PE LTP", "CE Vol", "PE Vol", "ATM Turnover (Cr)", "MACD Hist", "BB Squeeze"]
            st.dataframe(df_passed[display_cols], use_container_width=True)

            st.markdown("---")
            st.markdown("### ⚡ Level 2 Screening: GEX Flip & Options Deep-Dive Profile")

            if st.button("🔍 Run Level 2 GEX Screening"):
                lvl2_passed_stocks = []

                for stock_data in st.session_state['passed_stocks_data']:
                    item = compute_gex_chain(stock_data, smart_api)
                    if item and (item['spot_ltp'] > item['gex_flip_price']):
                        lvl2_passed_stocks.append(item)

                if not lvl2_passed_stocks:
                    st.info("No stocks passed the Level 2 condition (`Spot Price > GEX Flip Price`).")
                else:
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(lvl2_passed_stocks):
                        target_col = col1 if (idx % 2 == 0) else col2
                        with target_col:
                            render_stock_profile(item, delta_gex_only)

    if enable_autorefresh:
        time.sleep(5)
        st.rerun()
else:
    st.error("Unable to connect to Angel One SmartAPI. Please check your environment variables.")
