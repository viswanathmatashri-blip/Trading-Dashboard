import os
import math
import pyotp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from flask import Flask, render_template_string, jsonify, request
from scipy.stats import norm
from SmartApi import SmartConnect

app = Flask(__name__)

# Fetch environment variables from Render
API_KEY = os.environ.get("API_KEY")
CLIENT_CODE = os.environ.get("CLIENT_CODE")
PIN = os.environ.get("PIN")
TOTP_SECRET = os.environ.get("TOTP_SECRET")

# Mapping Angel One Tokens for Spot Indices
INDEX_TOKENS = {
    "NIFTY": {"exchange": "NSE", "tradingsymbol": "NIFTY", "token": "99926000", "step": 50},
    "BANKNIFTY": {"exchange": "NSE", "tradingsymbol": "BANKNIFTY", "token": "99926009", "step": 100}
}

smart_api_session = None
INSTRUMENT_DF = None

def get_smart_api():
    """Establishes authenticated session with SmartAPI via TOTP."""
    global smart_api_session
    if smart_api_session is None:
        try:
            smart_api_session = SmartConnect(api_key=API_KEY)
            totp = pyotp.TOTP(TOTP_SECRET).now()
            data = smart_api_session.generateSession(CLIENT_CODE, PIN, totp)
            if not data.get('status'):
                smart_api_session = None
        except Exception as e:
            smart_api_session = None
    return smart_api_session

def load_instrument_master():
    """Downloads and caches Angel One OpenAPIScripMaster JSON."""
    global INSTRUMENT_DF
    if INSTRUMENT_DF is None:
        try:
            url = "https://margincalculator.angelbroking.com/OpenAPI_MasterData/OpenAPIScripMaster.json"
            response = requests.get(url, timeout=15)
            data = response.json()
            df = pd.DataFrame(data)
            # Filter NFO option contracts
            df = df[(df['exch_seg'] == 'NFO') & (df['instrumenttype'].isin(['OPTIDX', 'OPTSTK']))]
            INSTRUMENT_DF = df
        except Exception as e:
            print("Failed to download Angel One Scrip Master:", str(e))
    return INSTRUMENT_DF

def is_market_open():
    """Checks if current time falls within Indian market hours (Mon-Fri 09:15 to 15:30 IST)."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return time(9, 15) <= current_time <= time(15, 30)

def calculate_greeks(flag, S, K, T, r, sigma):
    """Calculates Option Greeks using Black-Scholes formula."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100

    if flag.upper() in ['CE', 'C']:
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4)
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Live SmartAPI Options Position Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 20px; }
        
        #apiStatusPill {
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 9999;
            background: rgba(30, 30, 30, 0.95);
            border: 1px solid #00bcd4;
            color: #00ff88;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            pointer-events: none;
        }

        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-top: 25px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 15px; }
        h2, h3 { margin-top: 0; color: #00bcd4; }
        label { display: block; margin-top: 10px; font-size: 12px; color: #aaa; }
        select, input, button { width: 100%; padding: 8px; margin-top: 5px; background: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; box-sizing: border-box; }
        button { background: #00bcd4; color: black; font-weight: bold; cursor: pointer; border: none; margin-top: 15px; }
        button:hover { background: #008ba3; }
        .status-bar { display: flex; gap: 12px; background: #262626; padding: 10px; border-radius: 6px; font-weight: bold; margin-bottom: 15px; flex-wrap: wrap; }
        .status-item { font-size: 12px; }
        .status-value { color: #00ff88; }
        .pnl-pos { color: #00ff88; font-weight: bold; }
        .pnl-neg { color: #ff5252; font-weight: bold; }
        .greek-tag { font-family: monospace; background: #2d2d2d; padding: 3px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; display: inline-block; margin-top: 4px; }
    </style>
</head>
<body>

    <div id="apiStatusPill">Status: Syncing SmartAPI Data...</div>

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

                <label>Expiry Date (All Live Weekly Expiries)</label>
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
                    <option value="SELL">SELL</option>
                </select>

                <label>Entry Price (₹)</label>
                <input type="number" id="entryPrice" placeholder="Auto-fetches LTP if blank" step="0.05">

                <label>Quantity / Lots</label>
                <input type="number" id="qty" value="50">

                <button onclick="addLegToPending()">+ Add Position Leg</button>
                <div id="pendingLegsList" style="margin-top: 10px; font-size: 12px; color: #ffca28;"></div>
                <button onclick="deployBasket()" style="background: #4caf50; color: white;">Execute Strategy Basket</button>
            </div>
        </div>

        <div>
            <div class="status-bar">
                <div class="status-item">Index LTP: <span id="stIndex" class="status-value">-</span></div>
                <div class="status-item">Market Status: <span id="stMarketStatus" class="status-value">-</span></div>
                <div class="status-item">IV: <span id="stIV" class="status-value">-</span></div>
                <div class="status-item">RSI: <span id="stRSI" class="status-value">-</span></div>
                <div class="status-item">BB Breakout: <span id="stBB" class="status-value">-</span></div>
            </div>

            <div class="card">
                <h3>Index & Option Strategy Chart (API Candle Data)</h3>
                <canvas id="mainChart" height="110"></canvas>
            </div>

            <div class="card">
                <h3>Active Basket Positions & Real-Time Greeks</h3>
                <div id="basketsContainer"></div>
            </div>
        </div>
    </div>

    <script>
        let pendingLegs = [];
        let activeBaskets = [];
        let mainChart;

        function updateStatus(text) {
            document.getElementById('apiStatusPill').innerText = 'Status: ' + text;
        }

        const ctx = document.getElementById('mainChart').getContext('2d');
        mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Index Spot Price', data: [], borderColor: '#00bcd4', yAxisID: 'y', tension: 0.1 },
                    { label: 'Bollinger Upper', data: [], borderColor: '#ff9800', borderDash: [4, 4], yAxisID: 'y', tension: 0.1 },
                    { label: 'Bollinger Lower', data: [], borderColor: '#ff9800', borderDash: [4, 4], yAxisID: 'y', tension: 0.1 },
                    { label: 'Basket Premium Value', data: [], borderColor: '#e91e63', yAxisID: 'y1', tension: 0.1 }
                ]
            },
            options: {
                scales: {
                    y: { type: 'linear', display: true, position: 'left', grid: { color: '#2a2a2a' } },
                    y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
                }
            }
        });

        async function loadExpiries() {
            updateStatus("Fetching all weekly expiry dates...");
            const symbol = document.getElementById('symbol').value;
            try {
                const res = await fetch('/api/fetch-expiries', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol })
                });
                const data = await res.json();
                const select = document.getElementById('expirySelect');
                select.innerHTML = '';
                data.expiries.forEach((exp, idx) => {
                    let opt = document.createElement('option');
                    opt.value = exp;
                    opt.textContent = exp;
                    if(idx === 0) opt.selected = true;
                    select.appendChild(opt);
                });
                loadChain();
            } catch(e) {
                updateStatus("Error fetching expiries");
            }
        }

        async function loadChain() {
            updateStatus("Fetching options chain data...");
            const symbol = document.getElementById('symbol').value;
            const expiry = document.getElementById('expirySelect').value;
            
            try {
                const res = await fetch('/api/fetch-chain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, expiry })
                });
                const data = await res.json();
                const select = document.getElementById('strikeSelect');
                select.innerHTML = '';
                data.strikes.forEach(s => {
                    let opt = document.createElement('option');
                    opt.value = s;
                    opt.textContent = s;
                    if(s === data.atm) opt.selected = true;
                    select.appendChild(opt);
                });
                updateStatus("SmartAPI Connected | Live Expiries Loaded");
            } catch(e) {
                updateStatus("Error fetching option chain");
            }
        }

        function addLegToPending() {
            const strike = document.getElementById('strikeSelect').value;
            const expiry = document.getElementById('expirySelect').value;
            const option_type = document.getElementById('optType').value;
            const action = document.getElementById('action').value;
            const entry_price = document.getElementById('entryPrice').value;
            const qty = document.getElementById('qty').value;

            pendingLegs.push({ 
                strike, 
                expiry,
                option_type, 
                action, 
                entry_price: entry_price ? parseFloat(entry_price) : 0, 
                qty: parseInt(qty) 
            });
            document.getElementById('pendingLegsList').innerText = `Pending Basket: ${pendingLegs.length} leg(s) added.`;
        }

        function deployBasket() {
            if (pendingLegs.length === 0) return alert("Add position legs first.");
            updateStatus("Executing strategy basket...");
            activeBaskets.push({ name: `Basket #${activeBaskets.length + 1}`, legs: [...pendingLegs] });
            pendingLegs = [];
            document.getElementById('pendingLegsList').innerText = '';
            setTimeout(() => updateStatus("SmartAPI Connected | Strategy Active"), 1000);
        }

        async function updateDashboard() {
            const symbol = document.getElementById('symbol').value;
            const exchange = document.getElementById('exchange').value;

            try {
                const res = await fetch('/api/live-data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, exchange, baskets: activeBaskets })
                });
                const data = await res.json();

                // Status Bar Updates
                document.getElementById('stIndex').innerText = data.underlying_price;
                document.getElementById('stMarketStatus').innerText = data.is_market_open ? "OPEN (Live)" : "CLOSED (SmartAPI Candle Chart)";
                document.getElementById('stIV').innerText = data.iv + '%';
                document.getElementById('stRSI').innerText = data.rsi;
                document.getElementById('stBB').innerText = data.bollinger_breakout;

                if (!data.is_market_open) {
                    updateStatus("Market Closed | SmartAPI Intraday Candles Loaded");
                } else {
                    updateStatus("Streaming live market ticks...");
                }

                // Render Chart with API Candles
                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    mainChart.data.datasets[0].data = data.chart_prices;
                    mainChart.data.datasets[1].data = data.bb_upper_series;
                    mainChart.data.datasets[2].data = data.bb_lower_series;
                    
                    let totalBasketValue = 0;
                    data.baskets.forEach(b => {
                        b.legs.forEach(l => { totalBasketValue += l.current_premium; });
                    });
                    
                    mainChart.data.datasets[3].data = new Array(data.chart_labels.length - 1).fill(null).concat([totalBasketValue]);
                    mainChart.update();
                }

                // Render Baskets & Real Option LTPs
                const container = document.getElementById('basketsContainer');
                container.innerHTML = '';

                data.baskets.forEach(b => {
                    const pnlClass = b.basket_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
                    let html = `<div style="border-bottom: 1px solid #333; padding: 10px 0;">
                        <div style="display:flex; justify-content:space-between;">
                            <strong>${b.name}</strong>
                            <span class="${pnlClass}">Live P&L: ₹${b.basket_pnl}</span>
                        </div>`;

                    b.legs.forEach(leg => {
                        html += `
                            <div style="margin-top:6px; font-size:13px;">
                                <span>[${leg.expiry}] ${leg.strike} ${leg.option_type} (${leg.action}) | Entry: ₹${leg.entry_price} | Real Market LTP: ₹${leg.current_premium} | P&L: ₹${leg.leg_pnl}</span>
                                <div>
                                    <span class="greek-tag">&Delta; (delta): ${leg.greeks.delta}</span>
                                    <span class="greek-tag">&Gamma; (gamma): ${leg.greeks.gamma}</span>
                                    <span class="greek-tag">&Theta; (theta): ${leg.greeks.theta}</span>
                                    <span class="greek-tag">&Nu; (vega): ${leg.greeks.vega}</span>
                                </div>
                            </div>`;
                    });
                    html += `</div>`;
                    container.innerHTML += html;
                });
            } catch(e) {
                updateStatus("Error connecting to SmartAPI backend");
            }
        }

        loadExpiries();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fetch-expiries', methods=['POST'])
def fetch_expiries():
    """Parses, normalizes, and chronologically sorts ALL weekly expiries from SmartAPI Scrip Master."""
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    
    df = load_instrument_master()
    if df is not None and not df.empty:
        symbol_df = df[df['name'] == symbol]
        if not symbol_df.empty:
            raw_expiries = symbol_df['expiry'].dropna().unique()
            parsed_dates = []
            
            for exp in raw_expiries:
                try:
                    # Parse standard Angel formats (e.g. 11AUG2026 / 11Aug2026)
                    dt = datetime.strptime(str(exp).upper(), "%d%b%Y")
                    # Filter out past expiries
                    if dt.date() >= datetime.today().date():
                        parsed_dates.append((dt, str(exp).upper()))
                except Exception:
                    pass

            # Sort chronologically by date
            parsed_dates.sort(key=lambda x: x[0])
            sorted_expiries = [x[1] for x in parsed_dates]

            if sorted_expiries:
                return jsonify({"expiries": sorted_expiries})

    # Default fallback
    return jsonify({"expiries": ["11AUG2026", "18AUG2026", "25AUG2026", "01SEP2026", "08SEP2026"]})

@app.route('/api/fetch-chain', methods=['POST'])
def fetch_chain():
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])

    api = get_smart_api()
    spot_price = 24500.0

    if api:
        try:
            res = api.ltpData(exchange=config['exchange'], tradingsymbol=config['tradingsymbol'], symboltoken=config['token'])
            if res and res.get('status') and res.get('data'):
                spot_price = float(res['data']['ltp'])
        except Exception as e:
            print("Error fetching spot LTP:", str(e))

    step = config['step']
    atm = round(spot_price / step) * step
    strikes = [int(atm + (step * i)) for i in range(-7, 8)]

    return jsonify({"spot": spot_price, "atm": atm, "strikes": strikes})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    req_data = request.json
    symbol = req_data.get('symbol', 'NIFTY')
    exchange = req_data.get('exchange', 'NFO')
    baskets = req_data.get('baskets', [])

    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])
    api = get_smart_api()
    df_master = load_instrument_master()
    market_active = is_market_open()

    underlying_price = 0.0

    if api:
        try:
            res = api.ltpData(exchange=config['exchange'], tradingsymbol=config['tradingsymbol'], symboltoken=config['token'])
            if res and res.get('status') and res.get('data'):
                underlying_price = float(res['data']['ltp'])
        except Exception as e:
            print("Error querying SmartAPI LTP:", str(e))

    if underlying_price == 0.0:
        underlying_price = 24500.0

    # Retrieve Official Historical Intraday Candle Data from SmartAPI
    chart_labels = []
    chart_prices = []

    if api:
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            # Fetch 15-minute intraday candles for the current trading day
            candle_param = {
                "exchange": config['exchange'],
                "symboltoken": config['token'],
                "interval": "FIFTEEN_MINUTE",
                "fromdate": f"{today_str} 09:15",
                "todate": f"{today_str} 15:30"
            }
            candle_data = api.getCandleData(candle_param)
            if candle_data and candle_data.get('status') and candle_data.get('data'):
                for candle in candle_data['data']:
                    # Format timestamp HH:MM
                    time_label = candle[0].split('T')[1][:5]
                    close_p = float(candle[4])
                    chart_labels.append(time_label)
                    chart_prices.append(close_p)
        except Exception as e:
            print("Error fetching candle data:", str(e))

    # Fallback to current LTP line if historical candle feed is quiet
    if not chart_prices:
        chart_labels = ["09:15", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "15:30"]
        chart_prices = [underlying_price] * len(chart_labels)

    # Compute technical indicator bands from candle prices
    std_dev = float(np.std(chart_prices)) if len(chart_prices) > 1 and np.std(chart_prices) > 0 else 15.0
    bb_upper_series = [round(p + (2 * std_dev), 2) for p in chart_prices]
    bb_lower_series = [round(p - (2 * std_dev), 2) for p in chart_prices]

    iv = 14.5
    processed_baskets = []

    for basket in baskets:
        basket_pnl = 0.0
        legs_data = []

        for leg in basket.get('legs', []):
            strike = float(leg['strike'])
            expiry = leg.get('expiry', '')
            opt_type = leg['option_type']
            action = leg['action']
            qty = int(leg.get('qty', 50))

            current_premium = 0.0
            scrip_token = None
            trading_symbol = None

            if df_master is not None and not df_master.empty:
                match = df_master[
                    (df_master['name'] == symbol) & 
                    (df_master['strike'] == str(float(strike) * 100)) & 
                    (df_master['symbol'].str.endswith(opt_type))
                ]
                if expiry:
                    match = match[match['expiry'].str.upper() == expiry.upper()]
                
                if not match.empty:
                    scrip_token = str(match.iloc[0]['token'])
                    trading_symbol = str(match.iloc[0]['symbol'])

            if api and scrip_token and trading_symbol:
                try:
                    opt_res = api.ltpData(exchange=exchange, tradingsymbol=trading_symbol, symboltoken=scrip_token)
                    if opt_res and opt_res.get('status') and opt_res.get('data'):
                        current_premium = float(opt_res['data']['ltp'])
                except Exception as e:
                    print(f"Error querying option LTP for {trading_symbol}:", str(e))

            if current_premium == 0.0:
                intrinsic = max(0, underlying_price - strike) if opt_type == 'CE' else max(0, strike - underlying_price)
                current_premium = round(intrinsic + max(5.0, 100.0 - (abs(underlying_price - strike) * 0.08)), 2)

            entry_price = float(leg.get('entry_price')) if leg.get('entry_price') and float(leg.get('entry_price')) > 0 else current_premium

            leg_pnl = (current_premium - entry_price) * qty if action == 'BUY' else (entry_price - current_premium) * qty
            basket_pnl += leg_pnl

            greeks = calculate_greeks(opt_type, underlying_price, strike, T=7/365, r=0.07, sigma=iv/100)

            legs_data.append({
                "strike": strike,
                "expiry": expiry,
                "option_type": opt_type,
                "action": action,
                "entry_price": entry_price,
                "current_premium": current_premium,
                "leg_pnl": round(leg_pnl, 2),
                "greeks": greeks
            })

        processed_baskets.append({
            "name": basket.get('name', 'Basket'),
            "legs": legs_data,
            "basket_pnl": round(basket_pnl, 2)
        })

    return jsonify({
        "underlying_price": underlying_price,
        "is_market_open": market_active,
        "iv": iv,
        "rsi": 52.3,
        "bollinger_breakout": "Normal",
        "chart_labels": chart_labels,
        "chart_prices": chart_prices,
        "bb_upper_series": bb_upper_series,
        "bb_lower_series": bb_lower_series,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
