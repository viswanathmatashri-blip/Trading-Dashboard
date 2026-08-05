import os
import math
import pyotp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
from flask import Flask, render_template_string, jsonify, request
from scipy.stats import norm
from SmartApi import SmartConnect

app = Flask(__name__)

# Fetch environment variables from Render
API_KEY = os.environ.get("API_KEY")
CLIENT_CODE = os.environ.get("CLIENT_CODE")
PIN = os.environ.get("PIN")
TOTP_SECRET = os.environ.get("TOTP_SECRET")

INDEX_TOKENS = {
    "NIFTY": {"exchange": "NSE", "tradingsymbol": "NIFTY", "token": "99926000", "step": 50},
    "BANKNIFTY": {"exchange": "NSE", "tradingsymbol": "BANKNIFTY", "token": "99926009", "step": 100}
}

smart_api_session = None
INSTRUMENT_DF = None

def get_smart_api():
    """Authenticates with SmartAPI, renewing expired tokens automatically."""
    global smart_api_session
    
    if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
        return None, "Missing Render Env Variables (API_KEY/CLIENT_CODE/PIN/TOTP_SECRET)"

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

def load_instrument_master():
    """Downloads Angel One OpenAPIScripMaster JSON with browser headers and handles retries."""
    global INSTRUMENT_DF
    if INSTRUMENT_DF is None:
        url = "https://margincalculator.angelbroking.com/OpenAPI_MasterData/OpenAPIScripMaster.json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*"
        }
        
        for attempt in range(1, 4):
            try:
                print(f"[Scrip Master] Downloading JSON (Attempt {attempt}/3)...")
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    
                    # Keep only NFO options to save memory
                    df = df[(df['exch_seg'] == 'NFO') & (df['instrumenttype'].isin(['OPTIDX', 'OPTSTK']))].copy()
                    df['strike_price'] = df['strike'].astype(float) / 100.0
                    
                    INSTRUMENT_DF = df
                    print(f"[Scrip Master] Successfully loaded {len(INSTRUMENT_DF)} option contracts.")
                    break
                else:
                    print(f"[Scrip Master] HTTP Error {response.status_code}")
            except Exception as e:
                print(f"[Scrip Master] Download error on attempt {attempt}: {str(e)}")

    return INSTRUMENT_DF

def is_market_open():
    """Checks if current time falls within Indian market hours (Mon-Fri 09:15 to 15:30 IST)."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return time(9, 15) <= current_time <= time(15, 30)

def calculate_greeks(flag, S, K, T, r, sigma):
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
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #121212; color: #e0e0e0; margin: 20px; padding-top: 35px; }
        
        #apiStatusPill {
            position: fixed;
            top: 10px;
            left: 15px;
            z-index: 9999;
            background: rgba(20, 20, 20, 0.95);
            border: 1px solid #00bcd4;
            color: #00ff88;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            max-width: 90vw;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .pill-err { border-color: #ff5252 !important; color: #ff5252 !important; }
        .pill-warn { border-color: #ffca28 !important; color: #ffca28 !important; }

        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-top: 15px; }
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
        .pnl-err { color: #ff9800; font-weight: bold; }
        .greek-tag { font-family: monospace; background: #2d2d2d; padding: 3px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; display: inline-block; margin-top: 4px; }
    </style>
</head>
<body>

    <div id="apiStatusPill">Status: Logging in API...</div>

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
            </div>

            <div class="card">
                <h3>Index Strategy Chart (SmartAPI 15-Min Candles)</h3>
                <canvas id="mainChart" height="110"></canvas>
            </div>

            <div class="card">
                <h3>Active Basket Positions & Real-Time Option LTP</h3>
                <div id="basketsContainer"></div>
            </div>
        </div>
    </div>

    <script>
        let pendingLegs = [];
        let activeBaskets = [];
        let mainChart;

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
                    { label: 'Index Spot Price', data: [], borderColor: '#00bcd4', yAxisID: 'y', tension: 0.1 }
                ]
            },
            options: {
                scales: {
                    y: { type: 'linear', display: true, position: 'left', grid: { color: '#2a2a2a' } }
                }
            }
        });

        async function loadExpiries() {
            updateStatus("Logging in API...", "warn");
            const symbol = document.getElementById('symbol').value;
            try {
                const res = await fetch('/api/fetch-expiries', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol })
                });
                const data = await res.json();
                
                if (data.status) {
                    updateStatus(data.status, data.status === "API connection success" ? "info" : "error");
                }

                const select = document.getElementById('expirySelect');
                select.innerHTML = '';
                if(data.expiries && data.expiries.length > 0) {
                    data.expiries.forEach((exp, idx) => {
                        let opt = document.createElement('option');
                        opt.value = exp;
                        opt.textContent = exp;
                        if(idx === 0) opt.selected = true;
                        select.appendChild(opt);
                    });
                    loadChain();
                }
            } catch(e) {
                updateStatus("Couldnt log in: Network Error", "error");
            }
        }

        async function loadChain() {
            const symbol = document.getElementById('symbol').value;
            const expiry = document.getElementById('expirySelect').value;
            
            try {
                const res = await fetch('/api/fetch-chain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, expiry })
                });
                const data = await res.json();
                
                if (data.status) {
                    updateStatus(data.status, data.status === "API connection success" ? "info" : "error");
                }

                const select = document.getElementById('strikeSelect');
                select.innerHTML = '';
                if(data.strikes && data.strikes.length > 0) {
                    data.strikes.forEach(s => {
                        let opt = document.createElement('option');
                        opt.value = s;
                        opt.textContent = s;
                        if(s === data.atm) opt.selected = true;
                        select.appendChild(opt);
                    });
                }
            } catch(e) {
                updateStatus("Couldnt fetch live price from API", "error");
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
                entry_price: entry_price ? parseFloat(entry_price) : null, 
                qty: parseInt(qty) 
            });
            document.getElementById('pendingLegsList').innerText = `Pending Basket: ${pendingLegs.length} leg(s) added.`;
        }

        function deployBasket() {
            if (pendingLegs.length === 0) return alert("Add position legs first.");
            activeBaskets.push({ name: `Basket #${activeBaskets.length + 1}`, legs: [...pendingLegs] });
            pendingLegs = [];
            document.getElementById('pendingLegsList').innerText = '';
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

                if (data.status) {
                    updateStatus(data.status, data.status === "API connection success" ? "info" : "error");
                }

                if (data.error) {
                    document.getElementById('stIndex').innerText = data.error;
                    return;
                }

                document.getElementById('stIndex').innerText = data.underlying_price;
                document.getElementById('stMarketStatus').innerText = data.is_market_open ? "OPEN (Live)" : "CLOSED";

                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    mainChart.data.datasets[0].data = data.chart_prices;
                    mainChart.update();
                }

                const container = document.getElementById('basketsContainer');
                container.innerHTML = '';

                data.baskets.forEach(b => {
                    let pnlDisplay = typeof b.basket_pnl === 'number' ? `₹${b.basket_pnl}` : b.basket_pnl;
                    let pnlClass = typeof b.basket_pnl === 'number' ? (b.basket_pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : 'pnl-err';

                    let html = `<div style="border-bottom: 1px solid #333; padding: 10px 0;">
                        <div style="display:flex; justify-content:space-between;">
                            <strong>${b.name}</strong>
                            <span class="${pnlClass}">Live P&L: ${pnlDisplay}</span>
                        </div>`;

                    b.legs.forEach(leg => {
                        let ltpText = typeof leg.current_premium === 'number' ? `₹${leg.current_premium}` : leg.current_premium;
                        let entryText = typeof leg.entry_price === 'number' ? `₹${leg.entry_price}` : leg.entry_price;
                        let legPnlText = typeof leg.leg_pnl === 'number' ? `₹${leg.leg_pnl}` : leg.leg_pnl;

                        html += `
                            <div style="margin-top:6px; font-size:13px;">
                                <span>[${leg.expiry}] ${leg.strike} ${leg.option_type} (${leg.action}) | Entry: ${entryText} | Real LTP: ${ltpText} | P&L: ${legPnlText}</span>
                                <div>
                                    <span class="greek-tag">&Delta;: ${leg.greeks.delta}</span>
                                    <span class="greek-tag">&Gamma;: ${leg.greeks.gamma}</span>
                                    <span class="greek-tag">&Theta;: ${leg.greeks.theta}</span>
                                    <span class="greek-tag">&Nu;: ${leg.greeks.vega}</span>
                                </div>
                            </div>`;
                    });
                    html += `</div>`;
                    container.innerHTML += html;
                });
            } catch(e) {
                updateStatus("Couldnt fetch live price from API", "error");
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
    api, status_msg = get_smart_api()
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
                    dt = datetime.strptime(str(exp).upper(), "%d%b%Y")
                    if dt.date() >= datetime.today().date():
                        parsed_dates.append((dt, str(exp).upper()))
                except Exception:
                    pass

            parsed_dates.sort(key=lambda x: x[0])
            sorted_expiries = [x[1] for x in parsed_dates]

            return jsonify({"status": status_msg, "expiries": sorted_expiries})

    return jsonify({"status": status_msg, "expiries": []})

@app.route('/api/fetch-chain', methods=['POST'])
def fetch_chain():
    api, status_msg = get_smart_api()
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])

    if not api:
        return jsonify({"status": status_msg, "strikes": []})

    try:
        res = api.ltpData(exchange=config['exchange'], tradingsymbol=config['tradingsymbol'], symboltoken=config['token'])
        if res and res.get('status') and res.get('data'):
            spot_price = float(res['data']['ltp'])
            step = config['step']
            atm = round(spot_price / step) * step
            strikes = [int(atm + (step * i)) for i in range(-10, 11)]
            return jsonify({"status": status_msg, "spot": spot_price, "atm": atm, "strikes": strikes})
        else:
            msg = res.get('message', 'Failed to fetch LTP') if res else 'No response'
            return jsonify({"status": f"Couldnt fetch live price: {msg}", "strikes": []})
    except Exception as e:
        return jsonify({"status": f"Couldnt fetch live price: {str(e)}", "strikes": []})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    api, status_msg = get_smart_api()
    req_data = request.json
    symbol = req_data.get('symbol', 'NIFTY')
    exchange = req_data.get('exchange', 'NFO')
    baskets = req_data.get('baskets', [])

    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])
    df_master = load_instrument_master()

    if not api:
        return jsonify({"status": status_msg, "error": status_msg})

    if df_master is None or df_master.empty:
        return jsonify({"status": "Scrip Master Not Loaded", "error": "Scrip Master Not Loaded"})

    underlying_price = None
    try:
        res = api.ltpData(exchange=config['exchange'], tradingsymbol=config['tradingsymbol'], symboltoken=config['token'])
        if res and res.get('status') and res.get('data'):
            underlying_price = float(res['data']['ltp'])
    except Exception as e:
        print("Error querying SmartAPI LTP:", str(e))

    if underlying_price is None:
        return jsonify({"status": "Couldnt fetch live price from API", "error": "couldnt fetch live price from API"})

    chart_labels = []
    chart_prices = []
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
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
                time_label = candle[0].split('T')[1][:5]
                close_p = float(candle[4])
                chart_labels.append(time_label)
                chart_prices.append(close_p)
    except Exception as e:
        print("Error fetching candle data:", str(e))

    processed_baskets = []

    for basket in baskets:
        basket_pnl = 0.0
        legs_data = []
        has_leg_error = False

        for leg in basket.get('legs', []):
            strike = float(leg['strike'])
            expiry = leg.get('expiry', '').upper()
            opt_type = leg['option_type']
            action = leg['action']
            qty = int(leg.get('qty', 50))

            current_premium = None
            scrip_token = None
            trading_symbol = None

            match = df_master[
                (df_master['name'] == symbol) & 
                (abs(df_master['strike_price'] - strike) < 0.01) & 
                (df_master['symbol'].str.endswith(opt_type)) &
                (df_master['expiry'].str.upper() == expiry)
            ]

            if not match.empty:
                scrip_token = str(match.iloc[0]['token'])
                trading_symbol = str(match.iloc[0]['symbol'])

            if scrip_token and trading_symbol:
                try:
                    opt_res = api.ltpData(exchange=exchange, tradingsymbol=trading_symbol, symboltoken=scrip_token)
                    if opt_res and opt_res.get('status') and opt_res.get('data'):
                        current_premium = float(opt_res['data']['ltp'])
                except Exception as e:
                    print(f"Error querying option LTP for {trading_symbol}:", str(e))

            if current_premium is None:
                current_premium = "couldnt fetch live price from API"
                leg_pnl = "N/A"
                has_leg_error = True
                entry_price = leg.get('entry_price') if leg.get('entry_price') is not None else "N/A"
            else:
                entry_price = float(leg.get('entry_price')) if leg.get('entry_price') is not None and float(leg.get('entry_price')) > 0 else current_premium
                leg_pnl = (current_premium - entry_price) * qty if action == 'BUY' else (entry_price - current_premium) * qty
                if not has_leg_error:
                    basket_pnl += leg_pnl

            greeks = calculate_greeks(opt_type, underlying_price, strike, T=7/365, r=0.07, sigma=0.145)

            legs_data.append({
                "strike": strike,
                "expiry": expiry,
                "option_type": opt_type,
                "action": action,
                "entry_price": round(entry_price, 2) if isinstance(entry_price, float) else entry_price,
                "current_premium": round(current_premium, 2) if isinstance(current_premium, float) else current_premium,
                "leg_pnl": round(leg_pnl, 2) if isinstance(leg_pnl, float) else leg_pnl,
                "greeks": greeks
            })

        processed_baskets.append({
            "name": basket.get('name', 'Basket'),
            "legs": legs_data,
            "basket_pnl": round(basket_pnl, 2) if not has_leg_error else "couldnt fetch live price from API"
        })

    return jsonify({
        "status": status_msg,
        "underlying_price": underlying_price,
        "is_market_open": is_market_open(),
        "chart_labels": chart_labels,
        "chart_prices": chart_prices,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
