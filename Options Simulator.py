import os
import math
import pyotp
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from scipy.stats import norm
from SmartApi import SmartConnect

app = Flask(__name__)

# Environment variable retrieval for Render
API_KEY = os.environ.get("API_KEY")
CLIENT_CODE = os.environ.get("CLIENT_CODE")
PIN = os.environ.get("PIN")
TOTP_SECRET = os.environ.get("TOTP_SECRET")

smart_api = None

def get_smart_api_session():
    """Authenticates with Angel One SmartAPI using TOTP."""
    global smart_api
    if smart_api is None:
        smart_api = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_SECRET).now()
        data = smart_api.generateSession(CLIENT_CODE, PIN, totp)
        if not data.get('status'):
            raise Exception("SmartAPI Authentication Failed: " + str(data.get('message')))
    return smart_api

def calculate_greeks(flag, S, K, T, r, sigma):
    """Calculates Option Greeks using the Black-Scholes model."""
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
    <title>Live Options Trading Simulator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 20px; }
        .grid { display: grid; grid-template-columns: 350px 1fr; gap: 20px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 15px; }
        h2, h3 { margin-top: 0; color: #00bcd4; }
        label { display: block; margin-top: 10px; font-size: 12px; color: #aaa; }
        select, input, button { width: 100%; padding: 8px; margin-top: 5px; background: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; box-sizing: border-box; }
        button { background: #00bcd4; color: black; font-weight: bold; cursor: pointer; border: none; margin-top: 15px; }
        button:hover { background: #008ba3; }
        .status-bar { display: flex; gap: 15px; background: #262626; padding: 10px; border-radius: 6px; font-weight: bold; margin-bottom: 15px; flex-wrap: wrap; }
        .status-item { font-size: 13px; }
        .status-value { color: #00ff88; }
        .pnl-positive { color: #00ff88; font-weight: bold; }
        .pnl-negative { color: #ff5252; font-weight: bold; }
        .greek-tag { font-family: monospace; background: #2d2d2d; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin-right: 5px; display: inline-block; margin-top: 4px; }
    </style>
</head>
<body>
    <h2>Live Options Trading Simulator</h2>
    <div class="grid">
        <div>
            <div class="card">
                <h3>1. Select Asset</h3>
                <label>Index Symbol</label>
                <select id="symbol" onchange="loadChain()">
                    <option value="NIFTY">NIFTY 50</option>
                    <option value="BANKNIFTY">BANKNIFTY</option>
                </select>

                <label>Exchange</label>
                <select id="exchange">
                    <option value="NFO">NFO</option>
                    <option value="SFO">SFO</option>
                </select>
            </div>

            <div class="card">
                <h3>2. Create Basket</h3>
                <label>Strike Price</label>
                <select id="strikeSelect"></select>

                <label>Option Type</label>
                <select id="optType">
                    <option value="CE">CE (Call)</option>
                    <option value="PE">PE (Put)</option>
                </select>

                <label>Action</label>
                <select id="action">
                    <option value="BUY">BUY</option>
                    <option value="SELL">SELL</option>
                </select>

                <button onclick="addLegToTempBasket()">+ Add Leg to Pending Basket</button>
                <div id="tempLegs" style="margin-top: 10px; font-size: 12px; color: #ffca28;"></div>
                <button onclick="saveBasket()" style="background: #4caf50; color: white;">Execute Basket</button>
            </div>
        </div>

        <div>
            <div class="status-bar">
                <div class="status-item">Index: <span id="stIndex" class="status-value">-</span></div>
                <div class="status-item">Vol/sec: <span id="stVol" class="status-value">-</span></div>
                <div class="status-item">IV: <span id="stIV" class="status-value">-</span></div>
                <div class="status-item">RSI: <span id="stRSI" class="status-value">-</span></div>
                <div class="status-item">BB Breakout: <span id="stBB" class="status-value">-</span></div>
                <div class="status-item">RSI Div: <span id="stRSIDiv" class="status-value">-</span></div>
                <div class="status-item">MACD Div: <span id="stMACD" class="status-value">-</span></div>
            </div>

            <div class="card">
                <h3>Underlying Asset Live Chart (Primary Y: Price, Secondary Y: Premiums/IV)</h3>
                <canvas id="mainChart" height="100"></canvas>
            </div>

            <div class="card">
                <h3>Active Baskets & Live Greeks</h3>
                <div id="basketsContainer"></div>
            </div>
        </div>
    </div>

    <script>
        let pendingLegs = [];
        let activeBaskets = [];
        let mainChart;

        const ctx = document.getElementById('mainChart').getContext('2d');
        mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Underlying Price', data: [], borderColor: '#00bcd4', yAxisID: 'y' },
                    { label: 'Bollinger Upper', data: [], borderColor: '#ff9800', borderDash: [5, 5], yAxisID: 'y' },
                    { label: 'Bollinger Lower', data: [], borderColor: '#ff9800', borderDash: [5, 5], yAxisID: 'y' },
                    { label: 'Live IV (%)', data: [], borderColor: '#e91e63', yAxisID: 'y1' }
                ]
            },
            options: {
                scales: {
                    y: { type: 'linear', display: true, position: 'left', grid: { color: '#333' } },
                    y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
                }
            }
        });

        async function loadChain() {
            const symbol = document.getElementById('symbol').value;
            const exchange = document.getElementById('exchange').value;
            const res = await fetch('/api/fetch-chain', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ symbol, exchange })
            });
            const data = await res.json();
            const select = document.getElementById('strikeSelect');
            select.innerHTML = '';
            data.strikes.forEach(s => {
                let opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s;
                select.appendChild(opt);
            });
        }

        function addLegToTempBasket() {
            const strike = document.getElementById('strikeSelect').value;
            const option_type = document.getElementById('optType').value;
            const action = document.getElementById('action').value;
            pendingLegs.push({ strike, option_type, action, entry_price: 100, qty: 50 });
            document.getElementById('tempLegs').innerText = `Pending Legs: ${pendingLegs.length} legs selected.`;
        }

        function saveBasket() {
            if (pendingLegs.length === 0) return alert("Add legs first!");
            activeBaskets.push({ name: `Basket ${activeBaskets.length + 1}`, legs: [...pendingLegs] });
            pendingLegs = [];
            document.getElementById('tempLegs').innerText = '';
        }

        async function fetchLiveData() {
            const symbol = document.getElementById('symbol').value;
            const res = await fetch('/api/live-data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ symbol, baskets: activeBaskets })
            });
            const data = await res.json();

            document.getElementById('stIndex').innerText = data.underlying_price;
            document.getElementById('stVol').innerText = data.volume_sec;
            document.getElementById('stIV').innerText = data.iv + '%';
            document.getElementById('stRSI').innerText = data.rsi;
            document.getElementById('stBB').innerText = data.bollinger_breakout;
            document.getElementById('stRSIDiv').innerText = data.rsi_divergence;
            document.getElementById('stMACD').innerText = data.macd_divergence;

            const timeLabel = new Date().toLocaleTimeString();
            if (mainChart.data.labels.length > 20) {
                mainChart.data.labels.shift();
                mainChart.data.datasets.forEach(ds => ds.data.shift());
            }
            mainChart.data.labels.push(timeLabel);
            mainChart.data.datasets[0].data.push(data.underlying_price);
            mainChart.data.datasets[1].data.push(data.bb_upper);
            mainChart.data.datasets[2].data.push(data.bb_lower);
            mainChart.data.datasets[3].data.push(data.iv);
            mainChart.update();

            const container = document.getElementById('basketsContainer');
            container.innerHTML = '';

            data.baskets.forEach(b => {
                const pnlClass = b.basket_pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                let html = `<div style="border-bottom: 1px solid #444; padding: 10px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <strong>${b.name}</strong>
                        <span class="${pnlClass}">P&L: ₹${b.basket_pnl}</span>
                    </div>`;

                b.legs.forEach(leg => {
                    html += `
                        <div style="margin-top:5px;">
                            <span>${leg.strike} ${leg.option_type} ${leg.action[0]} @ ₹${leg.current_premium}</span>
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
        }

        loadChain();
        setInterval(fetchLiveData, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fetch-chain', methods=['POST'])
def fetch_chain():
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    base_price = 24500.0 if symbol.upper() == 'NIFTY' else 52000.0
    step = 50 if symbol.upper() == 'NIFTY' else 100
    strikes = [int(base_price + (step * i)) for i in range(-5, 6)]
    return jsonify({"base_price": base_price, "strikes": strikes})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    req_data = request.json
    symbol = req_data.get('symbol', 'NIFTY')
    baskets = req_data.get('baskets', [])

    underlying_price = round(24500 + float(np.random.normal(0, 8)), 2)
    volume_sec = int(np.random.randint(1500, 5000))
    iv = round(14.5 + float(np.random.uniform(-0.5, 0.5)), 2)

    prices = list(np.random.normal(loc=underlying_price, scale=12, size=30)) + [underlying_price]
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 1
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1
    rs = (avg_gain / avg_loss) if avg_loss != 0 else 1
    rsi = round(100 - (100 / (1 + rs)), 2)

    sma_20 = float(np.mean(prices[-20:]))
    std_20 = float(np.std(prices[-20:]))
    upper_bb = round(sma_20 + (2 * std_20), 2)
    lower_bb = round(sma_20 - (2 * std_20), 2)

    bollinger_breakout = "Upper Breakout" if underlying_price > upper_bb else ("Lower Breakout" if underlying_price < lower_bb else "Normal")
    rsi_divergence = "Bullish Divergence" if rsi < 30 and underlying_price > prices[-2] else ("Bearish Divergence" if rsi > 70 and underlying_price < prices[-2] else "None")
    macd_divergence = "Bullish Crossover" if rsi > 50 and prices[-1] > prices[-2] else "None"

    processed_baskets = []
    for basket in baskets:
        basket_pnl = 0.0
        legs_data = []
        for leg in basket.get('legs', []):
            strike = float(leg['strike'])
            opt_type = leg['option_type']
            action = leg['action']
            entry_price = float(leg.get('entry_price', 100.0))
            qty = int(leg.get('qty', 50))

            intrinsic = max(0, underlying_price - strike) if opt_type == 'CE' else max(0, strike - underlying_price)
            current_premium = round(intrinsic + max(5, 120 - (abs(underlying_price - strike) * 0.1)), 2)
            leg_pnl = (current_premium - entry_price) * qty if action == 'BUY' else (entry_price - current_premium) * qty
            basket_pnl += leg_pnl

            greeks = calculate_greeks(opt_type, underlying_price, strike, T=7/365, r=0.07, sigma=iv/100)
            legs_data.append({
                "strike": strike,
                "option_type": opt_type,
                "action": action,
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
        "volume_sec": volume_sec,
        "iv": iv,
        "rsi": rsi,
        "bb_upper": upper_bb,
        "bb_lower": lower_bb,
        "bollinger_breakout": bollinger_breakout,
        "rsi_divergence": rsi_divergence,
        "macd_divergence": macd_divergence,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
