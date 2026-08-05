import os
import json
import math
import pyotp
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
from flask import Flask, render_template_string, jsonify, request
from scipy.stats import norm
from SmartApi import SmartConnect

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")
CLIENT_CODE = os.environ.get("CLIENT_CODE")
PIN = os.environ.get("PIN")
TOTP_SECRET = os.environ.get("TOTP_SECRET")

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
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)

def bs_price(flag, S, K, T, r, sigma):
    """Black-Scholes theoretical price calculation."""
    if T <= 0.00001 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if flag.upper() in ['CE', 'C'] else max(0.0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if flag.upper() in ['CE', 'C']:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(flag, S, K, T, r, sigma):
    """Individual leg display Greeks."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "expected_day_theta": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100.0

    if flag.upper() in ['CE', 'C']:
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0

    days_left = max(T * 365.0, 0.5)
    expected_day_theta = theta * (1.0 / days_left)

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "expected_day_theta": round(expected_day_theta, 4)
    }

def calculate_basket_portfolio_value(legs, S, T, r=0.07, sigma=0.145):
    """Calculates total basket portfolio market value given index price S."""
    total_val = 0.0
    for leg in legs:
        flag = leg['option_type']
        strike = float(leg['strike'])
        qty = int(leg.get('qty', 65))
        direction = 1 if leg['action'] == 'BUY' else -1
        
        p = bs_price(flag, S, strike, T, r, sigma)
        total_val += (p * qty * direction)
    return total_val

def calculate_basket_greeks_from_price_diff(legs, S, T=7/365, r=0.07, sigma=0.145):
    """
    Computes Net Basket Greeks by bumping underlying index price (dS), time (dT), and IV (dSigma).
    Measures portfolio P&L / premium changes w.r.t index movements.
    """
    if not legs or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "expected_day_theta": 0.0}

    dS = 1.0          # 1 point shift in spot price
    dt = 1.0 / 365.0  # 1 day time decay shift
    dSigma = 0.01     # 1% IV shift

    # Base Portfolio Premium Value
    V0 = calculate_basket_portfolio_value(legs, S, T, r, sigma)

    # 1. Delta (Change in portfolio value per +1 point move in Spot)
    V_up = calculate_basket_portfolio_value(legs, S + dS, T, r, sigma)
    V_dn = calculate_basket_portfolio_value(legs, S - dS, T, r, sigma)
    net_delta = (V_up - V_dn) / (2 * dS)

    # 2. Gamma (Rate of change of Delta per +1 point move in Spot)
    net_gamma = (V_up - 2 * V0 + V_dn) / (dS ** 2)

    # 3. Theta (Change in portfolio value per 1 day passage of time)
    V_time = calculate_basket_portfolio_value(legs, S, max(T - dt, 0.00001), r, sigma)
    net_theta = V_time - V0

    # 4. Vega (Change in portfolio value per 1% rise in IV)
    V_vol = calculate_basket_portfolio_value(legs, S, T, r, sigma + dSigma)
    net_vega = V_vol - V0

    days_left = max(T * 365.0, 0.5)
    net_expected_day_theta = net_theta * (1.0 / days_left)

    return {
        "delta": round(net_delta, 2),
        "gamma": round(net_gamma, 4),
        "theta": round(net_theta, 2),
        "vega": round(net_vega, 2),
        "expected_day_theta": round(net_expected_day_theta, 2)
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
            position: fixed; top: 10px; left: 15px; z-index: 9999;
            background: rgba(20, 20, 20, 0.95); border: 1px solid #00bcd4;
            color: #00ff88; padding: 6px 14px; border-radius: 20px;
            font-size: 11px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            max-width: 90vw; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }

        .pill-err { border-color: #ff5252 !important; color: #ff5252 !important; }
        .pill-warn { border-color: #ffca28 !important; color: #ffca28 !important; }

        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; margin-top: 15px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 15px; }
        h2, h3 { margin-top: 0; color: #00bcd4; }
        
        .header-flex { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .header-flex h3 { margin: 0; }
        .chart-select { width: auto !important; padding: 4px 8px !important; margin: 0 !important; font-size: 12px; }

        label { display: block; margin-top: 10px; font-size: 12px; color: #aaa; }
        select, input, button { width: 100%; padding: 8px; margin-top: 5px; background: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; box-sizing: border-box; }
        button { background: #00bcd4; color: black; font-weight: bold; cursor: pointer; border: none; margin-top: 15px; }
        button:hover { background: #008ba3; }
        
        .btn-delete { background: #ff5252; color: white; border: none; padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 11px; width: auto; margin: 0; }
        .btn-delete:hover { background: #d32f2f; }
        
        .pending-leg-item { display: flex; justify-content: space-between; align-items: center; background: #2a2a2a; padding: 6px; border-radius: 4px; margin-top: 5px; font-size: 12px; }

        .status-bar { display: flex; gap: 12px; background: #262626; padding: 10px; border-radius: 6px; font-weight: bold; margin-bottom: 15px; flex-wrap: wrap; }
        .status-item { font-size: 12px; }
        .status-value { color: #00ff88; }
        .pnl-pos { color: #00ff88; font-weight: bold; }
        .pnl-neg { color: #ff5252; font-weight: bold; }
        .pnl-err { color: #ff9800; font-weight: bold; }
        .greek-tag { font-family: monospace; background: #2d2d2d; padding: 3px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px; display: inline-block; margin-top: 4px; }
        .basket-summary-tag { font-family: monospace; background: #1a3835; border: 1px solid #00bcd4; padding: 3px 7px; border-radius: 4px; font-size: 11px; color: #00e5ff; margin-left: 6px; display: inline-block; }
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
                <div class="status-item">Market Status: <span id="stMarketStatus" class="status-value">-</span></div>
            </div>

            <div class="card">
                <div class="header-flex">
                    <h3>Index Strategy Chart</h3>
                    <div>
                        <label style="display:inline; color:#aaa; font-size:12px; margin-right:5px;">Candle Timeframe:</label>
                        <select id="timeframeSelect" class="chart-select" onchange="updateDashboard()">
                            <option value="1">1 min</option>
                            <option value="3">3 mins</option>
                            <option value="5">5 mins</option>
                            <option value="15" selected>15 mins</option>
                        </select>
                    </div>
                </div>
                <canvas id="mainChart" height="100"></canvas>
            </div>

            <div class="card">
                <h3>Active Basket Positions & Real-Time Option LTP</h3>
                <div id="basketsContainer"></div>
            </div>

            <div class="card">
                <h3>Basket Legs Real-Time Premium Chart (1-Day Scale)</h3>
                <canvas id="legsPremiumChart" height="120"></canvas>
            </div>
        </div>
    </div>

    <script>
        let pendingLegs = [];
        let activeBaskets = [];
        let mainChart, legsChart;
        let legHistory = {};

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
            data: { labels: [], datasets: [{ label: 'Index Spot Price', data: [], borderColor: '#00bcd4', yAxisID: 'y', tension: 0.1 }] },
            options: { scales: { y: { display: true, position: 'left', grid: { color: '#2a2a2a' } } } }
        });

        const ctxLegs = document.getElementById('legsPremiumChart').getContext('2d');
        legsChart = new Chart(ctxLegs, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { grid: { color: '#2a2a2a' } },
                    y1: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Leg 1 Premium (₹)', color: '#00bcd4' }, grid: { color: '#2a2a2a' } },
                    y2: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Leg 2 Premium (₹)', color: '#ff9800' }, grid: { drawOnChartArea: false } }
                }
            }
        });

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
            activeBaskets.push({ id: Date.now(), name: `Basket #${activeBaskets.length + 1}`, legs: [...pendingLegs] });
            pendingLegs = [];
            renderPendingLegs();
            updateDashboard();
        }

        function deleteBasket(basketId) {
            activeBaskets = activeBaskets.filter(b => b.id !== basketId);
            updateDashboard();
        }

        async function updateDashboard() {
            const symbol = document.getElementById('symbol').value;
            const exchange = document.getElementById('exchange').value;
            const interval = document.getElementById('timeframeSelect').value;

            try {
                const res = await fetch('/api/live-data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ symbol, exchange, interval, baskets: activeBaskets })
                });
                const data = await res.json();

                if (data.status) updateStatus(data.status, "info");
                if (data.error) { document.getElementById('stIndex').innerText = data.error; return; }

                document.getElementById('stIndex').innerText = data.underlying_price;
                document.getElementById('stMarketStatus').innerText = data.is_market_open ? "OPEN (Live)" : "CLOSED";

                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    mainChart.data.datasets[0].data = data.chart_prices;
                    mainChart.update();
                }

                const container = document.getElementById('basketsContainer');
                container.innerHTML = '';

                let nowTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

                data.baskets.forEach(b => {
                    let pnlDisplay = typeof b.basket_pnl === 'number' ? `₹${b.basket_pnl}` : b.basket_pnl;
                    let pnlClass = typeof b.basket_pnl === 'number' ? (b.basket_pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : 'pnl-err';

                    let html = `<div style="border-bottom: 1px solid #333; padding: 10px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
                                <strong>${b.name}</strong>
                                <span class="${pnlClass}" style="margin-left: 10px; margin-right: 10px;">Live P&L: ${pnlDisplay}</span>
                                
                                <span class="basket-summary-tag">Net &Delta;: ₹${b.net_greeks.delta} /pt</span>
                                <span class="basket-summary-tag">Net &Gamma;: ₹${b.net_greeks.gamma} /pt&sup2;</span>
                                <span class="basket-summary-tag">Net &Theta;: ₹${b.net_greeks.theta} /day</span>
                                <span class="basket-summary-tag">Net &Nu;: ₹${b.net_greeks.vega} /1% IV</span>
                                <span class="basket-summary-tag" style="color:#00ff88; border-color:#00ff88; background:#1b3821;">Tot Decay: ₹${b.total_decay_till_date}</span>
                                <span class="basket-summary-tag" style="color:#ffca28; border-color:#ffca28; background:#38321b;">Exp Day &Theta;: ₹${b.net_greeks.expected_day_theta}</span>
                            </div>
                            <button class="btn-delete" onclick="deleteBasket(${b.id})" style="padding: 4px 8px; margin-top:4px;">🗑️ Delete Basket</button>
                        </div>`;

                    b.legs.forEach((leg, idx) => {
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
                                    <span class="greek-tag" style="color:#00ff88;">Decay Till Date: ₹${leg.theta_decay_till_date}</span>
                                    <span class="greek-tag" style="color:#00bcd4;">Exp Day Theta: ${leg.greeks.expected_day_theta}</span>
                                </div>
                            </div>`;
                    });
                    html += `</div>`;
                    container.innerHTML += html;
                });

                updateLegsChart(data.baskets, nowTime);

            } catch(e) { updateStatus("Couldnt fetch live price from API", "error"); }
        }

        function updateLegsChart(baskets, nowTime) {
            if (baskets.length === 0 || baskets[0].legs.length === 0) return;

            let basket = baskets[0];
            if (!legsChart.data.labels.includes(nowTime)) {
                legsChart.data.labels.push(nowTime);
            }

            if (legsChart.data.labels.length > 40) legsChart.data.labels.shift();

            let datasets = [];

            basket.legs.forEach((leg, idx) => {
                let legKey = `leg_${idx}`;
                if (!legHistory[legKey]) legHistory[legKey] = [];
                if (typeof leg.current_premium === 'number') legHistory[legKey].push(leg.current_premium);
                if (legHistory[legKey].length > 40) legHistory[legKey].shift();

                let color = idx === 0 ? '#00bcd4' : '#ff9800';
                let axis = idx === 0 ? 'y1' : 'y2';

                datasets.push({
                    label: `${leg.strike} ${leg.option_type} (${leg.action})`,
                    data: [...legHistory[legKey]],
                    borderColor: color,
                    backgroundColor: idx === 1 ? 'rgba(0, 255, 136, 0.15)' : 'transparent',
                    fill: idx === 1 ? '-1' : false,
                    yAxisID: axis,
                    tension: 0.2
                });
            });

            legsChart.data.datasets = datasets;
            legsChart.update();
        }

        loadExpiries();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    ensure_scrip_master_loading()
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fetch-expiries', methods=['POST'])
def fetch_expiries():
    ensure_scrip_master_loading()
    api, api_status = get_smart_api()
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    
    status_msg = SCRIP_MASTER_STATUS if INSTRUMENT_DF is None else api_status

    if INSTRUMENT_DF is not None and not INSTRUMENT_DF.empty:
        symbol_df = INSTRUMENT_DF[INSTRUMENT_DF['name'] == symbol]
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
    ensure_scrip_master_loading()
    api, api_status = get_smart_api()
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])

    status_msg = SCRIP_MASTER_STATUS if INSTRUMENT_DF is None else api_status

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
            return jsonify({"status": "Couldnt fetch live price", "strikes": []})
    except Exception as e:
        return jsonify({"status": f"Couldnt fetch live price: {str(e)}", "strikes": []})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    ensure_scrip_master_loading()
    api, api_status = get_smart_api()
    req_data = request.json
    symbol = req_data.get('symbol', 'NIFTY')
    exchange = req_data.get('exchange', 'NFO')
    interval_key = req_data.get('interval', '15')
    baskets = req_data.get('baskets', [])

    candle_interval = INTERVAL_MAP.get(interval_key, "FIFTEEN_MINUTE")
    config = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])

    if INSTRUMENT_DF is None:
        return jsonify({"status": SCRIP_MASTER_STATUS, "error": SCRIP_MASTER_STATUS})

    status_msg = api_status

    if not api:
        return jsonify({"status": status_msg, "error": status_msg})

    underlying_price = None
    try:
        res = api.ltpData(exchange=config['exchange'], tradingsymbol=config['tradingsymbol'], symboltoken=config['token'])
        if res and res.get('status') and res.get('data'):
            underlying_price = float(res['data']['ltp'])
    except Exception as e:
        print("Error querying SmartAPI LTP:", str(e))

    if underlying_price is None:
        return jsonify({"status": "Couldnt fetch live price from API", "error": "couldnt fetch live price from API"})

    chart_labels, chart_prices = [], []
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        candle_param = {
            "exchange": config['exchange'],
            "symboltoken": config['token'],
            "interval": candle_interval,
            "fromdate": f"{today_str} 09:15",
            "todate": f"{today_str} 15:30"
        }
        candle_data = api.getCandleData(candle_param)
        if candle_data and candle_data.get('status') and candle_data.get('data'):
            for candle in candle_data['data']:
                chart_labels.append(candle[0].split('T')[1][:5])
                chart_prices.append(float(candle[4]))
    except Exception as e:
        print("Error fetching candle data:", str(e))

    processed_baskets = []

    for basket in baskets:
        basket_pnl = 0.0
        legs_data = []
        has_leg_error = False
        total_decay_till_date = 0.0

        raw_legs = basket.get('legs', [])

        for leg in raw_legs:
            strike = float(leg['strike'])
            expiry = leg.get('expiry', '').upper()
            opt_type = leg['option_type']
            action = leg['action']
            qty = int(leg.get('qty', 65))

            current_premium, scrip_token, trading_symbol = None, None, None

            match = INSTRUMENT_DF[
                (INSTRUMENT_DF['name'] == symbol) & 
                (abs(INSTRUMENT_DF['strike_price'] - strike) < 0.01) & 
                (INSTRUMENT_DF['symbol'].str.endswith(opt_type)) &
                (INSTRUMENT_DF['expiry'].str.upper() == expiry)
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
                theta_decay_till_date = "N/A"
                has_leg_error = True
                entry_price = leg.get('entry_price') if leg.get('entry_price') is not None else "N/A"
            else:
                entry_price = float(leg.get('entry_price')) if leg.get('entry_price') is not None and float(leg.get('entry_price')) > 0 else current_premium
                leg_pnl = (current_premium - entry_price) * qty if action == 'BUY' else (entry_price - current_premium) * qty
                theta_decay_till_date = round((entry_price - current_premium) * qty, 2) if action == 'SELL' else round((current_premium - entry_price) * qty, 2)
                
                if not has_leg_error:
                    basket_pnl += leg_pnl
                    total_decay_till_date += theta_decay_till_date

            greeks = calculate_greeks(opt_type, underlying_price, strike, T=7/365, r=0.07, sigma=0.145)

            legs_data.append({
                "strike": strike,
                "expiry": expiry,
                "option_type": opt_type,
                "action": action,
                "qty": qty,
                "entry_price": round(entry_price, 2) if isinstance(entry_price, float) else entry_price,
                "current_premium": round(current_premium, 2) if isinstance(current_premium, float) else current_premium,
                "leg_pnl": round(leg_pnl, 2) if isinstance(leg_pnl, float) else leg_pnl,
                "theta_decay_till_date": theta_decay_till_date,
                "greeks": greeks
            })

        # Calculate Net Basket Greeks using Price Differences (Bump & Revalue method)
        net_greeks = calculate_basket_greeks_from_price_diff(raw_legs, underlying_price)

        processed_baskets.append({
            "id": basket.get('id'),
            "name": basket.get('name', 'Basket'),
            "legs": legs_data,
            "basket_pnl": round(basket_pnl, 2) if not has_leg_error else "couldnt fetch live price from API",
            "total_decay_till_date": round(total_decay_till_date, 2) if not has_leg_error else "N/A",
            "net_greeks": net_greeks
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
