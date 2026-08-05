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
    if T <= 0.00001 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if flag.upper() in ['CE', 'C'] else max(0.0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if flag.upper() in ['CE', 'C']:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(flag, S, K, T, r, sigma):
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
    if not legs or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "expected_day_theta": 0.0}

    dS = 1.0
    dt = 1.0 / 365.0
    dSigma = 0.01

    V0 = calculate_basket_portfolio_value(legs, S, T, r, sigma)

    V_up = calculate_basket_portfolio_value(legs, S + dS, T, r, sigma)
    V_dn = calculate_basket_portfolio_value(legs, S - dS, T, r, sigma)
    net_delta = (V_up - V_dn) / (2 * dS)

    net_gamma = (V_up - 2 * V0 + V_dn) / (dS ** 2)

    V_time = calculate_basket_portfolio_value(legs, S, max(T - dt, 0.00001), r, sigma)
    net_theta = V_time - V0

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
        .basket-summary-tag { font-family: monospace; background: #1a3835; border: 1px solid #00bcd4; padding: 3px 7px; border-radius: 4px; font-size: 11px; color: #00e5ff; margin-left: 6px; display: inline-block; }

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
            width: 14px;
            height: 14px;
            border: 2px solid rgba(0, 188, 212, 0.3);
            border-radius: 50%;
            border-top-color: #00bcd4;
            animation: spin 0.8s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-left: 8px;
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
                <div class="status-item">Current IV: <span id="stIv" class="status-value">-</span></div>
                <div class="status-item">Expected 1-Day Move: <span id="stMove" class="status-value">-</span></div>
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
                <div style="display: flex; align-items: center;">
                    <h3 style="margin: 0;">Active Basket Positions & Real-Time Option LTP</h3>
                    <span id="basketHeaderStatus"></span>
                </div>
                <div id="basketsContainer" style="margin-top: 15px;"></div>
            </div>

            <div class="card">
                <div class="header-flex">
                    <h3>Basket Legs Real-Time Premium Chart</h3>
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
        let deletedBasketIds = new Set();
        let mainChart, legsChart;
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
            data: { labels: [], datasets: [{ label: 'Index Spot Price', data: [], borderColor: '#00bcd4', borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, yAxisID: 'y', tension: 0.1 }] },
            options: { 
                animation: false,
                responsive: true,
                scales: { y: { display: true, position: 'left', grid: { color: '#2a2a2a' } } } 
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

            activeBaskets.push({ id: basketId, name: newBasketName, legs: [...pendingLegs] });
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
            
            if (selectedChartBasketId === basketId) {
                selectedChartBasketId = activeBaskets.length > 0 ? activeBaskets[0].id : null;
            }
            
            updateDashboard();
        }

        function toggleBasketChart(basketId) {
            selectedChartBasketId = selectedChartBasketId === basketId ? null : basketId;
            updateDashboard();
        }

        async function updateDashboard() {
            const symbol = document.getElementById('symbol').value;
            const exchange = document.getElementById('exchange').value;
            const interval = document.getElementById('timeframeSelect').value;
            const basketInterval = document.getElementById('basketTimeframeSelect').value;

            activeBaskets = activeBaskets.filter(b => !deletedBasketIds.has(b.id));

            try {
                const res = await fetch('/api/live-data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ 
                        symbol, exchange, interval, basket_interval: basketInterval, 
                        baskets: activeBaskets 
                    })
                });
                const data = await res.json();

                if (data.status) updateStatus(data.status, "info");
                if (data.error) { document.getElementById('stIndex').innerText = data.error; return; }

                document.getElementById('stIndex').innerText = data.underlying_price;
                document.getElementById('stIv').innerText = data.current_iv + "%";
                document.getElementById('stMove').innerText = "±" + data.expected_1day_move + " pts";
                document.getElementById('stMarketStatus').innerText = data.is_market_open ? "OPEN (Live)" : "CLOSED";

                if (data.chart_labels && data.chart_labels.length > 0) {
                    mainChart.data.labels = data.chart_labels;
                    mainChart.data.datasets[0].data = data.chart_prices;
                    mainChart.update('none');
                }

                const validBaskets = (data.baskets || []).filter(b => !deletedBasketIds.has(b.id));

                const container = document.getElementById('basketsContainer');
                container.innerHTML = '';

                validBaskets.forEach(b => {
                    let pnlDisplay = typeof b.basket_pnl === 'number' ? `₹${b.basket_pnl}` : b.basket_pnl;
                    let pnlClass = typeof b.basket_pnl === 'number' ? (b.basket_pnl >= 0 ? 'pnl-pos' : 'pnl-neg') : 'pnl-err';
                    let isChecked = selectedChartBasketId === b.id;

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
                            
                            <div style="display:flex; align-items:center;">
                                <label class="chart-checkbox-container">
                                    <input type="checkbox" ${isChecked ? 'checked' : ''} onchange="toggleBasketChart(${b.id})" style="width:auto; margin:0;">
                                    <span>Load Chart</span>
                                </label>
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
                        datasets.push({
                            label: series.label,
                            data: series.prices,
                            borderColor: CHART_COLORS[idx % CHART_COLORS.length],
                            borderWidth: 1.5,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            backgroundColor: 'transparent',
                            tension: 0.1
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
    expiries = sorted(filtered['expiry'].unique().tolist())
    
    return jsonify({"expiries": expiries, "status": "API connection success"})

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

    idx_info = INDEX_TOKENS.get(symbol, INDEX_TOKENS['NIFTY'])
    
    try:
        ltp_resp = smart_api.ltpData(idx_info["exchange"], idx_info["tradingsymbol"], idx_info["token"])
        if not (ltp_resp and ltp_resp.get('status') and ltp_resp.get('data')):
            return jsonify({"error": "Failed to fetch Index LTP", "status": "LTP Fetch Failed"})
        
        spot_price = float(ltp_resp['data']['ltp'])
    except Exception as e:
        return jsonify({"error": str(e), "status": f"API Error: {str(e)}"})

    now = datetime.now()
    from_date = now.strftime("%Y-%m-%d 09:15")
    to_date = now.strftime("%Y-%m-%d %H:%M")

    # Fetch Index Candle History
    chart_labels, chart_prices = [], []
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
            candles = hist_resp['data']
            chart_labels = [c[0].split('T')[1][:5] for c in candles]
            chart_prices = [float(c[4]) for c in candles]
    except Exception:
        pass

    iv_estimate = 14.5
    expected_1day_move = round(spot_price * (iv_estimate / 100.0) * math.sqrt(1 / 365.0), 2)

    processed_baskets = []
    for basket in baskets:
        basket_pnl = 0.0
        processed_legs = []
        leg_series_data = []
        basket_chart_labels = []

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

            if token and symbol_name:
                try:
                    opt_ltp_resp = smart_api.ltpData("NFO", symbol_name, token)
                    if opt_ltp_resp and opt_ltp_resp.get('status') and opt_ltp_resp.get('data'):
                        current_ltp = float(opt_ltp_resp['data']['ltp'])
                except Exception:
                    pass

                # Fetch Option Leg Candle History
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
                        if not basket_chart_labels:
                            basket_chart_labels = [c[0].split('T')[1][:5] for c in leg_candles]
                        
                        leg_prices = [float(c[4]) for c in leg_candles]
                        leg_series_data.append({
                            "label": f"{strike} {opt_type} ({action})",
                            "prices": leg_prices
                        })
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

            try:
                exp_dt = datetime.strptime(exp_date, "%d%b%Y")
                days_to_exp = max((exp_dt - datetime.now()).days, 0.5)
            except Exception:
                days_to_exp = 7.0

            T = days_to_exp / 365.0
            greeks = calculate_greeks(opt_type, spot_price, strike, T, 0.07, iv_estimate / 100.0)
            decay_till_date = round((entry_price - (current_ltp if isinstance(current_ltp, (int, float)) else entry_price)) * qty, 2)

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
                "theta_decay_till_date": decay_till_date
            })

        net_greeks = calculate_basket_greeks_from_price_diff(basket.get('legs', []), spot_price)
        max_prof, max_lss = calculate_max_profit_loss(basket.get('legs', []))

        processed_baskets.append({
            "id": basket['id'],
            "name": basket['name'],
            "basket_pnl": round(basket_pnl, 2),
            "max_profit": max_prof,
            "max_loss": max_lss,
            "net_greeks": net_greeks,
            "total_decay_till_date": round(basket_pnl, 2),
            "legs": processed_legs,
            "legs_historical": {
                "labels": basket_chart_labels,
                "leg_series": leg_series_data
            }
        })

    return jsonify({
        "status": conn_msg,
        "underlying_price": spot_price,
        "current_iv": iv_estimate,
        "expected_1day_move": expected_1day_move,
        "is_market_open": is_market_open(),
        "chart_labels": chart_labels,
        "chart_prices": chart_prices,
        "baskets": processed_baskets
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
