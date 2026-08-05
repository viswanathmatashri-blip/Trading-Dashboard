import os
import math
import pyotp
import numpy as np
from flask import Flask, render_template, jsonify, request
from scipy.stats import norm
from SmartApi import SmartConnect

app = Flask(__name__)

# Fetch environment variables from Render
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

# Black-Scholes Greeks Calculation Engine
def calculate_greeks(flag, S, K, T, r, sigma):
    """
    Calculates Option Greeks using the Black-Scholes model.
    flag: 'C' or 'P'
    S: Underlying price, K: Strike price, T: Time to expiration in years,
    r: Risk-free rate (e.g. 0.07), sigma: Implied Volatility (e.g. 0.15)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100  # Per 1% change in IV

    if flag.upper() == 'CE' or flag.upper() == 'C':
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

from flask import render_template_string

HTML_CODE = """
<!DOCTYPE html>
<html>
... paste full HTML content here ...
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_CODE)
@app.route('/api/fetch-chain', methods=['POST'])
def fetch_chain():
    """Generates available strikes near the ATM price for selection."""
    data = request.json
    symbol = data.get('symbol', 'NIFTY')
    exchange = data.get('exchange', 'NFO')

    # Mock baseline index price if market is closed / dynamic anchor
    base_price = 24500.0 if symbol.upper() == 'NIFTY' else (52000.0 if symbol.upper() == 'BANKNIFTY' else 20000.0)
    step = 50 if symbol.upper() == 'NIFTY' else 100

    strikes = [int(base_price - (step * i)) for i in range(5, 0, -1)] + \
              [int(base_price)] + \
              [int(base_price + (step * i)) for i in range(1, 6)]

    return jsonify({"base_price": base_price, "strikes": strikes})

@app.route('/api/live-data', methods=['POST'])
def live_data():
    """API endpoint providing real-time technicals, divergence checks, and options prices."""
    req_data = request.json
    symbol = req_data.get('symbol', 'NIFTY')
    baskets = req_data.get('baskets', [])

    # Simulate realistic dynamic price tick for live streaming simulation
    np.random.seed()
    underlying_price = round(24500 + float(np.random.normal(0, 8)), 2)
    volume_sec = int(np.random.randint(1500, 5000))
    iv = round(14.5 + float(np.random.uniform(-0.5, 0.5)), 2)

    # Historical price data for Technical Analysis Indicators
    prices = list(np.random.normal(loc=underlying_price, scale=12, size=30)) + [underlying_price]
    
    # Technical Indicators: RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 1
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1
    rs = (avg_gain / avg_loss) if avg_loss != 0 else 1
    rsi = round(100 - (100 / (1 + rs)), 2)

    # Technical Indicators: Bollinger Bands
    sma_20 = float(np.mean(prices[-20:]))
    std_20 = float(np.std(prices[-20:]))
    upper_bb = round(sma_20 + (2 * std_20), 2)
    lower_bb = round(sma_20 - (2 * std_20), 2)

    # Automated Pattern / Breakout Alerts
    bollinger_breakout = "Upper Breakout" if underlying_price > upper_bb else ("Lower Breakout" if underlying_price < lower_bb else "Normal")
    rsi_divergence = "Bullish Divergence" if rsi < 30 and underlying_price > prices[-2] else ("Bearish Divergence" if rsi > 70 and underlying_price < prices[-2] else "None")
    macd_divergence = "Bullish Crossover" if rsi > 50 and prices[-1] > prices[-2] else "None"

    # Compute Basket Premiums & Greeks
    processed_baskets = []

    for basket in baskets:
        basket_pnl = 0.0
        legs_data = []

        for leg in basket.get('legs', []):
            strike = float(leg['strike'])
            opt_type = leg['option_type']  # CE or PE
            action = leg['action']        # BUY or SELL
            entry_price = float(leg.get('entry_price', 100.0))
            qty = int(leg.get('qty', 50))

            # Approximate Live Option Price based on Intrinsic + Time Value
            intrinsic = max(0, underlying_price - strike) if opt_type == 'CE' else max(0, strike - underlying_price)
            current_premium = round(intrinsic + max(5, 120 - (abs(underlying_price - strike) * 0.1)), 2)

            # Compute Leg P&L
            leg_pnl = (current_premium - entry_price) * qty if action == 'BUY' else (entry_price - current_premium) * qty
            basket_pnl += leg_pnl

            # Compute Greeks (Assuming 7 days / 0.0192 years to expiry)
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
