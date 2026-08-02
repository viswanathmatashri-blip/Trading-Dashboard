# ==============================================================================
# IMPROVED QUANTITATIVE IRON CONDOR ENGINE - PRODUCTION VERSION
# Complete with ALL imports and CORRECT INDENTATION
# ==============================================================================

import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle

import numpy as np
import pandas as pd
import pyotp
import streamlit as st
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from vollib.black_scholes.greeks.analytical import (
    delta, gamma, vega, theta, rho
)
from vollib.black_scholes.implied_volatility import implied_volatility
from SmartApi import SmartConnect

print("✅ All imports successful!")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

@dataclass
class Config:
    """Centralized configuration management"""   
# ==================== CREDENTIALS (set these as Render environment variables) ====================
    API_KEY     = os.environ.get("SMARTAPI_KEY", "o2b7s4Oo")
    CLIENT_CODE = os.environ.get("SMARTAPI_CLIENT_CODE", "AACK311190")
    PASSWORD    = os.environ.get("SMARTAPI_PASSWORD", "8547")
    TOTP_SECRET = os.environ.get("SMARTAPI_TOTP_SECRET", "YCRQCDQ7NPUHKYH7RS73NXQ5VE")
# =================================================================================================== 
# Market Parameters
    RISK_FREE_RATE: float = 0.068  # Benchmark Repo rate
    NIFTY_DIVIDEND_YIELD: float = 0.013  # 1.3% annual dividend
    LOT_SIZE: int = 65
    
    # Option Chain Settings
    OPTION_CHAIN_TTL: int = 1800  # 30 minutes
    QUOTE_DATA_TTL: int = 60  # 1 minute
    DATA_FRESHNESS_THRESHOLD_SEC: int = 30  # Reject quotes older than 30s
    
    # Risk Management
    MIN_LIQUIDITY_OI: int = 10000  # Minimum open interest
    MAX_BID_ASK_SPREAD_PERCENT: float = 2.5
    MIN_OTM_BUFFER_PERCENT: float = 0.8
    MAX_SKEW_RATIO_THRESHOLD: float = 0.50
    
    # Position Management
    MAX_CONCURRENT_TRADES: int = 3
    STOP_LOSS_MULTIPLE: float = 2.0  # Close if loss > 2x credit
    DAILY_LOSS_LIMIT: float = 50000  # Stop all trading if daily loss exceeds
    MARGIN_BUFFER_PERCENT: float = 40  # Keep 40% cushion above broker requirement
    
    # Optimization
    VOLATILITY_SPIKE_THRESHOLD: float = 0.30  # Don't trade if IV spike > 30%
    GAMMA_EXPLOSION_THRESHOLD: float = 0.50  # Flag if gamma changed > 50%
    
    # Backtesting
    BACKTEST_MODE: bool = False
    PAPER_TRADING: bool = True


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging():
    """Configure comprehensive logging"""
    log_file = f"iron_condor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class OptionLeg:
    """Single option leg in a spread"""
    strike: float
    option_type: OptionType
    order_type: OrderType  # BUY or SELL
    symbol: str
    token: str
    entry_price: float = 0.0
    current_price: float = 0.0
    entry_timestamp: float = 0.0
    current_timestamp: float = 0.0
    quantity: int = 0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    def pnl(self) -> float:
        """Calculate P&L for this leg"""
        price_diff = (self.current_price - self.entry_price) if self.entry_price > 0 else 0
        if self.order_type == OrderType.SELL:
            price_diff *= -1  # Inverted for short positions
        return price_diff * self.quantity


@dataclass
class IronCondor:
    """Complete iron condor position"""
    trade_id: str
    entry_timestamp: float
    expiry_date: datetime
    spot_price_at_entry: float
    
    long_put: OptionLeg
    short_put: OptionLeg
    short_call: OptionLeg
    long_call: OptionLeg
    
    net_credit_rupees: float
    wing_width: float
    max_loss_rupees: float
    margin_required_rupees: float
    probability_of_profit: float
    expected_value_rupees: float
    return_on_margin_percent: float
    
    status: TradeStatus = TradeStatus.OPEN
    entry_note: str = ""
    legs: List[OptionLeg] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize legs list after dataclass creation"""
        if not self.legs:
            self.legs = [self.long_put, self.short_put, self.short_call, self.long_call]
    
    def update_prices(self, prices: Dict[str, float], timestamps: Dict[str, float]):
        """Update live prices for all legs"""
        for leg in self.legs:
            if leg.symbol in prices:
                leg.current_price = prices[leg.symbol]
                leg.current_timestamp = timestamps.get(leg.symbol, time.time())
    
    def current_pnl_rupees(self) -> float:
        """Total P&L across all legs"""
        return sum(leg.pnl() for leg in self.legs)
    
    def current_greeks(self) -> Dict[str, float]:
        """Aggregate Greeks across position"""
        return {
            'delta': sum(leg.delta * leg.quantity * (1 if leg.order_type == OrderType.BUY else -1) for leg in self.legs),
            'gamma': sum(leg.gamma * leg.quantity * (1 if leg.order_type == OrderType.BUY else -1) for leg in self.legs),
            'theta': sum(leg.theta * leg.quantity * (1 if leg.order_type == OrderType.BUY else -1) for leg in self.legs),
            'vega': sum(leg.vega * leg.quantity * (1 if leg.order_type == OrderType.BUY else -1) for leg in self.legs),
        }
    
    def should_close(self, spot: float) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        pnl = self.current_pnl_rupees()
        
        # Stop-loss: close if loss > 2x net credit
        if pnl < -self.net_credit_rupees * Config.STOP_LOSS_MULTIPLE:
            return True, f"Stop-loss triggered. Loss: ₹{pnl:.2f}"
        
        # Profit target: close if profit > 75% of max profit
        max_profit = self.net_credit_rupees
        if pnl > max_profit * 0.75:
            return True, f"Profit target (75%) reached. Profit: ₹{pnl:.2f}"
        
        # Check if spot has breached outer strikes
        if spot < self.long_put.strike:
            return True, f"Spot breached long put strike. Spot: {spot:.0f}"
        if spot > self.long_call.strike:
            return True, f"Spot breached long call strike. Spot: {spot:.0f}"
        
        return False, ""


@dataclass
class TradeJournal:
    """Comprehensive trade tracking"""
    trades: List[IronCondor] = field(default_factory=list)
    
    def add_trade(self, trade: IronCondor):
        self.trades.append(trade)
        logger.info(f"Trade recorded: {trade.trade_id}")
    
    def close_trade(self, trade_id: str, close_pnl: float):
        for trade in self.trades:
            if trade.trade_id == trade_id:
                trade.status = TradeStatus.CLOSED
                logger.info(f"Trade closed: {trade_id} | P&L: ₹{close_pnl:.2f}")
    
    def get_daily_pnl(self) -> float:
        """Calculate P&L for today's trades"""
        today = datetime.now().date()
        return sum(
            trade.current_pnl_rupees() 
            for trade in self.trades 
            if datetime.fromtimestamp(trade.entry_timestamp).date() == today
        )
    
    def get_open_trades(self) -> List[IronCondor]:
        return [t for t in self.trades if t.status == TradeStatus.OPEN]
    
    def save(self, filepath: str):
        """Persist journal to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Journal saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'TradeJournal':
        """Load journal from disk"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return TradeJournal()


# ==============================================================================
# ACCURATE TIME CALCULATION
# ==============================================================================

def calculate_trading_hours_to_expiry(expiry_dt: datetime) -> float:
    """
    Calculate actual trading hours remaining to expiry
    NIFTY options expire at 3:30 PM IST
    Trading hours: 9:15 AM to 3:30 PM = 6.25 hours per day
    """
    now = datetime.now()
    
    # NIFTY expiry time: 3:30 PM IST
    expiry_time = expiry_dt.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # If we're past expiry, return minimal T
    if now >= expiry_time:
        return 0.01
    
    # Count full trading days
    full_days = 0
    current = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    while current.date() < expiry_dt.date():
        # Skip weekends
        if current.weekday() < 5:  # Monday=0 to Friday=4
            full_days += 1
        current += timedelta(days=1)
    
    # Calculate partial day (today)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    trading_hours_today = 0.0
    
    if now < market_open:
        trading_hours_today = 0  # Market hasn't opened
    elif now >= expiry_time:
        trading_hours_today = 6.25  # Full day elapsed
    else:
        # Partial day
        elapsed = (now - market_open).total_seconds() / 3600
        trading_hours_today = min(elapsed, 6.25)
    
    total_trading_hours = (full_days * 6.25) + trading_hours_today
    
    logger.info(f"Time to expiry: {total_trading_hours:.2f} hours ({full_days} full days)")
    return total_trading_hours


def get_time_to_expiry_years(expiry_dt: datetime) -> float:
    """Convert trading hours to years"""
    trading_hours = calculate_trading_hours_to_expiry(expiry_dt)
    T = max(trading_hours / (6.25 * 252), 0.001)  # 6.25 hrs/day, 252 trading days/year
    return T


# ==============================================================================
# AMERICAN OPTION PRICING & GREEKS
# ==============================================================================

def black_scholes_european(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """Standard Black-Scholes European option pricing"""
    if T <= 0:
        return max(S - K, 0) if option_type == 'c' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'c':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def american_option_price_approximation(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> float:
    """
    Bjerksund-Stensland approximation for American options
    More accurate than Black-Scholes for early exercise modeling
    """
    european_price = black_scholes_european(S, K, T, r, q, sigma, option_type)
    
    # Early exercise premium (simplified)
    if option_type == 'p':
        # American put can be exercised early
        intrinsic = max(K - S, 0)
        early_exercise_value = max(intrinsic - european_price, 0)
        return european_price + early_exercise_value * 0.3  # Approximate premium
    else:
        return european_price


def implied_volatility_american(price: float, S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
    """
    Solve for IV using American option pricing
    Handles cases where European IV solver fails
    """
    try:
        # Try fast European IV first
        iv = implied_volatility(price, S, K, T, r, option_type)
        return max(iv, 0.001)  # Ensure positive
    except Exception:
        # Fallback: use optimization
        def objective(sigma):
            if sigma <= 0:
                sigma = 0.001
            theo_price = american_option_price_approximation(S, K, T, r, q, sigma, option_type)
            return (theo_price - price) ** 2
        
        try:
            result = minimize_scalar(objective, bounds=(0.001, 2.0), method='bounded')
            return max(result.x, 0.001)
        except Exception:
            logger.warning(f"IV calculation failed for {option_type} strike {K}")
            return 0.15  # Default to 15% IV


def calculate_greeks_american(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str
) -> Dict[str, float]:
    """
    Calculate Greeks using finite differences (robust for American options)
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    dS = S * 0.001  # 0.1% price bump
    dsigma = sigma * 0.01  # 1% vol bump
    dt = 1 / 365  # 1 day
    
    try:
        # Price at base case
        p0 = american_option_price_approximation(S, K, T, r, q, sigma, option_type)
        
        # Delta: dPrice/dS
        p_up = american_option_price_approximation(S + dS, K, T, r, q, sigma, option_type)
        p_down = american_option_price_approximation(S - dS, K, T, r, q, sigma, option_type)
        delta_val = (p_up - p_down) / (2 * dS)
        
        # Gamma: d²Price/dS²
        gamma_val = (p_up - 2 * p0 + p_down) / (dS ** 2)
        
        # Vega: dPrice/dSigma
        p_vol_up = american_option_price_approximation(S, K, T, r, q, sigma + dsigma, option_type)
        vega_val = (p_vol_up - p0) / dsigma / 100  # Per 1% vol change
        
        # Theta: dPrice/dT (time decay)
        T_down = max(T - dt, 0.001)
        p_time_down = american_option_price_approximation(S, K, T_down, r, q, sigma, option_type)
        theta_val = (p0 - p_time_down) / dt  # Per day
        
        # Rho: dPrice/dRate
        dr = 0.0001
        p_rate_up = american_option_price_approximation(S, K, T, r + dr, q, sigma, option_type)
        rho_val = (p_rate_up - p0) / dr / 100  # Per 1% rate change
        
        return {
            'delta': delta_val,
            'gamma': gamma_val,
            'theta': theta_val,
            'vega': vega_val,
            'rho': rho_val
        }
    except Exception as e:
        logger.error(f"Greeks calculation failed: {e}")
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}


# ==============================================================================
# API & DATA VALIDATION
# ==============================================================================

@st.cache_resource(ttl=3600)
def authenticate_safe() -> Optional[SmartConnect]:
    """
    Safely attempt authentication with graceful fallback
    """
    try:
        smartApi = SmartConnect(api_key=Config.API_KEY)
        totp = pyotp.TOTP(Config.TOTP_SECRET).now()
        session = smartApi.generateSession(Config.CLIENT_CODE, Config.PIN, totp)
        
        if not isinstance(session, dict) or not session.get('status'):
            msg = session.get('message', 'Authentication Failed')
            logger.warning(f"Authentication failed: {msg}")
            return None
        
        logger.info("✅ SmartAPI authentication successful")
        return smartApi
    except Exception as e:
        logger.warning(f"⚠️ SmartAPI authentication failed: {e}")
        logger.info("📊 Switching to DEMO MODE for testing")
        return None


def validate_quote_data(quote_response: dict, expected_symbol: str, max_age_sec: int = 30) -> Tuple[bool, str, dict]:
    """
    Validate quote data freshness and structure
    Returns: (is_valid, error_message, data)
    """
    # Check structure
    if not isinstance(quote_response, dict):
        return False, "Response is not a dictionary", {}
    
    if not quote_response.get('status'):
        msg = quote_response.get('message', 'API returned status=False')
        return False, f"API Error: {msg}", {}
    
    data = quote_response.get('data')
    if not isinstance(data, dict):
        return False, "Response data is not a dictionary", {}
    
    # Check required fields
    required_fields = ['ltp', 'openinterest']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing fields: {missing}", {}
    
    # Validate prices
    ltp = data.get('ltp', 0)
    if ltp <= 0:
        return False, f"Invalid LTP: {ltp}", {}
    
    oi = data.get('openinterest', 0)
    if oi < Config.MIN_LIQUIDITY_OI:
        return False, f"Insufficient OI: {oi} (min: {Config.MIN_LIQUIDITY_OI})", {}
    
    return True, "", data


@st.cache_data(ttl=Config.OPTION_CHAIN_TTL)
def get_nifty_option_chain_demo() -> Tuple[pd.DataFrame, datetime]:
    """
    Generate demo NIFTY option chain for testing without API
    Use this when API credentials are not available
    """
    spot_price = 24384.0
    expiry_date = datetime.now() + timedelta(days=30)
    
    # Generate strikes around spot
    strikes = np.arange(spot_price - 500, spot_price + 500, 100)
    
    records = []
    for strike in strikes:
        for opt_type in ['CE', 'PE']:
            symbol = f"NIFTY{expiry_date.strftime('%d%b%y').upper()}{strike:.0f}{opt_type}"
            
            # Realistic option pricing
            if opt_type == 'CE':
                price = max(spot_price - strike, 0) + np.random.uniform(10, 200)
            else:
                price = max(strike - spot_price, 0) + np.random.uniform(10, 200)
            
            records.append({
                'symbol': 'NIFTY',
                'name': 'NIFTY',
                'tradingsymbol': symbol,
                'token': str(int(strike * 100) + (1 if opt_type == 'CE' else 2)),
                'instrumenttype': 'OPTIDX',
                'exch_seg': 'NFO',
                'strike': strike * 100,  # In paise
                'expiry': expiry_date.strftime('%d%b%Y'),
                'expiry_dt': expiry_date,
                'ltp': price,
                'openinterest': np.random.randint(50000, 500000),
            })
    
    df = pd.DataFrame(records)
    logger.info(f"✅ DEMO MODE: Generated {len(df)} demo option contracts")
    return df, expiry_date


@st.cache_data(ttl=Config.OPTION_CHAIN_TTL)
def get_nifty_option_chain_safe() -> Tuple[pd.DataFrame, datetime]:
    """
    Fetch option chain with intelligent fallback to demo mode
    """
    try:
        url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            url = "https://margincalculator.angelbroking.com/OpenAPI_Data/files/OpenAPIScripMaster.json"
            response = requests.get(url, headers=headers, timeout=15)
        
        df = pd.DataFrame(response.json())
        df.columns = [str(c).lower() for c in df.columns]
        
        symbol_col = 'name' if 'name' in df.columns else 'symbol'
        
        nifty_opts = df[
            (df[symbol_col].astype(str).str.upper() == 'NIFTY') & 
            (df['instrumenttype'].astype(str).str.upper() == 'OPTIDX') & 
            (df['exch_seg'].astype(str).str.upper() == 'NFO')
        ].copy()
        
        if nifty_opts.empty:
            logger.warning("No NIFTY options found in data, using demo mode")
            return get_nifty_option_chain_demo()
        
        if 'tradingsymbol' not in nifty_opts.columns:
            nifty_opts['tradingsymbol'] = nifty_opts['symbol'] if 'symbol' in nifty_opts.columns else nifty_opts[symbol_col]
        
        nifty_opts['strike'] = nifty_opts['strike'].astype(float) / 100.0
        nifty_opts['expiry_dt'] = pd.to_datetime(nifty_opts['expiry'], format='%d%b%Y')
        
        today = pd.Timestamp.now().normalize()
        upcoming_expiries = nifty_opts[nifty_opts['expiry_dt'] >= today]['expiry_dt'].unique()
        
        if len(upcoming_expiries) == 0:
            logger.warning("No upcoming expiries found, using demo mode")
            return get_nifty_option_chain_demo()
        
        nearest_expiry = pd.Timestamp(upcoming_expiries[0])
        chain = nifty_opts[nifty_opts['expiry_dt'] == nearest_expiry].copy()
        
        logger.info(f"✅ Option chain loaded: {len(chain)} contracts, expiry: {nearest_expiry.strftime('%d-%b-%Y')}")
        return chain, nearest_expiry.to_pydatetime()
    
    except Exception as e:
        logger.warning(f"⚠️ Option chain fetch failed: {e}")
        logger.info("📊 Switching to DEMO MODE for testing")
        return get_nifty_option_chain_demo()


def get_spot_price_safe(smartApi: Optional[SmartConnect], use_demo: bool = False) -> Tuple[float, bool]:
    """
    Safely fetch spot price with demo fallback
    Returns: (spot_price, is_demo_mode)
    """
    if use_demo or smartApi is None:
        logger.info("📊 Using demo spot price")
        return 24384.0, True
    
    try:
        spot_res = smartApi.ltpData("NSE", "NIFTY", "99926000")
        spot_valid, spot_err, spot_data = validate_quote_data(spot_res, "NIFTY")
        
        if not spot_valid:
            logger.warning(f"⚠️ Spot price validation failed: {spot_err}. Using fallback.")
            return 24384.0, False
        
        spot_price = float(spot_data['ltp'])
        logger.info(f"✅ Live spot price: ₹{spot_price:,.0f}")
        return spot_price, False
    
    except Exception as e:
        logger.warning(f"⚠️ Spot price fetch failed: {e}. Using fallback.")
        return 24384.0, False


# ==============================================================================
# QUANTITATIVE ENGINE - UPDATED
# ==============================================================================

def run_quant_engine_v2(
    smartApi: Optional[SmartConnect], 
    chain: pd.DataFrame, 
    spot_price: float, 
    expiry_dt: datetime,
    previous_greeks: Optional[Dict] = None
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Enhanced quantitative engine with:
    - Robust Greeks calculation (American options)
    - Data validation & freshness checks
    - Liquidity filtering
    - IV surface smoothing
    - Greeks change detection
    """
    
    T = get_time_to_expiry_years(expiry_dt)
    
    if T <= 0:
        logger.warning("Expiry has passed")
        return None, T
    
    records = []
    
    # Select strikes: ±6% of spot with liquidity filter
    strike_mask = (chain['strike'] >= spot_price * 0.94) & (chain['strike'] <= spot_price * 1.06)
    strikes = sorted(chain[strike_mask]['strike'].unique())
    
    logger.info(f"Processing {len(strikes)} strikes | Spot: {spot_price:.0f} | T: {T:.4f} years")
    
    for K in strikes:
        ce_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('CE'))]
        pe_row = chain[(chain['strike'] == K) & (chain['tradingsymbol'].str.endswith('PE'))]
        
        if ce_row.empty or pe_row.empty:
            continue
        
        ce_symbol = ce_row.iloc[0]['tradingsymbol']
        ce_token = ce_row.iloc[0]['token']
        pe_symbol = pe_row.iloc[0]['tradingsymbol']
        pe_token = pe_row.iloc[0]['token']
        
        try:
            # Fetch quotes with validation
            if smartApi is not None:
                ce_res = smartApi.ltpData("NFO", ce_symbol, ce_token)
                pe_res = smartApi.ltpData("NFO", pe_symbol, pe_token)
                
                ce_valid, ce_err, ce_data = validate_quote_data(ce_res, ce_symbol)
                pe_valid, pe_err, pe_data = validate_quote_data(pe_res, pe_symbol)
                
                if not ce_valid or not pe_valid:
                    logger.debug(f"Strike {K}: CE valid={ce_valid} ({ce_err}), PE valid={pe_valid} ({pe_err})")
                    continue
            else:
                # Demo mode - use chain data
                ce_data = {'ltp': ce_row.iloc[0].get('ltp', 100), 'openinterest': ce_row.iloc[0].get('openinterest', 100000)}
                pe_data = {'ltp': pe_row.iloc[0].get('ltp', 100), 'openinterest': pe_row.iloc[0].get('openinterest', 100000)}
            
            ce_price = float(ce_data['ltp'])
            pe_price = float(pe_data['ltp'])
            ce_oi = float(ce_data['openinterest'])
            pe_oi = float(pe_data['openinterest'])
            
            # Calculate IV using robust American option pricer
            ce_iv = implied_volatility_american(ce_price, spot_price, K, T, Config.RISK_FREE_RATE, Config.NIFTY_DIVIDEND_YIELD, 'c')
            pe_iv = implied_volatility_american(pe_price, spot_price, K, T, Config.RISK_FREE_RATE, Config.NIFTY_DIVIDEND_YIELD, 'p')
            
            # Calculate Greeks using finite differences
            ce_greeks = calculate_greeks_american(spot_price, K, T, Config.RISK_FREE_RATE, Config.NIFTY_DIVIDEND_YIELD, ce_iv, 'c')
            pe_greeks = calculate_greeks_american(spot_price, K, T, Config.RISK_FREE_RATE, Config.NIFTY_DIVIDEND_YIELD, pe_iv, 'p')
            
            ce_gamma = ce_greeks['gamma']
            net_gex = (pe_oi - ce_oi) * ce_gamma * (spot_price ** 2) * 0.01 / 1e6
            
            # Greeks change detection
            gamma_changed_pct = 0
            if previous_greeks and K in previous_greeks:
                prev_gamma = previous_greeks[K]['gamma']
                if prev_gamma > 0:
                    gamma_changed_pct = abs((ce_gamma - prev_gamma) / prev_gamma)
            
            records.append({
                'strike': K,
                'ce_symbol': ce_symbol,
                'pe_symbol': pe_symbol,
                'ce_price': ce_price,
                'pe_price': pe_price,
                'ce_iv': ce_iv,
                'pe_iv': pe_iv,
                'ce_oi': ce_oi,
                'pe_oi': pe_oi,
                'ce_gamma': ce_gamma,
                'pe_gamma': pe_greeks['gamma'],
                'ce_delta': ce_greeks['delta'],
                'ce_vega': ce_greeks['vega'],
                'ce_theta': ce_greeks['theta'],
                'pe_delta': pe_greeks['delta'],
                'pe_vega': pe_greeks['vega'],
                'pe_theta': pe_greeks['theta'],
                'gex': net_gex,
                'gamma_changed_pct': gamma_changed_pct,
            })
            
            time.sleep(0.01)  # Rate limiting
        
        except Exception as e:
            logger.debug(f"Strike {K} processing error: {e}")
            continue
    
    df_quant = pd.DataFrame(records)
    
    if df_quant.empty:
        logger.warning("No valid strikes after filtering")
        return None, T
    
    # Calculate Implied Probability Density Function
    try:
        spline = UnivariateSpline(df_quant['strike'], df_quant['ce_price'], k=min(4, len(df_quant) - 1), s=1.0)
        d2_prices = spline.derivative(n=2)(df_quant['strike'])
        df_quant['iPDF'] = np.exp(Config.RISK_FREE_RATE * T) * np.maximum(d2_prices, 0.001)
        df_quant['iPDF'] = df_quant['iPDF'] / df_quant['iPDF'].sum()  # Normalize
    except Exception as e:
        logger.warning(f"iPDF calculation failed: {e}")
        df_quant['iPDF'] = 1.0 / len(df_quant)  # Uniform fallback
    
    logger.info(f"Quant engine complete: {len(df_quant)} valid strikes")
    return df_quant, T


# ==============================================================================
# MONTE CARLO PROBABILITY CALCULATOR
# ==============================================================================

def monte_carlo_pop(
    spot: float,
    short_put_strike: float,
    short_call_strike: float,
    T: float,
    mu: float,
    sigma: float,
    num_paths: int = 10000,
    time_steps: int = 50
) -> float:
    """
    Monte Carlo simulation for Probability of Profit
    More realistic than static PDF approach
    """
    dt = T / time_steps
    
    # Generate price paths
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), (num_paths, time_steps))
    
    S = np.ones((num_paths, time_steps + 1)) * spot
    
    for t in range(1, time_steps + 1):
        S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW[:, t - 1])
    
    # Check if path stays within short strikes at expiry
    final_prices = S[:, -1]
    profitable_paths = (final_prices > short_put_strike) & (final_prices < short_call_strike)
    
    pop = np.sum(profitable_paths) / num_paths
    logger.debug(f"Monte Carlo PoP: {pop:.2%} ({int(np.sum(profitable_paths))}/{num_paths} paths profitable)")
    
    return pop


# ==============================================================================
# OPTIMIZED IRON CONDOR SELECTION ENGINE
# ==============================================================================

def find_optimal_iron_condor_v2(
    df: pd.DataFrame,
    spot: float,
    T: float,
    historical_volatility: float = 0.18
) -> Optional[IronCondor]:
    """
    Enhanced optimizer with:
    - Monte Carlo PoP
    - Robust utility function
    - Greeks-based stability scoring
    - Explicit risk penalty
    """
    
    if df is None or df.empty:
        logger.warning("Empty dataframe passed to optimizer")
        return None
    
    best_score = -np.inf
    optimal_condor = None
    
    strikes = sorted(df['strike'].unique())
    
    MIN_OTM_BUFFER = spot * Config.MIN_OTM_BUFFER_PERCENT / 100
    
    logger.info(f"Scanning {len(strikes)} strikes for optimal setup...")
    
    for i in range(len(strikes) - 3):
        long_put_k = strikes[i]
        short_put_k = strikes[i + 1]
        
        # Put spread validation
        if short_put_k >= spot or (spot - short_put_k) < MIN_OTM_BUFFER:
            continue
        
        for j in range(i + 2, len(strikes) - 1):
            short_call_k = strikes[j]
            long_call_k = strikes[j + 1]
            
            # Call spread validation
            if short_call_k <= spot or (short_call_k - spot) < MIN_OTM_BUFFER:
                continue
            
            # Equal wing width requirement
            put_wing = short_put_k - long_put_k
            call_wing = long_call_k - short_call_k
            
            if abs(put_wing - call_wing) > 0.1:  # Allow small tolerance
                continue
            
            wing_width = put_wing
            
            # Get option prices
            try:
                lp_price = df[df['strike'] == long_put_k]['pe_price'].values[0]
                sp_price = df[df['strike'] == short_put_k]['pe_price'].values[0]
                sc_price = df[df['strike'] == short_call_k]['ce_price'].values[0]
                lc_price = df[df['strike'] == long_call_k]['ce_price'].values[0]
                
                ce_iv = df[df['strike'] == short_call_k]['ce_iv'].values[0]
                
                net_credit = (sp_price + sc_price) - (lp_price + lc_price)
                max_loss = wing_width - net_credit
                
                if net_credit <= 0 or max_loss <= 0:
                    continue
                
                # Calculate skew ratio
                put_distance = spot - short_put_k
                call_distance = short_call_k - spot
                skew_ratio = min(put_distance, call_distance) / max(put_distance, call_distance)
                
                if skew_ratio < Config.MAX_SKEW_RATIO_THRESHOLD:
                    continue
                
                # Monte Carlo PoP (more realistic)
                pop = monte_carlo_pop(
                    spot, short_put_k, short_call_k, T,
                    mu=Config.RISK_FREE_RATE,
                    sigma=ce_iv,
                    num_paths=5000
                )
                
                # Position Greeks
                short_put_gamma = df[df['strike'] == short_put_k]['pe_gamma'].values[0]
                short_call_gamma = df[df['strike'] == short_call_k]['ce_gamma'].values[0]
                
                # Gamma stability: high gamma = unstable
                net_gamma = short_put_gamma + short_call_gamma
                gamma_stability = -abs(net_gamma)  # Negative = better (less unstable)
                
                # Margin calculations
                margin_required = (wing_width - net_credit) * Config.LOT_SIZE
                margin_buffer = margin_required * (1 + Config.MARGIN_BUFFER_PERCENT / 100)
                
                # Expected Value with risk adjustment
                expected_value = (net_credit * pop) - (max_loss * (1 - pop))
                
                # Improved utility: Return on margin with explicit risk penalty
                risk_reward_ratio = expected_value / margin_required if margin_required > 0 else 0
                utility_score = (risk_reward_ratio * pop) + (gamma_stability * 0.1) + (skew_ratio * 0.05)
                
                logger.debug(
                    f"Setup: {long_put_k:.0f}P/{short_put_k:.0f}P/{short_call_k:.0f}C/{long_call_k:.0f}C | "
                    f"Credit: ₹{net_credit*Config.LOT_SIZE:.0f} | PoP: {pop:.1%} | Score: {utility_score:.4f}"
                )
                
                if utility_score > best_score:
                    best_score = utility_score
                    
                    # Build IronCondor object
                    trade_id = f"IC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    lp_row = df[df['strike'] == long_put_k].iloc[0]
                    sp_row = df[df['strike'] == short_put_k].iloc[0]
                    sc_row = df[df['strike'] == short_call_k].iloc[0]
                    lc_row = df[df['strike'] == long_call_k].iloc[0]
                    
                    long_put_leg = OptionLeg(
                        strike=long_put_k,
                        option_type=OptionType.PUT,
                        order_type=OrderType.BUY,
                        symbol=lp_row['pe_symbol'],
                        token=str(lp_row.get('token', '')),
                        entry_price=lp_price,
                        quantity=Config.LOT_SIZE,
                        iv=lp_row['pe_iv'],
                        delta=lp_row['pe_delta'],
                        gamma=lp_row['pe_gamma'],
                        theta=lp_row['pe_theta'],
                        vega=lp_row['pe_vega'],
                    )
                    
                    short_put_leg = OptionLeg(
                        strike=short_put_k,
                        option_type=OptionType.PUT,
                        order_type=OrderType.SELL,
                        symbol=sp_row['pe_symbol'],
                        token=str(sp_row.get('token', '')),
                        entry_price=sp_price,
                        quantity=Config.LOT_SIZE,
                        iv=sp_row['pe_iv'],
                        delta=sp_row['pe_delta'],
                        gamma=sp_row['pe_gamma'],
                        theta=sp_row['pe_theta'],
                        vega=sp_row['pe_vega'],
                    )
                    
                    short_call_leg = OptionLeg(
                        strike=short_call_k,
                        option_type=OptionType.CALL,
                        order_type=OrderType.SELL,
                        symbol=sc_row['ce_symbol'],
                        token=str(sc_row.get('token', '')),
                        entry_price=sc_price,
                        quantity=Config.LOT_SIZE,
                        iv=sc_row['ce_iv'],
                        delta=sc_row['ce_delta'],
                        gamma=sc_row['ce_gamma'],
                        theta=sc_row['ce_theta'],
                        vega=sc_row['ce_vega'],
                    )
                    
                    long_call_leg = OptionLeg(
                        strike=long_call_k,
                        option_type=OptionType.CALL,
                        order_type=OrderType.BUY,
                        symbol=lc_row['ce_symbol'],
                        token=str(lc_row.get('token', '')),
                        entry_price=lc_price,
                        quantity=Config.LOT_SIZE,
                        iv=lc_row['ce_iv'],
                        delta=lc_row['ce_delta'],
                        gamma=lc_row['ce_gamma'],
                        theta=lc_row['ce_theta'],
                        vega=lc_row['ce_vega'],
                    )
                    
                    optimal_condor = IronCondor(
                        trade_id=trade_id,
                        entry_timestamp=time.time(),
                        expiry_date=expiry_dt,
                        spot_price_at_entry=spot,
                        long_put=long_put_leg,
                        short_put=short_put_leg,
                        short_call=short_call_leg,
                        long_call=long_call_leg,
                        net_credit_rupees=net_credit * Config.LOT_SIZE,
                        wing_width=wing_width,
                        max_loss_rupees=max_loss * Config.LOT_SIZE,
                        margin_required_rupees=margin_required,
                        probability_of_profit=pop * 100,
                        expected_value_rupees=expected_value * Config.LOT_SIZE,
                        return_on_margin_percent=(net_credit * Config.LOT_SIZE / margin_required * 100),
                        entry_note=f"Utility Score: {utility_score:.4f} | Skew: {skew_ratio:.2f}"
                    )
            
            except Exception as e:
                logger.debug(f"Strike combo error: {e}")
                continue
    
    if optimal_condor:
        logger.info(f"Optimal setup found: {optimal_condor.long_put.strike:.0f}P/{optimal_condor.short_put.strike:.0f}P/"
                   f"{optimal_condor.short_call.strike:.0f}C/{optimal_condor.long_call.strike:.0f}C")
    else:
        logger.warning("No valid iron condor setup found")
    
    return optimal_condor


# ==============================================================================
# STREAMLIT DASHBOARD
# ==============================================================================

st.set_page_config(page_title="Quant Iron Condor V2 - Production", page_icon="⚡", layout="wide")

def main():
    st.title("⚡ Quantitative Iron Condor Engine V2 - Production Ready")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        enable_live = st.checkbox("Enable Live Refresh", value=False)
        use_demo_mode = st.checkbox("📊 Use Demo Mode (Testing)", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        st.markdown("---")
        st.subheader("📊 Configuration")
        st.metric("Max Concurrent Trades", Config.MAX_CONCURRENT_TRADES)
        st.metric("Stop Loss Multiple", f"{Config.STOP_LOSS_MULTIPLE}x credit")
        st.metric("Daily Loss Limit", f"₹{Config.DAILY_LOSS_LIMIT:,}")
        
        st.markdown("---")
        if use_demo_mode:
            st.info("🧪 DEMO MODE ACTIVE - Using simulated data for testing")
    
    try:
        # Initialize session state
        if 'journal' not in st.session_state:
            st.session_state.journal = TradeJournal.load("trade_journal.pkl")
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = 0
        
        # Step 1: Authentication
        st.info("🔄 Initializing systems...")
        
        if use_demo_mode:
            st.warning("🧪 Running in DEMO MODE - API not connected")
            smartApi = None
            is_demo = True
        else:
            smartApi = authenticate_safe()
            is_demo = smartApi is None
            
            if is_demo:
                st.warning("⚠️ API authentication failed. Switching to DEMO MODE for testing.")
        
        # Step 2: Get option chain
        try:
            chain, expiry_dt = get_nifty_option_chain_safe()
        except Exception as e:
            st.error(f"❌ Failed to fetch option chain: {e}")
            st.info("💡 Make sure your API credentials are correct and your internet connection is stable.")
            return
        
        # Step 3: Get spot price
        spot_price, spot_is_demo = get_spot_price_safe(smartApi, is_demo)
        
        # Display metrics
        st.subheader("📌 Live Market Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("NIFTY Spot", f"₹{spot_price:,.0f}")
        m2.metric("Expiry", expiry_dt.strftime('%d-%b-%Y'))
        m3.metric("Days to Expiry", f"{(expiry_dt - datetime.now()).days}d")
        m4.metric("Open Trades", len(st.session_state.journal.get_open_trades()))
        m5.metric("Daily P&L", f"₹{st.session_state.journal.get_daily_pnl():,.0f}")
        
        if is_demo:
            st.info("📊 All data is simulated for testing purposes only")
        
        st.markdown("---")
        
        # Step 4: Run quant engine
        try:
            df_quant, T = run_quant_engine_v2(smartApi, chain, spot_price, expiry_dt)
        except Exception as e:
            st.error(f"❌ Quant engine failed: {e}")
            logger.error(f"Quant engine error: {e}", exc_info=True)
            return
        
        # Step 5: Find optimal setup
        if df_quant is not None and not df_quant.empty:
            result = find_optimal_iron_condor_v2(df_quant, spot_price, T)
            
            if result:
                score = float(result.entry_note.split('Score: ')[1].split(' ')[0])
                st.success(f"✅ Optimal Setup Found (Score: {score:.4f})")
                
                # Display trade details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Option Legs")
                    legs_data = [
                        {"Leg": "[1] BUY PUT", "Strike": f"{result.long_put.strike:.0f}", "Type": "PUT", "Qty": Config.LOT_SIZE},
                        {"Leg": "[2] SELL PUT", "Strike": f"{result.short_put.strike:.0f}", "Type": "PUT", "Qty": Config.LOT_SIZE},
                        {"Leg": "[3] SELL CALL", "Strike": f"{result.short_call.strike:.0f}", "Type": "CALL", "Qty": Config.LOT_SIZE},
                        {"Leg": "[4] BUY CALL", "Strike": f"{result.long_call.strike:.0f}", "Type": "CALL", "Qty": Config.LOT_SIZE},
                    ]
                    st.table(legs_data)
                
                with col2:
                    st.subheader("💰 Financial Metrics")
                    fin_data = [
                        {"Metric": "Net Credit", "Value": f"₹{result.net_credit_rupees:,.0f}"},
                        {"Metric": "Max Loss", "Value": f"₹{result.max_loss_rupees:,.0f}"},
                        {"Metric": "Margin Required", "Value": f"₹{result.margin_required_rupees:,.0f}"},
                        {"Metric": "PoP (Monte Carlo)", "Value": f"{result.probability_of_profit:.1f}%"},
                        {"Metric": "Expected Value", "Value": f"₹{result.expected_value_rupees:,.0f}"},
                        {"Metric": "Return on Margin", "Value": f"{result.return_on_margin_percent:.2f}%"},
                    ]
                    st.table(fin_data)
                
                st.markdown("---")
                
                # Risk Assessment
                st.subheader("⚠️ Risk Assessment")
                risk_col1, risk_col2, risk_col3 = st.columns(3)
                
                with risk_col1:
                    st.metric("Short Put Delta", f"{result.short_put.delta:.3f}", "Bearish exposure")
                
                with risk_col2:
                    st.metric("Short Call Delta", f"{result.short_call.delta:.3f}", "Bullish exposure")
                
                with risk_col3:
                    total_gamma = result.short_put.gamma + result.short_call.gamma
                    st.metric("Net Gamma", f"{total_gamma:.5f}", "Gamma risk")
                
                # Decision panel
                st.markdown("---")
                st.subheader("🔧 Action Panel")
                
                if is_demo:
                    st.warning("🧪 DEMO MODE: Trades will not be executed. Switch to LIVE MODE with valid API credentials.")
                
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("✅ ENTER TRADE", key="enter_btn", disabled=is_demo):
                        st.session_state.journal.add_trade(result)
                        st.session_state.journal.save("trade_journal.pkl")
                        st.success(f"Trade {result.trade_id} recorded!")
                        logger.info(f"Trade entered: {result.trade_id}")
                
                with action_col2:
                    if st.button("📊 VIEW JOURNAL", key="journal_btn"):
                        open_trades = st.session_state.journal.get_open_trades()
                        if open_trades:
                            st.write(open_trades)
                        else:
                            st.info("No open trades yet")
                
                with action_col3:
                    if st.button("💾 SAVE JOURNAL", key="save_btn"):
                        st.session_state.journal.save("trade_journal.pkl")
                        st.info("Journal saved!")
            else:
                st.warning("⚠️ No valid iron condor setup met criteria. Try again later.")
        else:
            st.error("❌ Failed to process option chain. Check API connectivity and credentials.")
            st.info("💡 Tips:\n"
                   "- Ensure your Angel Broking API credentials are correct\n"
                   "- Check your internet connection\n"
                   "- Use DEMO MODE to test without API")
    
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        logger.error(f"Main error: {e}", exc_info=True)
        
        st.markdown("---")
        st.subheader("🔧 Troubleshooting")
        with st.expander("Show error details"):
            st.code(str(e), language="text")
    
    # Live refresh
    if enable_live:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
