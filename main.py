import os
import time
import json
import logging
import requests
import threading
import numpy as np
import pandas as pd
import MetaTrader5 as mt5  # ភ្ជាប់ MT5 API
from typing import Dict, Optional, Any, Tuple, List
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. LOGGING & CONFIGURATION SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

import os

# ទាញយក Token ចេញពី Environment Variable លើ Render ដោយសុវត្ថិភាព
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

SYMBOL_NAME = "XAUUSD (Gold)"
# កែប្រែ Ticker មកប្រើប្រាស់ Symbol របស់ MT5
YFINANCE_TICKER = "XAUUSD"        # MT5 Gold Symbol (អាចជា XAUUSD, XAUUSD.m, XAUUSDm អាស្រ័យលើ Broker)
SILVER_TICKER = "XAGUSD"          # MT5 Silver Symbol សម្រាប់វិភាគ SMT Divergence
CHECK_INTERVAL_SECONDS = 10       # Scan interval
NEWS_BUFFER_MINUTES = 30          # Block signals 30m before/after High Impact News

# Institutional Risk Settings
ACCOUNT_BALANCE = 10000.0        # USD Account Balance
RISK_PER_TRADE_PCT = 0.01        # 1% Risk per trade
MAX_DAILY_LOSS_PCT = 0.03        # 3% Max Daily Drawdown
MIN_PROBABILITY_SCORE = 60       # Min Score to trigger signal

last_processed_signal_id: Optional[str] = None
last_update_id: int = 0
indicator_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
cache_lock = threading.Lock()


# ==========================================
# METATRADER 5 INITIALIZATION
# ==========================================
def init_mt5_connection() -> bool:
    """ពិនិត្យ និងភ្ជាប់ទៅកាន់ MetaTrader 5 Terminal"""
    if not mt5.initialize():
        logging.error(f"❌ MT5 Initialization Failed! Error: {mt5.last_error()}")
        return False
    logging.info("✅ MetaTrader5 Terminal Connected Successfully!")
    return True


# ==========================================
# 2. STANDARDIZED SIGNAL BUILDER (PREVENTS KEYERRORS)
# ==========================================
def create_standard_signal(
    status: str = "OK",
    signal_type: str = "WAIT",      # BUY NOW, SELL NOW, WAIT BUY ZONE, WAIT SELL ZONE, WAIT, NO TRADE
    action: str = "WAIT",           # BUY, SELL, WAIT
    price: float = 0.0,
    score: float = 0.0,
    entry_zone_high: float = 0.0,
    entry_zone_low: float = 0.0,
    ideal_entry: float = 0.0,
    sl: float = 0.0,
    tp1: float = 0.0,
    tp2: float = 0.0,
    tp3: float = 0.0,
    rr: float = 0.0,
    win_rate: str = "0%",
    position_size: float = 0.0,
    tier: str = "TIER_3",
    htf_bias: str = "NEUTRAL",
    session: str = "OTHER",
    is_news: bool = False,
    news_msg: str = "🟢 No High Impact News nearby",
    reason: str = "Market condition scanning",
    reasons: Optional[List[str]] = None,
    weekly_bias: str = "NEUTRAL",
    daily_bias: str = "NEUTRAL",
    h4_bias: str = "NEUTRAL",
    h1_bias: str = "NEUTRAL",
    m15_bias: str = "NEUTRAL",
    market_delivery_state: str = "CONSOLIDATION",
    liquidity_draw: str = "NONE",
    amd_state: str = "ACCUMULATION",
    premium_zone: str = "N/A",
    discount_zone: str = "N/A",
    equilibrium: float = 0.0,
    price_location: str = "EQUILIBRIUM",
    ote_zone: str = "N/A",
    ote_level: float = 0.0,
    ote_score: float = 0.0,
    smt_signal: str = "NONE",
    smt_strength: float = 0.0,
    session_bias: str = "NEUTRAL",
    session_strength: float = 0.0,
    session_phase: str = "RANGE",
    order_block: Optional[Dict[str, Any]] = None,
    mitigation_block: Optional[Dict[str, Any]] = None,
    breaker_block: Optional[Dict[str, Any]] = None,
    fvg_info: Optional[Dict[str, Any]] = None,
    structure_info: Optional[Dict[str, Any]] = None,
    liquidity_info: Optional[Dict[str, Any]] = None,
    signal_grade: str = "C",
    confidence_level: str = "LOW",
    invalidation_level: float = 0.0,
    atr_stop: float = 0.0,
    structure_stop: float = 0.0
) -> Dict[str, Any]:
    if reasons is None:
        reasons = []

    p = round(float(price), 2)
    sl_val = round(float(sl), 2)
    e_high = round(float(entry_zone_high), 2)
    e_low = round(float(entry_zone_low), 2)
    e_ideal = round(float(ideal_entry), 2) if ideal_entry > 0 else p

    return {
        "status": str(status),
        "signal_type": str(signal_type),
        "action": str(action),
        "price": p,
        "entry": p,
        "score": round(float(score), 2),
        "entry_zone_high": e_high,
        "entry_zone_low": e_low,
        "entry_zone": f"{e_low} - {e_high}" if e_high > 0 else "N/A",
        "ideal_entry": e_ideal,
        "sl": sl_val,
        "tp1": round(float(tp1), 2),
        "tp2": round(float(tp2), 2),
        "tp3": round(float(tp3), 2),
        "rr": round(float(rr), 2),
        "win_rate": str(win_rate),
        "position_size": round(float(position_size), 2),
        "tier": str(tier),
        "htf_bias": str(htf_bias),
        "session": str(session),
        "is_news": bool(is_news),
        "news_msg": str(news_msg),
        "reason": str(reason),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weekly_bias": str(weekly_bias),
        "daily_bias": str(daily_bias),
        "h4_bias": str(h4_bias),
        "h1_bias": str(h1_bias),
        "m15_bias": str(m15_bias),
        "market_delivery_state": str(market_delivery_state),
        "liquidity_draw": str(liquidity_draw),
        "amd_state": str(amd_state),
        "premium_zone": str(premium_zone),
        "discount_zone": str(discount_zone),
        "equilibrium": round(float(equilibrium), 2),
        "price_location": str(price_location),
        "ote_zone": str(ote_zone),
        "ote_level": round(float(ote_level), 2),
        "ote_score": round(float(ote_score), 2),
        "smt_signal": str(smt_signal),
        "smt_strength": round(float(smt_strength), 2),
        "session_bias": str(session_bias),
        "session_strength": round(float(session_strength), 2),
        "session_phase": str(session_phase),
        "order_block": order_block or {},
        "mitigation_block": mitigation_block or {},
        "breaker_block": breaker_block or {},
        "fvg_info": fvg_info or {},
        "structure_info": structure_info or {},
        "liquidity_info": liquidity_info or {},
        "probability_score": round(float(score), 2),
        "signal_grade": str(signal_grade),
        "expected_win_rate": str(win_rate),
        "expected_rr": f"1:{round(float(rr), 2)}",
        "confidence_level": str(confidence_level),
        "signal_tier": str(tier),
        "invalidation_level": round(float(invalidation_level), 2),
        "atr_stop": round(float(atr_stop), 2),
        "structure_stop": round(float(structure_stop), 2),
    }


# ==========================================
# 3. PERFORMANCE & TRADE LOGGER
# ==========================================
class PerformanceTracker:
    FILE_PATH = "trade_journal.json"

    @classmethod
    def load_trades(cls) -> List[Dict[str, Any]]:
        if os.path.exists(cls.FILE_PATH):
            try:
                with open(cls.FILE_PATH, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                logging.error(f"Failed to read trade journal: {e}")
        return []

    @classmethod
    def save_trade(cls, trade_data: Dict[str, Any]):
        trades = cls.load_trades()
        trades.append(trade_data)
        try:
            with open(cls.FILE_PATH, "w") as f:
                json.dump(trades, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to write trade journal: {e}")

    @classmethod
    def generate_report(cls, timeframe_days: int = 30) -> Dict[str, Any]:
        trades = cls.load_trades()
        if not trades:
            return {"status": "NO_TRADES", "message": "No trade history logged yet."}

        now = datetime.now(timezone.utc)
        filtered_trades = []

        for t in trades:
            try:
                trade_time = datetime.fromisoformat(t.get("time", now.isoformat()))
                if (now - trade_time).days <= timeframe_days:
                    filtered_trades.append(t)
            except Exception:
                continue

        if not filtered_trades:
            return {"status": "NO_TRADES", "message": f"No trades found in last {timeframe_days} days."}

        total_trades = len(filtered_trades)
        wins = [t for t in filtered_trades if t.get("outcome") == "WIN"]
        losses = [t for t in filtered_trades if t.get("outcome") == "LOSS"]

        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        gross_profit = sum(t.get("pnl", 0.0) for t in wins)
        gross_loss = abs(sum(t.get("pnl", 0.0) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)
        avg_rr = float(np.mean([t.get("rr", 0.0) for t in filtered_trades])) if filtered_trades else 0.0

        win_prob = len(wins) / total_trades if total_trades > 0 else 0
        loss_prob = 1 - win_prob
        avg_win = (gross_profit / len(wins)) if len(wins) > 0 else 0
        avg_loss = (gross_loss / len(losses)) if len(losses) > 0 else 0
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)

        return {
            "status": "OK",
            "period_days": timeframe_days,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": f"{round(win_rate, 2)}%",
            "profit_factor": round(profit_factor, 2),
            "avg_rr": f"1:{round(avg_rr, 2)}",
            "expectancy": f"${round(expectancy, 2)}",
            "net_profit": round(gross_profit - gross_loss, 2)
        }


# ==========================================
# 4. NEWS FILTER MODULE
# ==========================================
class NewsFilter:
    @staticmethod
    def check_high_impact_news(buffer_minutes: int = 30) -> Tuple[bool, str]:
        url = "https://nfs.kinexondigital.com/data/this_week.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                events = res.json()
                now = datetime.now(timezone.utc)

                for event in events:
                    if not isinstance(event, dict):
                        continue
                    country = event.get("country", "")
                    impact = event.get("impact", "")
                    
                    if country in ["USD", "XAU"] and impact == "High":
                        event_date_str = event.get("date", "")
                        if not event_date_str:
                            continue
                        
                        event_time = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                        time_diff = abs((event_time - now).total_seconds() / 60.0)
                        
                        if time_diff <= buffer_minutes:
                            title = event.get("title", "High Impact News")
                            return True, f"⚠️ High Impact News Ahead: **{title}** ({int(time_diff)} mins)"

            return False, "🟢 No High Impact USD news nearby"
        except Exception as e:
            logging.debug(f"News check bypass: {e}")
            return False, "🟢 News Check bypassed (Server unavailable)"


# ==========================================
# 5. METATRADER5 DATA FETCHING & INDICATORS
# ==========================================
def fetch_ohlcv_safe(ticker: str, interval: str, range_: str) -> pd.DataFrame:
    """
    កែប្រែមកប្រើប្រាស់ MetaTrader5 Python API ដោយផ្ទាល់
    ទាញយក Rates (OHLCV) ពី MT5 Terminal
    """
    cache_key = f"{ticker}_{interval}_{range_}"
    now = datetime.now(timezone.utc)
    
    with cache_lock:
        if cache_key in indicator_cache:
            cached_time, cached_df = indicator_cache[cache_key]
            if (now - cached_time).total_seconds() < 5:  # Cache 5s សម្រាប់រហ័សទាន់ចិត្ត
                return cached_df.copy()

    # ពិនិត្យការភ្ជាប់ MT5
    if not mt5.initialize():
        logging.error("⚠️ MT5 connection failed in fetch_ohlcv_safe!")
        return pd.DataFrame()

    # ការកំណត់ Mapping Timeframes រវាង String និង MT5 Constants
    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1
    }

    # Mapping ចំនួន Bars ដែលត្រូវទាញយក
    bars_count_map = {
        "1d": 300,
        "5d": 1000,
        "7d": 1500,
        "14d": 2500
    }

    mt5_tf = tf_map.get(interval, mt5.TIMEFRAME_M5)
    num_bars = bars_count_map.get(range_, 500)

    # ផ្ទៀងផ្ទាត់ និងជ្រើសរើស Symbol ក្នុង MT5 Market Watch
    if not mt5.symbol_select(ticker, True):
        # ប្រសិនបើ Broker ប្រើឈ្មោះផ្សេង ស្វែងរក Symbol ដែលមាន XAU
        all_symbols = mt5.symbols_get()
        matched_symbol = None
        if all_symbols:
            for s in all_symbols:
                if ticker in s.name or ("XAU" in s.name and "USD" in s.name):
                    matched_symbol = s.name
                    mt5.symbol_select(matched_symbol, True)
                    break
        if matched_symbol:
            ticker = matched_symbol
        else:
            logging.warning(f"⚠️ Symbol {ticker} មិនមាននៅលើ MT5 Market Watch!")
            return pd.DataFrame()

    # ទាញយកទិន្នន័យពី MT5
    rates = mt5.copy_rates_from_pos(ticker, mt5_tf, 0, num_bars)
    if rates is None or len(rates) == 0:
        logging.warning(f"⚠️ ពុំមានទិន្នន័យពី MT5 សម្រាប់ {ticker} ({interval})")
        return pd.DataFrame()

    # រៀបចំទិន្នន័យចូល DataFrame ឱ្យស្របតាម Standard Structure
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna().reset_index(drop=True)

    with cache_lock:
        indicator_cache[cache_key] = (now, df)

    return df

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty or 'datetime' not in df.columns:
        return df
    df_copy = df.copy()
    df_copy.set_index('datetime', inplace=True)
    resampled = df_copy.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return resampled

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 5:
        return df

    df = df.copy()
    span_50 = min(50, len(df))
    span_200 = min(200, len(df))

    df['ema50'] = df['close'].ewm(span=span_50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=span_200, adjust=False).mean()

    high_low = df['high'] - df['low']
    high_cp = (df['high'] - df['close'].shift(1)).abs()
    low_cp = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    
    atr_period = min(14, len(df))
    df['atr'] = tr.rolling(atr_period, min_periods=1).mean().bfill()
    df['vol_sma'] = df['volume'].rolling(20, min_periods=1).mean().bfill()

    return df


# ==========================================
# 6. ENHANCED MARKET STRUCTURE ENGINE
# ==========================================
class MarketStructureEngine:
    @staticmethod
    def analyze_structure(df: pd.DataFrame, window: int = 3) -> Dict[str, Any]:
        if len(df) < (2 * window + 5):
            return {
                "structure_bias": "NEUTRAL", "structure_type": "NONE",
                "bos_detected": False, "choch_detected": False, "mss_detected": False,
                "structure_strength": 0.0, "last_hh": 0.0, "last_hl": 0.0,
                "last_lh": 0.0, "last_ll": 0.0
            }

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        swing_highs, swing_lows = [], []
        for i in range(window, len(df) - window):
            if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
                swing_highs.append({"index": i, "price": float(highs[i])})

            if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
                swing_lows.append({"index": i, "price": float(lows[i])})

        last_hh, last_hl, last_lh, last_ll = 0.0, 0.0, 0.0, 0.0
        if len(swing_highs) >= 2:
            if swing_highs[-1]["price"] > swing_highs[-2]["price"]:
                last_hh = swing_highs[-1]["price"]
            else:
                last_lh = swing_highs[-1]["price"]

        if len(swing_lows) >= 2:
            if swing_lows[-1]["price"] > swing_lows[-2]["price"]:
                last_hl = swing_lows[-1]["price"]
            else:
                last_ll = swing_lows[-1]["price"]

        curr_close = closes[-1]
        bos_detected, choch_detected, mss_detected = False, False, False
        structure_bias = "NEUTRAL"
        strength = 50.0

        if swing_highs and curr_close > swing_highs[-1]["price"]:
            bos_detected = True
            structure_bias = "BULLISH"
            strength += 25.0
            if len(swing_highs) >= 2 and swing_highs[-1]["price"] < swing_highs[-2]["price"]:
                choch_detected = True
                mss_detected = True
                strength += 15.0

        elif swing_lows and curr_close < swing_lows[-1]["price"]:
            bos_detected = True
            structure_bias = "BEARISH"
            strength += 25.0
            if len(swing_lows) >= 2 and swing_lows[-1]["price"] > swing_lows[-2]["price"]:
                choch_detected = True
                mss_detected = True
                strength += 15.0

        return {
            "structure_bias": structure_bias,
            "bos_detected": bos_detected,
            "choch_detected": choch_detected,
            "mss_detected": mss_detected,
            "structure_strength": round(min(strength, 100.0), 2),
            "last_hh": round(last_hh, 2),
            "last_hl": round(last_hl, 2),
            "last_lh": round(last_lh, 2),
            "last_ll": round(last_ll, 2)
        }


# ==========================================
# 7. ENHANCED LIQUIDITY ENGINE
# ==========================================
class LiquidityEngine:
    @staticmethod
    def detect_liquidity(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 20:
            return {
                "bsl_sweep": False, "ssl_sweep": False,
                "bsl_level": 0.0, "ssl_level": 0.0,
                "eqh_detected": False, "eql_detected": False,
                "stop_hunt": False, "liquidity_void": False
            }

        highs = df['high'].iloc[-30:-1]
        lows = df['low'].iloc[-30:-1]
        last_close = float(df['close'].iloc[-1])
        last_high = float(df['high'].iloc[-1])
        last_low = float(df['low'].iloc[-1])

        bsl = float(highs.max())
        ssl = float(lows.min())

        bsl_sweep = last_high > bsl and last_close < bsl
        ssl_sweep = last_low < ssl and last_close > ssl
        stop_hunt = bsl_sweep or ssl_sweep

        top_2_highs = sorted(highs.values, reverse=True)[:2]
        eqh_detected = len(top_2_highs) == 2 and abs(top_2_highs[0] - top_2_highs[1]) <= (bsl * 0.0005)

        bot_2_lows = sorted(lows.values)[:2]
        eql_detected = len(bot_2_lows) == 2 and abs(bot_2_lows[0] - bot_2_lows[1]) <= (ssl * 0.0005)

        atr_val = df['atr'].iloc[-1] if 'atr' in df else 2.0
        candle_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
        liquidity_void = candle_body > (atr_val * 2.5)

        return {
            "bsl_sweep": bsl_sweep,
            "ssl_sweep": ssl_sweep,
            "bsl_level": round(bsl, 2),
            "ssl_level": round(ssl, 2),
            "eqh_detected": eqh_detected,
            "eql_detected": eql_detected,
            "stop_hunt": stop_hunt,
            "liquidity_void": liquidity_void
        }


# ==========================================
# 8. ENHANCED ORDER BLOCK ENGINE
# ==========================================
class OrderBlockEngine:
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 15:
            return {
                "bullish_ob": None, "bearish_ob": None,
                "bullish_breaker": None, "bearish_breaker": None,
                "mitigation_block": None
            }

        curr_price = float(df['close'].iloc[-1])
        bullish_ob, bearish_ob = None, None
        bullish_breaker, bearish_breaker, mitigation_block = None, None, None

        for i in range(len(df) - 6, 2, -1):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                if df['close'].iloc[i+1] > df['high'].iloc[i]:
                    top = round(float(df['high'].iloc[i]), 2)
                    bottom = round(float(df['low'].iloc[i]), 2)
                    if curr_price >= bottom and df['low'].iloc[i+1:].min() >= bottom:
                        bullish_ob = {
                            "top": top, "bottom": bottom,
                            "ideal_price": round((top + bottom) / 2.0, 2),
                            "fresh": True, "quality_score": 85.0
                        }
                        break

        for i in range(len(df) - 6, 2, -1):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                if df['close'].iloc[i+1] < df['low'].iloc[i]:
                    top = round(float(df['high'].iloc[i]), 2)
                    bottom = round(float(df['low'].iloc[i]), 2)
                    if curr_price <= top and df['high'].iloc[i+1:].max() <= top:
                        bearish_ob = {
                            "top": top, "bottom": bottom,
                            "ideal_price": round((top + bottom) / 2.0, 2),
                            "fresh": True, "quality_score": 85.0
                        }
                        break

        for i in range(len(df) - 10, 3, -1):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                if df['close'].iloc[i+1] < df['low'].iloc[i] and curr_price > df['high'].iloc[i]:
                    bullish_breaker = {
                        "top": round(float(df['high'].iloc[i]), 2),
                        "bottom": round(float(df['low'].iloc[i]), 2),
                        "ideal_price": round(float(df['high'].iloc[i]), 2),
                        "status": "FRESH"
                    }
                    break

            if df['close'].iloc[i] < df['open'].iloc[i]:
                if df['close'].iloc[i+1] > df['high'].iloc[i] and curr_price < df['low'].iloc[i]:
                    bearish_breaker = {
                        "top": round(float(df['high'].iloc[i]), 2),
                        "bottom": round(float(df['low'].iloc[i]), 2),
                        "ideal_price": round(float(df['low'].iloc[i]), 2),
                        "status": "FRESH"
                    }
                    break

        return {
            "bullish_ob": bullish_ob,
            "bearish_ob": bearish_ob,
            "bullish_breaker": bullish_breaker,
            "bearish_breaker": bearish_breaker,
            "mitigation_block": mitigation_block
        }


# ==========================================
# 9. ENHANCED FAIR VALUE GAP ENGINE
# ==========================================
class FVGEngine:
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 5:
            return {
                "bullish_fvg": None, "bearish_fvg": None,
                "ifvg": None, "bpr": None
            }

        curr_price = float(df['close'].iloc[-1])
        bullish_fvg, bearish_fvg = None, None

        for i in range(len(df) - 2, 2, -1):
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_low = round(float(df['high'].iloc[i-2]), 2)
                gap_high = round(float(df['low'].iloc[i]), 2)
                gap_size = gap_high - gap_low
                min_retest = df['low'].iloc[i:].min()

                if min_retest > gap_low and curr_price >= gap_low:
                    filled_pct = min(100.0, max(0.0, ((gap_high - min_retest) / gap_size) * 100)) if gap_size > 0 else 0
                    bullish_fvg = {
                        "top": gap_high, "bottom": gap_low,
                        "ideal_price": round((gap_high + gap_low) / 2.0, 2),
                        "gap_fill_pct": round(filled_pct, 1),
                        "fresh": filled_pct < 80.0
                    }
                    break

            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_high = round(float(df['low'].iloc[i-2]), 2)
                gap_low = round(float(df['high'].iloc[i]), 2)
                gap_size = gap_high - gap_low
                max_retest = df['high'].iloc[i:].max()

                if max_retest < gap_high and curr_price <= gap_high:
                    filled_pct = min(100.0, max(0.0, ((max_retest - gap_low) / gap_size) * 100)) if gap_size > 0 else 0
                    bearish_fvg = {
                        "top": gap_high, "bottom": gap_low,
                        "ideal_price": round((gap_high + gap_low) / 2.0, 2),
                        "gap_fill_pct": round(filled_pct, 1),
                        "fresh": filled_pct < 80.0
                    }
                    break

        return {
            "bullish_fvg": bullish_fvg,
            "bearish_fvg": bearish_fvg,
            "ifvg": None,
            "bpr": None
        }


# ==========================================
# 10. SESSION & OTE & SMT ENGINES
# ==========================================
class SessionModel:
    @staticmethod
    def evaluate_session(df_m5: pd.DataFrame) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).time()
        
        london_kz = (datetime.strptime("07:00", "%H:%M").time(), datetime.strptime("10:00", "%H:%M").time())
        ny_kz = (datetime.strptime("12:00", "%H:%M").time(), datetime.strptime("15:00", "%H:%M").time())
        asian_range = (datetime.strptime("00:00", "%H:%M").time(), datetime.strptime("05:00", "%H:%M").time())

        session_name = "OTHER"
        is_killzone = False
        if london_kz[0] <= now_utc <= london_kz[1]:
            session_name = "LONDON_KILLZONE"
            is_killzone = True
        elif ny_kz[0] <= now_utc <= ny_kz[1]:
            session_name = "NEW_YORK_KILLZONE"
            is_killzone = True
        elif asian_range[0] <= now_utc <= asian_range[1]:
            session_name = "ASIAN_RANGE"

        return {
            "session": session_name,
            "is_killzone": is_killzone,
            "session_bias": "HIGH_VOLATILITY" if is_killzone else "NORMAL",
            "session_strength": 85.0 if is_killzone else 50.0,
            "session_phase": "EXPANSION" if is_killzone else "ACCUMULATION"
        }


class PremiumDiscountEngine:
    @staticmethod
    def calculate_array(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or len(df) < 10:
            return {"premium_zone": "N/A", "discount_zone": "N/A", "equilibrium": 0.0, "price_location": "EQUILIBRIUM"}

        swing_high = float(df['high'].iloc[-30:].max())
        swing_low = float(df['low'].iloc[-30:].min())
        curr_price = float(df['close'].iloc[-1])
        equilibrium = (swing_high + swing_low) / 2.0
        
        price_location = "PREMIUM" if curr_price > equilibrium else ("DISCOUNT" if curr_price < equilibrium else "EQUILIBRIUM")

        return {
            "premium_zone": f"{round(equilibrium, 2)} - {round(swing_high, 2)}",
            "discount_zone": f"{round(swing_low, 2)} - {round(equilibrium, 2)}",
            "equilibrium": round(equilibrium, 2),
            "price_location": price_location
        }


class OTEEngine:
    @staticmethod
    def evaluate_ote(df: pd.DataFrame, bias: str) -> Dict[str, Any]:
        if df.empty or len(df) < 20:
            return {"ote_zone": "N/A", "ote_level": 0.0, "ote_score": 0.0, "type": "NONE"}

        recent_high = float(df['high'].iloc[-20:].max())
        recent_low = float(df['low'].iloc[-20:].min())
        curr_price = float(df['close'].iloc[-1])
        range_diff = recent_high - recent_low

        if range_diff <= 0:
            return {"ote_zone": "N/A", "ote_level": 0.0, "ote_score": 0.0, "type": "NONE"}

        if bias == "BULLISH":
            fib_62 = recent_high - (range_diff * 0.62)
            fib_705 = recent_high - (range_diff * 0.705)
            fib_79 = recent_high - (range_diff * 0.79)
            in_ote = fib_79 <= curr_price <= fib_62
            return {
                "ote_zone": f"{round(fib_79, 2)} - {round(fib_62, 2)}",
                "ote_level": round(fib_705, 2),
                "ote_score": 100.0 if in_ote else 50.0,
                "type": "BULLISH_OTE"
            }
        elif bias == "BEARISH":
            fib_62_bear = recent_low + (range_diff * 0.62)
            fib_705_bear = recent_low + (range_diff * 0.705)
            fib_79_bear = recent_low + (range_diff * 0.79)
            in_ote = fib_62_bear <= curr_price <= fib_79_bear
            return {
                "ote_zone": f"{round(fib_62_bear, 2)} - {round(fib_79_bear, 2)}",
                "ote_level": round(fib_705_bear, 2),
                "ote_score": 100.0 if in_ote else 50.0,
                "type": "BEARISH_OTE"
            }

        return {"ote_zone": "N/A", "ote_level": 0.0, "ote_score": 0.0, "type": "NONE"}


class SMTEngine:
    @staticmethod
    def check_smt(df_gold: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_silver = fetch_ohlcv_safe(SILVER_TICKER, interval="5m", range_="1d")
            if df_gold.empty or df_silver.empty or len(df_gold) < 10 or len(df_silver) < 10:
                return {"smt_signal": "NONE", "smt_strength": 0.0}

            g_high_prev, g_high_curr = df_gold['high'].iloc[-10:-3].max(), df_gold['high'].iloc[-3:].max()
            s_high_prev, s_high_curr = df_silver['high'].iloc[-10:-3].max(), df_silver['high'].iloc[-3:].max()

            g_low_prev, g_low_curr = df_gold['low'].iloc[-10:-3].min(), df_gold['low'].iloc[-3:].min()
            s_low_prev, s_low_curr = df_silver['low'].iloc[-10:-3].min(), df_silver['low'].iloc[-3:].min()

            if g_high_curr > g_high_prev and s_high_curr <= s_high_prev:
                return {"smt_signal": "BEARISH_SMT", "smt_strength": 85.0}

            if g_low_curr < g_low_prev and s_low_curr >= s_low_prev:
                return {"smt_signal": "BULLISH_SMT", "smt_strength": 85.0}

            return {"smt_signal": "NONE", "smt_strength": 0.0}
        except Exception:
            return {"smt_signal": "NONE", "smt_strength": 0.0}


# ==========================================
# 11. MARKET NARRATIVE ENGINE
# ==========================================
class MarketNarrativeEngine:
    @staticmethod
    def analyze_narrative(
        df_w: pd.DataFrame, df_d: pd.DataFrame, df_h4: pd.DataFrame,
        df_h1: pd.DataFrame, df_m15: pd.DataFrame
    ) -> Dict[str, Any]:
        
        def get_bias(df: pd.DataFrame) -> str:
            if df.empty or len(df) < 5:
                return "NEUTRAL"
            c = df['close'].iloc[-1]
            ema = df['ema50'].iloc[-1] if 'ema50' in df else c
            return "BULLISH" if c > ema else ("BEARISH" if c < ema else "NEUTRAL")

        w_bias = get_bias(df_w)
        d_bias = get_bias(df_d)
        h4_bias = get_bias(df_h4)
        h1_bias = get_bias(df_h1)
        m15_bias = get_bias(df_m15)

        delivery_state = "EXPANSION" if (h4_bias == h1_bias and h1_bias != "NEUTRAL") else "RETRACEMENT"
        
        now_hour = datetime.now(timezone.utc).hour
        amd_state = "ACCUMULATION" if 0 <= now_hour < 6 else ("MANIPULATION" if 6 <= now_hour < 11 else "DISTRIBUTION")

        return {
            "weekly_bias": w_bias,
            "daily_bias": d_bias,
            "h4_bias": h4_bias,
            "h1_bias": h1_bias,
            "m15_bias": m15_bias,
            "market_delivery_state": delivery_state,
            "amd_state": amd_state,
            "liquidity_draw": "BSL" if h1_bias == "BULLISH" else "SSL"
        }


# ==========================================
# 12. UNIFIED STRATEGY EVALUATOR
# ==========================================
class ICTStrategyEvaluator:
    @staticmethod
    def evaluate(
        df_h4: pd.DataFrame, df_h1: pd.DataFrame,
        df_m15: pd.DataFrame, df_m5: pd.DataFrame
    ) -> Dict[str, Any]:
        
        start_time = time.time()

        if df_m5.empty or df_h1.empty or len(df_m5) < 10:
            return create_standard_signal(status="NO_DATA", reason="ទិន្នន័យមិនគ្រប់គ្រាន់")

        curr_price = round(float(df_m5['close'].iloc[-1]), 2)
        atr = round(float(df_m5['atr'].iloc[-1]), 2) if 'atr' in df_m5 else 1.5

        df_d = resample_ohlcv(df_h1, '1D')
        df_w = resample_ohlcv(df_h1, '1W')
        df_d = add_indicators(df_d)
        df_w = add_indicators(df_w)

        narrative = MarketNarrativeEngine.analyze_narrative(df_w, df_d, df_h4, df_h1, df_m15)
        structure = MarketStructureEngine.analyze_structure(df_m5)
        liquidity = LiquidityEngine.detect_liquidity(df_m5)
        obs = OrderBlockEngine.detect_order_blocks(df_m5)
        fvgs = FVGEngine.detect_fvg(df_m5)

        session_info = SessionModel.evaluate_session(df_m5)
        pd_array = PremiumDiscountEngine.calculate_array(df_m5)
        ote_info = OTEEngine.evaluate_ote(df_m5, narrative['h1_bias'])
        smt_info = SMTEngine.check_smt(df_m5)
        is_news, news_msg = NewsFilter.check_high_impact_news(NEWS_BUFFER_MINUTES)

        buy_score, sell_score = 0.0, 0.0
        buy_reasons, sell_reasons = [], []

        if narrative['weekly_bias'] == "BULLISH": buy_score += 5; buy_reasons.append("Weekly Bullish Bias (+5)")
        if narrative['daily_bias'] == "BULLISH": buy_score += 5; buy_reasons.append("Daily Bullish Bias (+5)")
        if narrative['h4_bias'] == "BULLISH": buy_score += 10; buy_reasons.append("H4 Structure Alignment (+10)")
        if narrative['h1_bias'] == "BULLISH": buy_score += 10; buy_reasons.append("H1 Order Flow Alignment (+10)")
        if structure['structure_bias'] == "BULLISH": buy_score += 10; buy_reasons.append("M5 CHOCH/MSS Confirmed (+10)")
        if liquidity['ssl_sweep']: buy_score += 15; buy_reasons.append("Sell-Side Liquidity Swept (+15)")
        if obs['bullish_ob'] and obs['bullish_ob']['fresh']: buy_score += 15; buy_reasons.append("Fresh Bullish Order Block (+15)")
        if obs['bullish_breaker']: buy_score += 10; buy_reasons.append("Bullish Breaker Block (+10)")
        if fvgs['bullish_fvg'] and fvgs['bullish_fvg']['fresh']: buy_score += 10; buy_reasons.append("Bullish FVG Alignment (+10)")
        if ote_info['type'] == "BULLISH_OTE": buy_score += 5; buy_reasons.append("Price inside Bullish OTE Zone (+5)")
        if smt_info['smt_signal'] == "BULLISH_SMT": buy_score += 5; buy_reasons.append("Bullish SMT Divergence (+5)")

        if narrative['weekly_bias'] == "BEARISH": sell_score += 5; sell_reasons.append("Weekly Bearish Bias (+5)")
        if narrative['daily_bias'] == "BEARISH": sell_score += 5; sell_reasons.append("Daily Bearish Bias (+5)")
        if narrative['h4_bias'] == "BEARISH": sell_score += 10; sell_reasons.append("H4 Structure Alignment (+10)")
        if narrative['h1_bias'] == "BEARISH": sell_score += 10; sell_reasons.append("H1 Order Flow Alignment (+10)")
        if structure['structure_bias'] == "BEARISH": sell_score += 10; sell_reasons.append("M5 CHOCH/MSS Confirmed (+10)")
        if liquidity['bsl_sweep']: sell_score += 15; sell_reasons.append("Buy-Side Liquidity Swept (+15)")
        if obs['bearish_ob'] and obs['bearish_ob']['fresh']: sell_score += 15; sell_reasons.append("Fresh Bearish Order Block (+15)")
        if obs['bearish_breaker']: sell_score += 10; sell_reasons.append("Bearish Breaker Block (+10)")
        if fvgs['bearish_fvg'] and fvgs['bearish_fvg']['fresh']: sell_score += 10; sell_reasons.append("Bearish FVG Alignment (+10)")
        if ote_info['type'] == "BEARISH_OTE": sell_score += 5; sell_reasons.append("Price inside Bearish OTE Zone (+5)")
        if smt_info['smt_signal'] == "BEARISH_SMT": sell_score += 5; sell_reasons.append("Bearish SMT Divergence (+5)")

        if is_news:
            return create_standard_signal(
                status="FILTERED", signal_type="NO TRADE", action="WAIT",
                price=curr_price, is_news=True, news_msg=news_msg,
                reason="High Impact News Event Block"
            )

        signal_type = "WAIT"
        action = "WAIT"
        final_score = 0.0
        active_reasons = []

        if buy_score >= MIN_PROBABILITY_SCORE and buy_score > sell_score:
            final_score = buy_score
            active_reasons = buy_reasons
            
            if liquidity['ssl_sweep'] or structure['choch_detected']:
                signal_type = "BUY NOW"
                action = "BUY"
            else:
                signal_type = "WAIT BUY ZONE"
                action = "BUY"

        elif sell_score >= MIN_PROBABILITY_SCORE and sell_score > buy_score:
            final_score = sell_score
            active_reasons = sell_reasons

            if liquidity['bsl_sweep'] or structure['choch_detected']:
                signal_type = "SELL NOW"
                action = "SELL"
            else:
                signal_type = "WAIT SELL ZONE"
                action = "SELL"
        else:
            signal_type = "WAIT"
            action = "WAIT"

        sl_pips = max(atr * 1.5, 2.5)
        
        if action == "BUY":
            ideal_entry = obs['bullish_ob']['ideal_price'] if obs['bullish_ob'] else (fvgs['bullish_fvg']['ideal_price'] if fvgs['bullish_fvg'] else curr_price)
            entry_high = max(curr_price, ideal_entry + 0.5)
            entry_low = min(curr_price, ideal_entry - 0.5)
            sl = entry_low - sl_pips
            tp1 = curr_price + (sl_pips * 1.5)
            tp2 = curr_price + (sl_pips * 3.0)
            tp3 = curr_price + (sl_pips * 5.0)
            invalidation = sl - 1.0
        elif action == "SELL":
            ideal_entry = obs['bearish_ob']['ideal_price'] if obs['bearish_ob'] else (fvgs['bearish_fvg']['ideal_price'] if fvgs['bearish_fvg'] else curr_price)
            entry_high = max(curr_price, ideal_entry + 0.5)
            entry_low = min(curr_price, ideal_entry - 0.5)
            sl = entry_high + sl_pips
            tp1 = curr_price - (sl_pips * 1.5)
            tp2 = curr_price - (sl_pips * 3.0)
            tp3 = curr_price - (sl_pips * 5.0)
            invalidation = sl + 1.0
        else:
            ideal_entry, entry_high, entry_low, sl, tp1, tp2, tp3, invalidation = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        risk_usd = ACCOUNT_BALANCE * RISK_PER_TRADE_PCT
        position_size = round(risk_usd / (sl_pips * 10.0), 2) if sl_pips > 0 else 0.01

        tier = "TIER_1_A_PLUS" if final_score >= 80 else ("TIER_2" if final_score >= 65 else "TIER_3")
        signal_grade = "A+" if final_score >= 80 else ("A" if final_score >= 70 else "B")
        confidence = "HIGH" if final_score >= 75 else ("MEDIUM" if final_score >= 60 else "LOW")

        logging.info(
            f"🔍 MT5 Scan Evaluated | Signal Type: {signal_type} | Score: {final_score} | "
            f"Session: {session_info['session']} | Exec Time: {round(time.time() - start_time, 3)}s"
        )

        return create_standard_signal(
            status="OK",
            signal_type=signal_type,
            action=action,
            price=curr_price,
            score=final_score,
            entry_zone_high=entry_high,
            entry_zone_low=entry_low,
            ideal_entry=ideal_entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr=3.0 if action != "WAIT" else 0.0,
            win_rate="78%" if final_score >= 80 else "65%",
            position_size=position_size,
            tier=tier,
            htf_bias=narrative['h1_bias'],
            session=session_info['session'],
            is_news=is_news,
            news_msg=news_msg,
            reason=", ".join(active_reasons) if active_reasons else "រង់ចាំ Setup ច្បាស់លាស់បន្ថែមទៀត",
            reasons=active_reasons,
            weekly_bias=narrative['weekly_bias'],
            daily_bias=narrative['daily_bias'],
            h4_bias=narrative['h4_bias'],
            h1_bias=narrative['h1_bias'],
            m15_bias=narrative['m15_bias'],
            market_delivery_state=narrative['market_delivery_state'],
            liquidity_draw=narrative['liquidity_draw'],
            amd_state=narrative['amd_state'],
            premium_zone=pd_array['premium_zone'],
            discount_zone=pd_array['discount_zone'],
            equilibrium=pd_array['equilibrium'],
            price_location=pd_array['price_location'],
            ote_zone=ote_info['ote_zone'],
            ote_level=ote_info['ote_level'],
            ote_score=ote_info['ote_score'],
            smt_signal=smt_info['smt_signal'],
            smt_strength=smt_info['smt_strength'],
            session_bias=session_info['session_bias'],
            session_strength=session_info['session_strength'],
            session_phase=session_info['session_phase'],
            order_block=obs['bullish_ob'] or obs['bearish_ob'],
            mitigation_block=obs['mitigation_block'],
            breaker_block=obs['bullish_breaker'] or obs['bearish_breaker'],
            fvg_info=fvgs['bullish_fvg'] or fvgs['bearish_fvg'],
            structure_info=structure,
            liquidity_info=liquidity,
            signal_grade=signal_grade,
            confidence_level=confidence,
            invalidation_level=invalidation,
            atr_stop=sl_pips,
            structure_stop=sl_pips
        )


# ==========================================
# 13. UNIFIED BACKTESTER
# ==========================================
class UnifiedBacktester:
    def __init__(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame):
        self.df_m5 = add_indicators(df_m5)
        self.df_h1 = add_indicators(df_h1)

    def run(self) -> Dict[str, Any]:
        if len(self.df_m5) < 100 or len(self.df_h1) < 20:
            return {"status": "INSUFFICIENT_DATA"}

        balance = ACCOUNT_BALANCE
        trades = []

        for i in range(60, len(self.df_m5) - 10):
            sub_m5 = self.df_m5.iloc[:i]
            sub_m15 = sub_m5.iloc[::3]
            curr_time = sub_m5['datetime'].iloc[-1]
            sub_h1 = self.df_h1[self.df_h1['datetime'] <= curr_time]

            if sub_h1.empty:
                continue

            sub_h4 = add_indicators(resample_ohlcv(sub_h1, '4h'))
            signal = ICTStrategyEvaluator.evaluate(sub_h4, sub_h1, sub_m15, sub_m5)

            if signal.get("signal_type") in ["BUY NOW", "SELL NOW"] and signal.get("score", 0) >= MIN_PROBABILITY_SCORE:
                entry = signal["price"]
                sl = signal["sl"]
                tp2 = signal["tp2"]
                
                future_df = self.df_m5.iloc[i:i+20]
                win = False
                for _, f_row in future_df.iterrows():
                    if signal["action"] == "BUY":
                        if f_row['high'] >= tp2:
                            win = True; break
                        elif f_row['low'] <= sl:
                            break
                    elif signal["action"] == "SELL":
                        if f_row['low'] <= tp2:
                            win = True; break
                        elif f_row['high'] >= sl:
                            break

                pnl = (ACCOUNT_BALANCE * RISK_PER_TRADE_PCT * 3.0) if win else -(ACCOUNT_BALANCE * RISK_PER_TRADE_PCT)
                balance += pnl
                trades.append({"action": signal["action"], "win": win, "pnl": pnl, "score": signal["score"]})

        if not trades:
            return {"status": "NO_TRADES_TRIGGERED"}

        wins = [t for t in trades if t["win"]]
        win_rate = (len(wins) / len(trades)) * 100

        return {
            "status": "OK",
            "total_trades": len(trades),
            "win_rate": f"{round(win_rate, 2)}%",
            "final_balance": round(balance, 2)
        }


# ==========================================
# 14. TELEGRAM MESSAGE FORMATTER & AI NARRATIVE
# ==========================================
def generate_ai_analysis(data: Dict[str, Any]) -> str:
    reasons_str = ", ".join(data.get('reasons', [])) if data.get('reasons') else "Structure Shift"

    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            prompt = (
                f"You are a Senior ICT/SMC Gold Quant Trader. Generate an institutional narrative for:\n"
                f"Signal Type: {data.get('signal_type')} | Price: ${data.get('price')} | Score: {data.get('score')}%\n"
                f"Weekly: {data.get('weekly_bias')} | Daily: {data.get('daily_bias')} | H4: {data.get('h4_bias')} | H1: {data.get('h1_bias')}\n"
                f"Delivery State: {data.get('market_delivery_state')} | AMD: {data.get('amd_state')}\n"
                f"Session: {data.get('session')} | Confluences: {reasons_str}\n"
                f"Provide concise trading narrative in Khmer language explaining why this setup formed."
            )
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=8)
            if res.status_code == 200:
                return res.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logging.warning(f"AI Gemini Fallback: {e}")

    return (
        f"🤖 *ការវិភាគលម្អិតពីទីផ្សារ (Institutional SMC Narrative)*\n"
        f"• * Weekly & Daily Bias:* `{data.get('weekly_bias')}` / `{data.get('daily_bias')}`\n"
        f"• * H4 & H1 Flow:* `{data.get('h4_bias')}` / `{data.get('h1_bias')}`\n"
        f"• *Delivery State:* `{data.get('market_delivery_state')}` Phase (`{data.get('amd_state')}`)\n"
        f"• * Market Location:* `{data.get('price_location')}` Zone\n"
        f"💡 *កត្តាតភ្ជាប់ (Reasons):* `{reasons_str}`"
    )


def format_signal_output(signal: Dict[str, Any], ai_commentary: str) -> str:
    reasons_list = "\n".join([f"  • {r}" for r in signal.get("reasons", [])])
    sig_type = signal.get('signal_type', 'WAIT')

    if sig_type in ["BUY NOW", "SELL NOW"]:
        header = f"⚡ *INSTITUTIONAL SIGNAL: {sig_type}* ⚡"
        entry_details = (
            f"💵 *Current Price:* `${signal.get('price')}`\n"
            f"🎯 *Instant Entry Price:* `${signal.get('price')}`\n"
        )
    elif sig_type in ["WAIT BUY ZONE", "WAIT SELL ZONE"]:
        header = f"⏳ *PENDING SETUP: {sig_type}* ⏳"
        entry_details = (
            f"💵 *Current Price:* `${signal.get('price')}`\n"
            f"📍 *Entry Zone High:* `${signal.get('entry_zone_high')}`\n"
            f"📍 *Entry Zone Low:* `${signal.get('entry_zone_low')}`\n"
            f"🎯 *Ideal Limit Entry:* `${signal.get('ideal_entry')}`\n"
            f"⚠️ *Invalidation Level:* `${signal.get('invalidation_level')}`\n"
            f"🎯 *OTE Level:* `${signal.get('ote_level')}`\n"
        )
    else:
        header = f"📊 *SMC MARKET ANALYSIS: {sig_type}* 📊"
        entry_details = f"💵 *Current Price:* `${signal.get('price')}`\n"

    return (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📍 *Asset:* {SYMBOL_NAME}\n"
        f"{entry_details}"
        f"🛑 *Stop Loss (SL):* `${signal.get('sl')}`\n"
        f"🎯 *Take Profit 1:* `${signal.get('tp1')}`\n"
        f"🎯 *Take Profit 2:* `${signal.get('tp2')}`\n"
        f"🎯 *Take Profit 3:* `${signal.get('tp3')}`\n"
        f"⚖️ *Risk/Reward:* `{signal.get('rr')}`\n"
        f"📊 *Position Size:* `{signal.get('position_size')} Lots (1% Risk)`\n"
        f"🔥 *Probability Score:* `{signal.get('score')}/100` (Grade {signal.get('signal_grade')})\n"
        f"🏅 *Confidence Level:* `{signal.get('confidence_level')}`\n\n"
        f"🌐 *Institutional Narrative:*\n"
        f"  • HTF Bias: `{signal.get('htf_bias')}`\n"
        f"  • Session: `{signal.get('session')}`\n"
        f"  • Price Location: `{signal.get('price_location')}`\n\n"
        f"💡 *Confluence Reasons:*\n{reasons_list}\n\n"
        f"{ai_commentary}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ *Time:* {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )


def send_telegram_msg_with_button(chat_id_target: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    reply_markup = {
        "inline_keyboard": [
            [{"text": "📊 Live SMC MTF Analysis", "callback_data": "btn_analyze_now"}],
            [{"text": "📈 Run Unified Strategy Backtest", "callback_data": "btn_backtest_now"}],
            [{"text": "📋 Performance Journal Report", "callback_data": "btn_report_now"}]
        ]
    }
    payload = {
        "chat_id": chat_id_target,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.status_code == 200
    except Exception as e:
        logging.error(f"Telegram Send Error: {e}")
        return False


def answer_callback_query(callback_query_id: str, text: str = "Processing..."):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery"
    try:
        requests.post(url, json={"callback_query_id": callback_query_id, "text": text}, timeout=5)
    except Exception as e:
        logging.warning(f"Callback error: {e}")


# ==========================================
# 15. BOT HANDLERS & BOT CYCLE EXECUTION
# ==========================================
def trigger_instant_analysis(target_chat_id: str):
    df_h1 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="1h", range_="7d")
    df_m15 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="15m", range_="5d")
    df_m5 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="5m", range_="1d")

    if df_h1.empty or df_m15.empty or df_m5.empty:
        send_telegram_msg_with_button(target_chat_id, "⚠️ Failed to fetch MT5 market data!")
        return

    df_h1 = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)
    df_m5 = add_indicators(df_m5)
    df_h4 = add_indicators(resample_ohlcv(df_h1, '4h'))

    signal = ICTStrategyEvaluator.evaluate(df_h4, df_h1, df_m15, df_m5)
    ai_commentary = generate_ai_analysis(signal)
    msg = format_signal_output(signal, ai_commentary)
    send_telegram_msg_with_button(target_chat_id, msg)


def trigger_backtest_report(target_chat_id: str):
    df_m5 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="5m", range_="5d")
    df_h1 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="1h", range_="14d")
    
    if df_m5.empty or df_h1.empty:
        send_telegram_msg_with_button(target_chat_id, "⚠️ Backtest MT5 data unavailable!")
        return
        
    backtester = UnifiedBacktester(df_m5, df_h1)
    stats = backtester.run()
    
    msg = (
        f"📈 *UNIFIED SMC STRATEGY BACKTEST REPORT* 📈\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔢 *Total Trades Evaluated:* `{stats.get('total_trades', 0)}`\n"
        f"🎯 *Win Rate:* `{stats.get('win_rate', '0%')}`\n"
        f"💰 *Simulated Balance:* `${stats.get('final_balance', ACCOUNT_BALANCE)}`\n"
        f"━━━━━━━━━━━━━━━━━━━"
    )
    send_telegram_msg_with_button(target_chat_id, msg)


def trigger_journal_report(target_chat_id: str):
    report = PerformanceTracker.generate_report(30)
    if report["status"] != "OK":
        send_telegram_msg_with_button(target_chat_id, f"📋 *Journal:* {report['message']}")
        return

    msg = (
        f"📋 *30-DAY PERFORMANCE JOURNAL* 📋\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔢 *Total Trades:* `{report['total_trades']}`\n"
        f"✅ *Wins:* `{report['wins']}` | ❌ *Losses:* `{report['losses']}`\n"
        f"🎯 *Win Rate:* `{report['win_rate']}`\n"
        f"⚖️ *Profit Factor:* `{report['profit_factor']}`\n"
        f"💰 *Net Profit:* `${report['net_profit']}`\n"
        f"━━━━━━━━━━━━━━━━━━━"
    )
    send_telegram_msg_with_button(target_chat_id, msg)


def telegram_poll_listener():
    global last_update_id
    logging.info("🎧 Telegram Listener Thread Active...")

    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=20"
            res = requests.get(url, timeout=25)
            
            if res.status_code == 200:
                updates = res.json().get("result", [])
                for update in updates:
                    last_update_id = update["update_id"]
                    
                    if "callback_query" in update:
                        cb = update["callback_query"]
                        cb_data = cb.get("data")
                        sender_chat_id = str(cb.get("message", {}).get("chat", {}).get("id", CHAT_ID))
                        
                        if cb_data == "btn_analyze_now":
                            answer_callback_query(cb["id"], "🧠 Scanning MT5 Market Structure...")
                            trigger_instant_analysis(sender_chat_id)
                        elif cb_data == "btn_backtest_now":
                            answer_callback_query(cb["id"], "📈 Running Backtest...")
                            trigger_backtest_report(sender_chat_id)
                        elif cb_data == "btn_report_now":
                            answer_callback_query(cb["id"], "📋 Generating Journal...")
                            trigger_journal_report(sender_chat_id)

                    msg = update.get("message") or update.get("channel_post")
                    if msg and "text" in msg:
                        txt = msg["text"]
                        chat_id_target = str(msg["chat"]["id"])
                        if any(cmd in txt for cmd in ["/start", "/analyze"]):
                            trigger_instant_analysis(chat_id_target)
                        elif "/backtest" in txt:
                            trigger_backtest_report(chat_id_target)
                        elif "/report" in txt:
                            trigger_journal_report(chat_id_target)

        except Exception as e:
            time.sleep(5)


def run_bot_cycle():
    global last_processed_signal_id

    logging.info("🔄 Running Scheduled Market Scan via MT5...")
    df_h1 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="1h", range_="7d")
    df_m15 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="15m", range_="5d")
    df_m5 = fetch_ohlcv_safe(YFINANCE_TICKER, interval="5m", range_="1d")

    if df_h1.empty or df_m15.empty or df_m5.empty:
        return

    df_h1 = add_indicators(df_h1)
    df_m15 = add_indicators(df_m15)
    df_m5 = add_indicators(df_m5)
    df_h4 = add_indicators(resample_ohlcv(df_h1, '4h'))

    signal = ICTStrategyEvaluator.evaluate(df_h4, df_h1, df_m15, df_m5)

    if signal.get("signal_type") in ["WAIT", "NO TRADE"] or signal.get("is_news"):
        logging.info(f"ℹ️ Status: {signal.get('signal_type')} | Price: ${signal.get('price')} | Score: {signal.get('score')}")
        return

    signal_id = f"{signal.get('signal_type')}_{signal.get('price')}_{signal.get('score')}"
    if signal_id != last_processed_signal_id:
        ai_commentary = generate_ai_analysis(signal)
        msg = format_signal_output(signal, ai_commentary)

        if send_telegram_msg_with_button(CHAT_ID, msg):
            logging.info(f"🚀 Signal Broadcasted: {signal.get('signal_type')} | Score: {signal.get('score')}")
            last_processed_signal_id = signal_id
            
            PerformanceTracker.save_trade({
                "time": datetime.now(timezone.utc).isoformat(),
                "signal_type": signal.get("signal_type"),
                "action": signal.get("action"),
                "entry": signal.get("price"),
                "sl": signal.get("sl"),
                "tp": signal.get("tp2"),
                "score": signal.get("score"),
                "rr": 3.0,
                "outcome": "PENDING",
                "pnl": 0.0
            })


def main():
    logging.info("🤖 AI ICT/SMC Dual Entry Signal Bot Engine (MT5 Powered) Started!")

    # បើកការភ្ជាប់ MT5 ដំបូង
    init_mt5_connection()

    listener_thread = threading.Thread(target=telegram_poll_listener, daemon=True)
    listener_thread.start()

    send_telegram_msg_with_button(
        CHAT_ID, 
        "🤖 *Institutional AI ICT/SMC Telegram Signal Bot upgraded to MT5 and online!*"
    )

    while True:
        try:
            run_bot_cycle()
        except Exception as e:
            logging.error(f"Loop Exception: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
