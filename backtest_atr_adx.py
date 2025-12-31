# Filename: backtest_retrain_top_bidask.py
"""
Retrain top-N configs with bid/ask-aware SL/TP using existing summary.csv.

- Reads original TRAIN summary from backtest_results/summary.csv
- Selects top N configs by existing 'rank'
- Retrains those configs on TRAIN with corrected SL/TP execution
- Then re-tests them on TEST
- Writes new results to backtest_results_bidask_top/

Bid/Ask modelling:
- Candles are assumed to be MID prices (common in many providers).
- We approximate bid/ask using a configurable spread:
    * BID ≈ MID - spread/2
    * ASK ≈ MID + spread/2
- IMPORTANT (Capital.com / CFD-style sizing): we model spread via entry/exit prices,
  so we DO NOT subtract an additional round-trip spread cost from P&L (to avoid double-counting).
- P&L uses a configurable CONTRACT_MULTIPLIER and UNIT_MODE to handle non-lot sizing.

ENV additions:
  - UNIT_MODE: "UNITS" | "LOTS" | "CONTRACTS" (default: "CONTRACTS")
  - LOT_SIZE: units per 1 lot for FX (default: 100000)
  - CONTRACT_MULTIPLIER: multiplier applied to (price change * position_units) (default: 1.0)
"""

import os
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import math
import itertools

# Parallelism
import multiprocessing
from joblib import Parallel, delayed

# =========================
# Config & Globals
# =========================

load_dotenv()

TOP_N_RETRAIN = int(os.getenv("TOP_N_RETRAIN", "50"))
TOP_N_TEST = int(os.getenv("TOP_N_TEST", "20"))
N_JOBS = int(os.getenv("N_JOBS", multiprocessing.cpu_count()))

ENV = os.getenv("OANDA_ENV", "practice")  # "practice" | "live"
TOKEN = os.getenv("OANDA_TOKEN", "")
REVERSE_SIGNALS = os.getenv("REVERSE_SIGNALS", "0").lower() in ("1", "true", "yes", "y")

DEFAULT_INSTRUMENT = os.getenv("INSTRUMENT", "EUR_USD")
DEFAULT_GRAN = os.getenv("GRANULARITY", "M5")

# Strategy feature flags (base)
USE_VOLUME_CONFIRM_BASE = os.getenv("USE_VOLUME_CONFIRM", "1").lower() in ("1", "true", "yes")
USE_TIME_FILTER_BASE = os.getenv("USE_TIME_FILTER", "1").lower() in ("1", "true", "yes")
USE_MULTI_TF_BASE = os.getenv("USE_MULTI_TF", "0").lower() in ("1", "true", "yes")
USE_SWING_DETECT_BASE = os.getenv("USE_SWING_DETECT", "1").lower() in ("1", "true", "yes")

# DB writing (optional)
WRITE_TO_DB = os.getenv("WRITE_TO_DB", "0").lower() in ("1", "true", "yes")

BASE = (
    "https://api-fxpractice.oanda.com"
    if ENV == "practice"
    else "https://api-fxtrade.oanda.com"
)

DB_PATH = os.getenv("BACKTEST_DB_PATH", "backtest_trades.db")

START_EQUITY = float(os.getenv("START_EQUITY", "10000"))
DAYS_BACK = int(os.getenv("DAYS_BACK", "1095"))

# Approx spread in pips (round-trip) for EURUSD on M5
SPREAD_PIPS = float(os.getenv("SPREAD_PIPS", "1.2"))  # set to Capital-like spread for transferability
# Backtest sizing mode:
#   - OANDA: broker-neutral USD-risk sizing using spot-FX "units" semantics (recommended when training on OANDA candles)
#   - CAPITAL: CFD-style sizing adapter (contracts/lots) using CONTRACT_MULTIPLIER, LOT_SIZE, MIN_SIZE, SIZE_STEP
BACKTEST_BROKER = os.getenv("BACKTEST_BROKER", "OANDA").strip().upper()

# --- Position sizing model (Capital.com / CFDs) ---
# Many CFD brokers (incl. Capital.com) do not use "lots" the same way as spot FX.
# We support:
#   - CONTRACTS/UNITS: position size is the raw "units" used in P&L (default)
#   - LOTS: size specified in lots; converted to units via LOT_SIZE (FX default 100k)
if BACKTEST_BROKER == "CAPITAL":
    # Many CFD brokers (incl. Capital.com) do not use "lots" the same way as spot FX.
    # We support:
    #   - CONTRACTS/UNITS: position size is the raw "units" used in P&L (default)
    #   - LOTS: size specified in lots; converted to units via LOT_SIZE (FX default 100k)
    UNIT_MODE = os.getenv("UNIT_MODE", "CONTRACTS").strip().upper()  # "UNITS"|"LOTS"|"CONTRACTS"
    LOT_SIZE = float(os.getenv("LOT_SIZE", "100000"))  # FX convention if you still want lots
    CONTRACT_MULTIPLIER = float(os.getenv("CONTRACT_MULTIPLIER", "1.0"))
else:
    # OANDA-style backtest: treat position size as spot-FX "units" so USD-risk sizing is stable across brokers.
    UNIT_MODE = "UNITS"
    LOT_SIZE = 100000.0
    CONTRACT_MULTIPLIER = 1.0


# Train/test split fraction (e.g. 0.7 = first 70% train, last 30% test)
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.7"))

# Minimum trades filter when ranking configs
MIN_TRADES = int(os.getenv("MIN_TRADES", "200"))

# Number of top configs (by score/ rank) to re-train
TOP_N_RETRAIN = int(os.getenv("TOP_N_RETRAIN", "50"))

# Number of top configs (by score) to re-test on TEST data
TOP_N_TEST = int(os.getenv("TOP_N_TEST", "20"))

# Optional: override n_jobs via ENV, else use cores-2
N_JOBS_ENV = os.getenv("N_JOBS", "").strip()

session = requests.Session()
if not TOKEN:
    raise SystemExit("Set OANDA_TOKEN in .env")
session.headers.update({"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})

CACHE_DIR = "cache_hist"
os.makedirs(CACHE_DIR, exist_ok=True)

# NEW results dir for bid/ask retrain of top configs
RESULTS_DIR = "v2_backtest_atr_adx"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_RAW_PATH = os.path.join(RESULTS_DIR, "summary_raw.csv")
SUMMARY_RANKED_PATH = os.path.join(RESULTS_DIR, "summary.csv")

# Chunk size for parallel retrain
CHUNK_SIZE = 10

# Hard floor + ceiling for SL in pips (for safety)
MIN_SL_PIPS = float(os.getenv("MIN_SL_PIPS", "4.0"))
MAX_SL_PIPS = float(os.getenv("MAX_SL_PIPS", "60.0"))

STRATEGY_MODES = ["FULL", "RANGE_ONLY","TREND_ONLY",  "NO_MTF"]

# Risk per trade (leave full range, you can narrow later if needed)
RISK_PCT_RANGE = [0.01, 0.02, 0.005, 0.015]

# ATR-based SL multipliers (core + fractional + extended for strong trends)
SL_ATR_MULTIPLIER_RANGE = [
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
]

# RR: TP = SL * RR
RR_RANGE = [1.5, 1.75, 2.0, 2.5, 3.0]

# Max losing trades per day
DAILY_LOSSES_RANGE = [1, 2, 3]

PARAM_CONFIGS = list(
    itertools.product(
        STRATEGY_MODES,
        RISK_PCT_RANGE,
        SL_ATR_MULTIPLIER_RANGE,
        RR_RANGE,
        DAILY_LOSSES_RANGE,
    )
)
print(f"[init] Brute force will test {len(PARAM_CONFIGS)} parameter+strategy combinations")


# Balanced trend regime (Donchian + ATR percentile + ADX)
DONCHIAN_TREND_WINDOW = int(os.getenv("DONCHIAN_TREND_WINDOW", "60"))
ADX_N = int(os.getenv("ADX_N", "14"))
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", "25.0"))
ADX_RANGE_MAX = float(os.getenv("ADX_RANGE_MAX", "20.0"))
ATR_TREND_PCT = float(os.getenv("ATR_TREND_PCT", "45.0"))
ATR_RANGE_PCT = float(os.getenv("ATR_RANGE_PCT", "30.0"))

ENGINE_VERSION = "v2_backtest_atr_adx"


print(f"[init] Retrain top {TOP_N_RETRAIN} configs with bid/ask-aware SL/TP")

# =========================
# Cache helpers
# =========================

def _cache_path(instrument: str, granularity: str, days_back: int) -> str:
    fname = f"{instrument}_{granularity}_{days_back}d.parquet"
    return os.path.join(CACHE_DIR, fname)


def load_or_fetch_history(
    instrument: str,
    granularity: str,
    days_back: int = DAYS_BACK,
) -> pd.DataFrame:
    """Load from Parquet cache or fetch from OANDA."""
    path = _cache_path(instrument, granularity, days_back)

    if os.path.exists(path):
        print(f"[cache] loading {path}")
        df = pd.read_parquet(path)
        print(f"[cache] loaded {len(df)} candles from cache")
        return df

    print(f"[cache] no cache found, fetching from OANDA...")
    df = fetch_history_oanda(instrument, granularity, days_back=days_back)

    print(f"[cache] saving history -> {path}")
    df.to_parquet(path, index=False)
    return df


# =========================
# SQLite helpers
# =========================

def db_connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init(db_path: str = DB_PATH):
    """Create tables for backtest logs (idempotent)."""
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          open_ts_utc   TEXT NOT NULL,
          close_ts_utc  TEXT NOT NULL,
          instrument    TEXT NOT NULL,
          granularity   TEXT NOT NULL,
          side          TEXT NOT NULL,
          units         INTEGER NOT NULL,
          entry_price   REAL NOT NULL,
          exit_price    REAL NOT NULL,
          sl_price      REAL NOT NULL,
          tp_price      REAL NOT NULL,
          pl            REAL NOT NULL,
          result        TEXT NOT NULL,
          risk_pct      REAL,
          sl_pips       REAL,
          tp_pips       REAL
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS equity (
          ts_utc      TEXT NOT NULL,
          equity      REAL NOT NULL,
          granularity TEXT NOT NULL,
          risk_pct    REAL,
          PRIMARY KEY (ts_utc, risk_pct)
        );
    """
    )

    conn.commit()
    conn.close()


def log_equity(
    equity: float,
    gran: str,
    risk_pct: float,
    ts_utc: str,
    db_path: str = DB_PATH,
):
    """Log equity using bar timestamp."""
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO equity (ts_utc, equity, granularity, risk_pct)
        VALUES (?, ?, ?, ?);
    """,
        (ts_utc, equity, gran, risk_pct),
    )
    conn.commit()
    conn.close()


def log_trade_row(
    open_ts: str,
    close_ts: str,
    instrument: str,
    gran: str,
    side: str,
    units: float,
    entry: float,
    exit_px: float,
    sl: float,
    tp: float,
    pl: float,
    result: str,
    risk_pct: float,
    sl_pips: float,
    tp_pips: float,
    db_path: str = DB_PATH,
):
    if not WRITE_TO_DB:
        return

    conn = db_connect(db_path)
    conn.execute(
        """
        INSERT INTO trades (
            open_ts_utc, close_ts_utc, instrument, granularity,
            side, units, entry_price, exit_price,
            sl_price, tp_price, pl, result, risk_pct, sl_pips, tp_pips
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """,
        (
            open_ts,
            close_ts,
            instrument,
            gran,
            side,
            units,
            entry,
            exit_px,
            sl,
            tp,
            pl,
            result,
            risk_pct,
            sl_pips,
            tp_pips,
        ),
    )
    conn.commit()
    conn.close()


# =========================
# Utility & pricing
# =========================

def pip_value(instrument: str) -> float:
    """Rough pip size: 0.01 for JPY crosses, 0.0001 otherwise."""
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def _price_decimals(instrument: str) -> int:
    """OANDA typical quoting: 3 decimals for JPY, 5 otherwise."""
    return 3 if instrument.endswith("_JPY") else 5


def price_add_pips(price: float, pips: float, side: str, instrument: str) -> float:
    """Add/subtract pips from price depending on side."""
    delta = pips * pip_value(instrument)
    dec = _price_decimals(instrument)
    return round(price + (delta if side.upper() == "LONG" else -delta), dec)


# =========================
# OANDA history fetcher
# =========================

def _max_chunk_days_for_gran(granularity: str) -> int:
    """OANDA max 5000 candles per request. Estimate safe days-per-chunk."""
    granularity = granularity.upper()
    candles_per_day_map = {
        "M1": 24 * 60,
        "M5": 24 * 12,
        "M15": 24 * 4,
        "M30": 24 * 2,
        "H1": 24,
        "H4": 6,
        "D": 1,
    }
    candles_per_day = candles_per_day_map.get(granularity, 500)
    max_days = max(1, 5000 // candles_per_day)
    return max_days


def fetch_history_oanda(
    instrument: str,
    granularity: str,
    days_back: int = DAYS_BACK,
) -> pd.DataFrame:
    """Pull historical candles from OANDA with automatic chunking."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    all_rows = []
    max_chunk_days = _max_chunk_days_for_gran(granularity)

    print(
        f"[hist] fetching ~{days_back} days of {instrument} {granularity} "
        f"with chunk size {max_chunk_days} day(s)..."
    )

    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=max_chunk_days), end)

        params = {
            "granularity": granularity,
            "from": cur_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": cur_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }
        url = f"{BASE}/v3/instruments/{instrument}/candles"

        r = session.get(url, params=params, timeout=30)

        if not r.ok:
            print("[hist:error]", r.status_code, r.text)
            r.raise_for_status()

        js = r.json()
        candles = js.get("candles", [])

        print(f"[hist] {cur_start.date()} → {cur_end.date()} got {len(candles)} candles")

        for c in candles:
            if not c.get("complete", False):
                continue
            all_rows.append(
                {
                    "t": c["time"],
                    "o": float(c["mid"]["o"]),
                    "h": float(c["mid"]["h"]),
                    "l": float(c["mid"]["l"]),
                    "c": float(c["mid"]["c"]),
                    "v": int(c["volume"]),
                }
            )

        cur_start = cur_end
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError("No candles returned from OANDA.")

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates(subset=["t"])
        .sort_values("t")
        .reset_index(drop=True)
    )
    print(f"[hist] total candles: {len(df)}")
    return df


# =========================
# Indicators & Enhancements
# =========================

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range (simple mean)."""
    h, l, c = df["h"].values, df["l"].values, df["c"].values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, index=df.index).rolling(period).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): EMA with alpha=1/period."""
    return series.ewm(alpha=1/period, adjust=False).mean()


def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder) for trend strength.

    Typical interpretation: ADX > 25 implies a stronger trend; ADX < 20 implies a weak / ranging market.
    """
    h, l, c = df["h"], df["l"], df["c"]
    prev_h, prev_l, prev_c = h.shift(1), l.shift(1), c.shift(1)

    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    up_move = h - prev_h
    dn_move = prev_l - l

    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)

    tr_rma = rma(tr, period)
    plus_di = 100.0 * rma(plus_dm, period) / tr_rma
    minus_di = 100.0 * rma(minus_dm, period) / tr_rma

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return rma(dx, period)


def donchian(df: pd.DataFrame, n=20) -> Tuple[pd.Series, pd.Series]:
    """Upper/lower channel extremes over n bars."""
    return df["h"].rolling(n).max(), df["l"].rolling(n).min()


def bollinger(df: pd.DataFrame, n=20, k=2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands on close."""
    ma = df["c"].rolling(n).mean()
    sd = df["c"].rolling(n).std(ddof=0)
    return ma, ma + k * sd, ma - k * sd


def rsi(df: pd.DataFrame, n=14) -> pd.Series:
    """Classic RSI (simple average)."""
    delta = df["c"].diff()
    up = (delta.clip(lower=0)).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def atr_percentile(current_atr: float, atr_series: pd.Series, lookback=500) -> float:
    """Where current ATR ranks vs recent ATR distribution (0..100%)."""
    base = atr_series.tail(min(len(atr_series), lookback)).dropna()
    if len(base) < 10:
        return 50.0
    return float((base <= current_atr).mean() * 100.0)


# ========== Enhancement 1: Volume Confirmation ==========

def volume_spike(df: pd.DataFrame, n=20, mult=1.5) -> bool:
    """Is current volume significantly above average?"""
    if len(df) < n:
        return True
    vol_ma = df["v"].rolling(n).mean()
    current_vol = float(df["v"].iloc[-1])
    vol_ma_val = float(vol_ma.iloc[-1])
    if vol_ma_val == 0:
        return True
    return current_vol > vol_ma_val * mult


# ========== Enhancement 2: Time-of-Day Liquidity Filter ==========

def is_liquid_hour(ts: pd.Timestamp) -> bool:
    """Avoid low-liquidity sessions (Tokyo/Sydney close + late Friday)."""
    hour = ts.hour
    weekday = ts.weekday()

    # Late Friday
    if weekday == 4 and hour >= 19:
        return False

    # Low-liquid Asia-only window (approx)
    if 2 <= hour < 8:
        return False

    return True


# ========== Enhancement 3: Swing Detection ==========

def swing_setup(df: pd.DataFrame, n=3) -> Tuple[bool, bool]:
    """
    up_swing: recent candles making higher lows (uptrend structure)
    dn_swing: recent candles making lower highs (downtrend structure)
    """
    if len(df) < n + 1:
        return False, False

    lows = df["l"].tail(n).values
    highs = df["h"].tail(n).values

    up_swing = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    dn_swing = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))

    return up_swing, dn_swing



# ========== Enhancement 5: Multi-Timeframe Confirmation ==========

def get_multi_tf_signal(signal_m5: str, signal_m15: str) -> str:
    """Only take trades if M15 + M5 align on direction."""
    if signal_m5 == "FLAT" or signal_m15 == "FLAT":
        return "FLAT"
    if signal_m5 == signal_m15:
        return signal_m5
    return "FLAT"


def _resample_to_tf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample current df to higher TF (e.g. '15T') for multi-timeframe signal.
    Keeps same column structure: t, o, h, l, c, v
    """
    if df.empty:
        return df.copy()

    tmp = df.copy()
    ts = pd.to_datetime(tmp["t"], utc=True)
    tmp = tmp.set_index(ts)

    ohlc = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    out = tmp.resample(rule).agg(ohlc).dropna().reset_index(names="ts")
    out["t"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out = out[["t", "o", "h", "l", "c", "v"]]
    return out


# =========================
# Core Strategy Signal (single TF)
# =========================

def _compute_signal_single_tf(
    df: pd.DataFrame,
    instrument: str,
    sl_atr_mult: float,
    rr: float,
    strategy_mode: str,
    use_volume_confirm: bool,
    use_time_filter: bool,
    use_swing_detect: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Single timeframe signal logic (trend/range, filters).
    SL is based on ATR multiple, then clipped by MIN_SL_PIPS/MAX_SL_PIPS.
    Does NOT do multi-TF alignment.
    """
    required_bars = max(100, DONCHIAN_TREND_WINDOW + 5, ADX_N * 5)
    if len(df) < required_bars:
        return "FLAT", {"why": f"not_enough_bars_{len(df)}<{required_bars}"}

    df = df.copy()
    df["ATR14"] = atr(df, 14)
    df["ADX14"] = adx_series(df, ADX_N)
    dc_hi, dc_lo = donchian(df, DONCHIAN_TREND_WINDOW)
    ma20, bb_up, bb_lo = bollinger(df, 20, 2.0)
    df["RSI14"] = rsi(df, 14)

    c = df["c"].iloc[-1]
    prev_c = df["c"].iloc[-2]
    atr_now = float(df["ATR14"].iloc[-1])
    adx_now = float(df["ADX14"].iloc[-1])
    atr_pct = atr_percentile(atr_now, df["ATR14"], 500)

    ts = pd.to_datetime(df["t"].iloc[-1], utc=True)
    if use_time_filter and not is_liquid_hour(ts):
        return "FLAT", {"why": "illiquid_hour"}

    # ATR in pips
    atr_pips = atr_now / pip_value(instrument) if atr_now > 0 else 0.0

    trend_regime = (atr_pct >= ATR_TREND_PCT) and (adx_now >= ADX_TREND_THRESHOLD)
    range_regime = (atr_pct <= ATR_RANGE_PCT) or (adx_now < ADX_RANGE_MAX)

    # --- Strategy-mode gating ---
    use_trend_block = strategy_mode in ("FULL", "TREND_ONLY", "NO_MTF")
    use_range_block = strategy_mode in ("FULL", "RANGE_ONLY", "NO_MTF")

    # TREND_ONLY: ignore range setups entirely
    if strategy_mode == "TREND_ONLY" and not trend_regime:
        return "FLAT", {"why": f"weak_trend_atr{atr_pct:.0f}_adx{adx_now:.0f}"}

    # RANGE_ONLY: ignore trend setups entirely
    if strategy_mode == "RANGE_ONLY" and not range_regime:
        return "FLAT", {"why": f"weak_range_atr{atr_pct:.0f}_adx{adx_now:.0f}"}

    # Helper: build SL/TP in pips from ATR
    def build_sl_tp(atr_pips_val: float, current_price: float) -> Tuple[float, float]:
        """
        Build SL/TP distances in pips from ATR and clip them safely.

        - Base SL = ATR(pips) * sl_atr_mult
        - Enforce hard bounds: MIN_SL_PIPS .. MAX_SL_PIPS
        - Optional percent floor: MIN_SL_PCT_OF_PRICE (disabled by default)
        """
        if atr_pips_val <= 0:
            return 0.0, 0.0

        sl_pips_raw = atr_pips_val * sl_atr_mult

        # Optional percent-of-price floor (disabled by default; can be enabled via env)
        # Example: 0.0001 = 0.01% of price. Convert that price distance to pips.
        min_sl_pct = float(os.getenv("MIN_SL_PCT_OF_PRICE", "0.0"))
        min_sl_pips_from_pct = 0.0
        if min_sl_pct > 0:
            min_dist_price = current_price * min_sl_pct
            min_sl_pips_from_pct = min_dist_price / pip_value(instrument)

        sl_pips = max(MIN_SL_PIPS, min(sl_pips_raw, MAX_SL_PIPS))
        if min_sl_pips_from_pct > 0:
            sl_pips = max(sl_pips, min_sl_pips_from_pct)

        tp_pips = sl_pips * rr

        # Round to 0.1 pip for stability
        sl_pips = round(sl_pips, 1)
        tp_pips = round(tp_pips, 1)

        return sl_pips, tp_pips

    # --- Trend regime: Donchian breakout ---
    if use_trend_block and trend_regime:
        dc_hi_prev = dc_hi.shift(1)
        dc_lo_prev = dc_lo.shift(1)
        # Breakout of the *previous completed* channel (avoids current-bar leakage)
        up_break = (c > dc_hi_prev.iloc[-1]) and (prev_c <= dc_hi_prev.iloc[-2])
        dn_break = (c < dc_lo_prev.iloc[-1]) and (prev_c >= dc_lo_prev.iloc[-2])

        if up_break:
            if use_volume_confirm and not volume_spike(df, 20, 1.5):
                return "FLAT", {"why": "no_volume_confirm"}

            if use_swing_detect:
                up_swing, _ = swing_setup(df, 3)
                if not up_swing:
                    return "FLAT", {"why": "no_upswing"}

            sl_pips, tp_pips = build_sl_tp(atr_pips, c)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "LONG", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

        if dn_break:
            if use_volume_confirm and not volume_spike(df, 20, 1.5):
                return "FLAT", {"why": "no_volume_confirm"}

            if use_swing_detect:
                _, dn_swing = swing_setup(df, 3)
                if not dn_swing:
                    return "FLAT", {"why": "no_dnswing"}

            sl_pips, tp_pips = build_sl_tp(atr_pips, c)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "SHORT", {"mode": "trend", "sl_pips": sl_pips, "tp_pips": tp_pips}

    # --- Range regime: Bollinger + RSI ---
    if use_range_block and range_regime:
        rsi_val = df["RSI14"].iloc[-1]

        # Long mean-reversion
        if (c < bb_lo.iloc[-1]) and (rsi_val < 30):
            sl_pips, tp_pips = build_sl_tp(atr_pips, c)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "LONG", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

        # Short mean-reversion
        if (c > bb_up.iloc[-1]) and (rsi_val > 70):
            sl_pips, tp_pips = build_sl_tp(atr_pips, c)
            if sl_pips <= 0 or tp_pips <= 0:
                return "FLAT", {"why": "invalid_sl_tp"}
            return "SHORT", {"mode": "range", "sl_pips": sl_pips, "tp_pips": tp_pips}

    return "FLAT", {"why": "no_setup"}


def compute_signal(
    df: pd.DataFrame,
    instrument: str,
    sl_atr_mult: float,
    rr: float,
    strategy_mode: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Regime-aware logic with all enhancements + optional multi-timeframe confirmation.
    strategy_mode: "FULL", "TREND_ONLY", "RANGE_ONLY", "NO_MTF"
    """
    use_volume_confirm = USE_VOLUME_CONFIRM_BASE
    use_time_filter = USE_TIME_FILTER_BASE
    use_swing_detect = USE_SWING_DETECT_BASE

    # NO_MTF disables multi-TF even if env says yes
    use_multi_tf = USE_MULTI_TF_BASE and (strategy_mode != "NO_MTF")

    # Base signal on current timeframe (e.g. M5)
    signal_main, meta_main = _compute_signal_single_tf(
        df,
        instrument,
        sl_atr_mult,
        rr,
        strategy_mode,
        use_volume_confirm,
        use_time_filter,
        use_swing_detect,
    )

    # If multi-TF disabled or no directional signal, we're done
    if not use_multi_tf or signal_main in ("FLAT",):
        return signal_main, meta_main

    # Multi-TF: resample window to M15 and compute a second signal
    try:
        df_m15 = _resample_to_tf(df, "15T")
    except Exception as e:
        return signal_main, {**meta_main, "mtf_error": str(e)}

    if len(df_m15) < 100:
        return "FLAT", {"why": "not_enough_mtf_bars"}

    signal_m15, meta_m15 = _compute_signal_single_tf(
        df_m15,
        instrument,
        sl_atr_mult,
        rr,
        strategy_mode,
        use_volume_confirm,
        use_time_filter,
        use_swing_detect,
    )

    final_signal = get_multi_tf_signal(signal_main, signal_m15)

    if final_signal == "FLAT":
        return "FLAT", {
            "why": "multi_tf_mismatch",
            "m5_signal": signal_main,
            "m15_signal": signal_m15,
        }

    # Use SL/TP from main TF meta
    return final_signal, meta_main


# =========================
# Sizing
# =========================

def normalize_position_units(size: float) -> float:
    """
    Convert the configured size into internal 'position_units' used in P&L.

    - CONTRACTS/UNITS: pass-through
    - LOTS: convert to units via LOT_SIZE (FX convention)
    """
    mode = UNIT_MODE
    if mode in ("LOTS", "LOT"):
        return float(size) * float(LOT_SIZE)
    # CONTRACTS / UNITS
    return float(size)


def position_size(nav: float, sl_pips: float, instrument: str, risk_pct: float) -> float:
    """
    Position sizing based on risk per trade.

    Returns a broker-style "size" (may be contracts or units), but internally we normalize it
    using normalize_position_units() before P&L.

    For CFDs, you may want to set CONTRACT_MULTIPLIER to match the broker's point value.
    """
    risk_amount = nav * risk_pct
    if sl_pips <= 0:
        return 0.0

    # In price terms, stop distance = sl_pips * pip_size.
    stop_dist_price = sl_pips * pip_value(instrument)

    # If your instrument is not spot FX, set CONTRACT_MULTIPLIER accordingly.
    # P&L ~= (price change) * position_units * CONTRACT_MULTIPLIER
    # So risk_amount ~= stop_dist_price * position_units * CONTRACT_MULTIPLIER
    position_units = risk_amount / max(1e-12, (stop_dist_price * CONTRACT_MULTIPLIER))

    # Convert internal position_units to broker "size"
    if UNIT_MODE in ("LOTS", "LOT"):
        size = position_units / float(LOT_SIZE)
    else:
        size = position_units

    # Capital.com typically allows fractional sizes; keep 2 decimals by default
    size_step = float(os.getenv("SIZE_STEP", "0.01"))
    size = max(size_step, math.floor(size / size_step) * size_step)
    return float(size)


# =========================
# Backtest core (with bid/ask-aware SL/TP triggers)
# =========================

def simulate_backtest(
    df: pd.DataFrame,
    instrument: str,
    gran: str,
    start_equity: float,
    risk_pct: float,
    sl_atr_mult: float,
    rr: float,
    daily_max_losses: int,
    strategy_mode: str,
    config_name: str,
) -> Dict[str, Any]:
    """Run bar-by-bar backtest with given parameters and strategy mode."""

    # Create unique DB for this config (or disable via WRITE_TO_DB)
    if WRITE_TO_DB:
        safe_name = config_name.replace(".", "_").replace(" ", "_")
        db_path = os.path.join(RESULTS_DIR, f"{safe_name}.db")
        db_init(db_path)
    else:
        db_path = ""

    equity = start_equity
    equity_peak = start_equity
    max_dd_pct = 0.0

    position: Optional[Dict[str, Any]] = None

    trade_count = 0
    win_count = 0
    loss_count = 0
    total_pl = 0.0

    # Daily loss-limit tracking
    current_day = None
    daily_loss_count = 0
    daily_limit_hit = False  # stop new entries for the day once hit

    # For reporting: average SL/TP pips encountered (approx)
    sl_pips_sum = 0.0
    tp_pips_sum = 0.0
    sl_tp_count = 0

    # Precompute half-spread price for bid/ask approximation
    half_spread_price = 0.5 * SPREAD_PIPS * pip_value(instrument)

    for i in range(100, len(df)):
        bar = df.iloc[i]
        t_bar = pd.to_datetime(bar["t"], utc=True)
        o, h_mid, l_mid, c_mid = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])

        # --- Handle new day boundaries for daily loss logic ---
        day = t_bar.date()
        if current_day is None or day != current_day:
            current_day = day
            daily_loss_count = 0
            daily_limit_hit = False

        # 1) Manage open position
        if position is not None:
            side = position["side"]
            sl = position["sl"]
            tp = position["tp"]

            exit_price = None

            # Approximate BID/ASK extremes from MID extremes
            if side == "LONG":
                # LONG closes on BID -> BID = MID - half_spread
                eff_l = l_mid - half_spread_price
                eff_h = h_mid - half_spread_price

                sl_hit = eff_l <= sl
                tp_hit = eff_h >= tp

                if sl_hit and tp_hit:
                    exit_price = sl  # conservative: SL first
                elif sl_hit:
                    exit_price = sl
                elif tp_hit:
                    exit_price = tp
            else:  # SHORT
                # SHORT closes on ASK -> ASK = MID + half_spread
                eff_l = l_mid + half_spread_price
                eff_h = h_mid + half_spread_price

                sl_hit = eff_h >= sl
                tp_hit = eff_l <= tp

                if sl_hit and tp_hit:
                    exit_price = sl  # conservative
                elif sl_hit:
                    exit_price = sl
                elif tp_hit:
                    exit_price = tp

            if exit_price is not None:
                units = position["units"]
                entry = position["entry"]
                side_mult = 1 if side == "LONG" else -1

                # P&L (spread is already modeled via bid/ask entry+exit)
                position_units = normalize_position_units(units)
                pl = (exit_price - entry) * position_units * side_mult * CONTRACT_MULTIPLIER

                equity += pl
                total_pl += pl

                result = "WIN" if pl > 0 else "LOSS" if pl < 0 else "EVEN"

                log_trade_row(
                    open_ts=position["open_ts"],
                    close_ts=t_bar.isoformat(),
                    instrument=instrument,
                    gran=gran,
                    side=side,
                    units=units,
                    entry=entry,
                    exit_px=exit_price,
                    sl=sl,
                    tp=tp,
                    pl=pl,
                    result=result,
                    risk_pct=risk_pct,
                    sl_pips=position["sl_pips_pips"],
                    tp_pips=position["tp_pips_pips"],
                    db_path=db_path,
                )

                if result == "WIN":
                    win_count += 1
                elif result == "LOSS":
                    loss_count += 1
                    daily_loss_count += 1

                position = None

        # 2) Check daily loss limit
        if not daily_limit_hit and daily_loss_count >= daily_max_losses:
            daily_limit_hit = True

        # 3) Track max drawdown
        equity_peak = max(equity_peak, equity)
        if equity_peak > 0:
            dd_pct = (equity_peak - equity) / equity_peak * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

        # 4) Log equity using bar time
        log_equity(equity, gran, risk_pct, ts_utc=t_bar.isoformat(), db_path=db_path)

        # 5) Check for new signal (if no open position & daily loss limit not hit)
        if position is None and not daily_limit_hit:
            window = df.iloc[: i + 1]
            signal, meta = compute_signal(window, instrument, sl_atr_mult, rr, strategy_mode)
            if signal not in ("LONG", "SHORT"):
                continue

            if REVERSE_SIGNALS:
                actual_side = "SHORT" if signal == "LONG" else "LONG"
            else:
                actual_side = signal

            sl_pips_calc = float(meta.get("sl_pips", 0.0))
            tp_pips_calc = float(meta.get("tp_pips", 0.0))
            if sl_pips_calc <= 0 or tp_pips_calc <= 0:
                continue

            units = position_size(equity, sl_pips_calc, instrument, risk_pct)
            if units <= 0:
                continue

            # Entry modeled at the worse side of the spread
        if actual_side == "LONG":
            entry = c_mid + half_spread_price  # buy at ASK
        else:
            entry = c_mid - half_spread_price  # sell at BID
            sl_price = price_add_pips(
                entry,
                sl_pips_calc,
                "SHORT" if actual_side == "LONG" else "LONG",
                instrument,
            )
            tp_price = price_add_pips(entry, tp_pips_calc, actual_side, instrument)

            position = {
                "open_ts": t_bar.isoformat(),
                "side": actual_side,
                "units": float(units),
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "sl_pips_pips": sl_pips_calc,
                "tp_pips_pips": tp_pips_calc,
            }

            trade_count += 1
            sl_pips_sum += sl_pips_calc
            tp_pips_sum += tp_pips_calc
            sl_tp_count += 1

    # 6) Close any remaining open position at last bar close
    if position is not None:
        last_bar = df.iloc[-1]
        t_last = pd.to_datetime(last_bar["t"], utc=True)
        c_last_mid = float(last_bar["c"])
        side = position["side"]
        units = position["units"]
        entry = position["entry"]

        side_mult = 1 if side == "LONG" else -1
        position_units = normalize_position_units(units)
        pl = (c_last_mid - entry) * position_units * side_mult * CONTRACT_MULTIPLIER

        equity += pl
        total_pl += pl
        result = "WIN" if pl > 0 else "LOSS" if pl < 0 else "EVEN"

        log_trade_row(
            open_ts=position["open_ts"],
            close_ts=t_last.isoformat(),
            instrument=instrument,
            gran=gran,
            side=side,
            units=units,
            entry=entry,
            exit_px=c_last_mid,
            sl=position["sl"],
            tp=position["tp"],
            pl=pl,
            result=result,
            risk_pct=risk_pct,
            sl_pips=position["sl_pips_pips"],
            tp_pips=position["tp_pips_pips"],
            db_path=db_path,
        )

        if result == "WIN":
            win_count += 1
        elif result == "LOSS":
            loss_count += 1

        # final equity already accounted

    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
    return_pct = ((equity - start_equity) / start_equity * 100) if start_equity > 0 else 0.0
    pl_per_trade = (total_pl / trade_count) if trade_count > 0 else 0.0

    avg_sl_pips = (sl_pips_sum / sl_tp_count) if sl_tp_count > 0 else 0.0
    avg_tp_pips = (tp_pips_sum / sl_tp_count) if sl_tp_count > 0 else 0.0

    # Simple robustness score: return / max_dd
    if max_dd_pct > 0:
        score = return_pct / max_dd_pct
    else:
        score = return_pct  # no drawdown case

    return {
        "engine_version": ENGINE_VERSION,
        "strategy_mode": strategy_mode,
        "config": config_name,
        "total_trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "total_pl": round(total_pl, 2),
        "final_equity": round(equity, 2),
        "return_pct": round(return_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "pl_per_trade": round(pl_per_trade, 4),
        "score": round(score, 4),
        "db_path": db_path,
        "risk_pct": risk_pct,
        "sl_atr_mult": sl_atr_mult,
        "rr": rr,
        "avg_sl_pips": round(avg_sl_pips, 2),
        "avg_tp_pips": round(avg_tp_pips, 2),
        "daily_max_losses": daily_max_losses,
    }


# =========================
# Parallel worker for RETRAIN
# =========================

def run_one_train_config(args):
    (
        strategy_mode,
        risk_pct,
        sl_atr_mult,
        rr,
        daily_max_losses,
        instr,
        gran,
        start_equity,
        df_train,
    ) = args

    risk_str = f"{risk_pct:.3f}".rstrip("0").rstrip(".")
    config_name = f"{strategy_mode}_R{risk_str}_ATR{sl_atr_mult}_RR{rr}_DL{daily_max_losses}"

    print(f"[TRAIN] {config_name} ...", flush=True)

    try:
        result = simulate_backtest(
            df_train,
            instrument=instr,
            gran=gran,
            start_equity=start_equity,
            risk_pct=risk_pct,
            sl_atr_mult=sl_atr_mult,
            rr=rr,
            daily_max_losses=daily_max_losses,
            strategy_mode=strategy_mode,
            config_name=config_name,
        )
        return result
    except Exception as e:
        print(f"[ERROR] TRAIN {config_name}: {e}")
        return None


# =========================
# Helper: chunking
# =========================

def chunked(iterable: List[Any], size: int):
    """Yield successive chunks of length `size` from `iterable`."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


# =========================
# Main
# =========================

if __name__ == "__main__":
    instr = DEFAULT_INSTRUMENT
    gran = DEFAULT_GRAN

    print(f"\n[main] Loading historical data: {instr} {gran}...")
    df_hist = load_or_fetch_history(instr, gran, days_back=DAYS_BACK)

    if len(df_hist) < 200:
        raise SystemExit("Not enough candles for train/test.")

    split_idx = int(len(df_hist) * TRAIN_FRACTION)
    df_train = df_hist.iloc[:split_idx].reset_index(drop=True)
    df_test = df_hist.iloc[split_idx:].reset_index(drop=True)

    print(f"[main] Train size: {len(df_train)} bars, Test size: {len(df_test)} bars")
    print(f"[main] RESULTS_DIR = {RESULTS_DIR}")
    print(f"[main] Using ENGINE_VERSION = {ENGINE_VERSION}")

    # --- Resume support for TRAIN brute force ---
    done_configs = set()
    if os.path.exists(SUMMARY_RAW_PATH):
        try:
            existing = pd.read_csv(SUMMARY_RAW_PATH)
            if "config" in existing.columns and "engine_version" in existing.columns:
                existing = existing[existing["engine_version"] == ENGINE_VERSION].copy()
                done_configs = set(existing["config"].astype(str))
                print(f"[resume] Found {len(done_configs)} completed TRAIN configs in summary_raw.csv")
        except Exception as e:
            print(f"[resume] Could not read existing summary_raw.csv: {e}")
            done_configs = set()

    # --- Build task list over full grid ---
    tasks = []
    for (strategy_mode, risk_pct, sl_atr_mult, rr, daily_max_losses) in PARAM_CONFIGS:
        risk_str = f"{float(risk_pct):.3f}".rstrip("0").rstrip(".")
        config_name = f"{strategy_mode}_R{risk_str}_ATR{sl_atr_mult}_RR{rr}_DL{daily_max_losses}"
        if config_name in done_configs:
            continue

        tasks.append(
            (
                strategy_mode,
                float(risk_pct),
                float(sl_atr_mult),
                float(rr),
                int(daily_max_losses),
                instr,
                gran,
                START_EQUITY,
                df_train,
            )
        )

    if not tasks:
        print("[main] No new TRAIN configs to run.")
    else:
        CORES = multiprocessing.cpu_count()
        if N_JOBS_ENV:
            try:
                N_JOBS = int(N_JOBS_ENV)
            except ValueError:
                N_JOBS = max(1, CORES - 2)
        else:
            N_JOBS = max(1, CORES - 2)

        print(f"[main] Using {N_JOBS} parallel workers out of {CORES} cores")
        print(f"[main] Will run {len(tasks)} TRAIN configs in parallel (chunk size = {CHUNK_SIZE})\n")

        total_tasks = len(tasks)
        total_chunks = (total_tasks + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx, task_chunk in enumerate(chunked(tasks, CHUNK_SIZE), start=1):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Running TRAIN chunk {chunk_idx}/{total_chunks} with {len(task_chunk)} configs")

            results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
                delayed(run_one_train_config)(args) for args in task_chunk
            )

            for result in results:
                if result is None:
                    continue

                df_row = pd.DataFrame([result])
                write_header = not os.path.exists(SUMMARY_RAW_PATH)
                df_row.to_csv(
                    SUMMARY_RAW_PATH,
                    mode="a",
                    index=False,
                    header=write_header,
                )

                print(
                    f"[TRAIN DONE] {result['config']}: "
                    f"{result['total_trades']} trades | {result['win_rate_pct']}% WR | "
                    f"{result['return_pct']}% return | score={result['score']}"
                )

    # === Build ranked TRAIN summary from summary_raw.csv ===
    if not os.path.exists(SUMMARY_RAW_PATH):
        raise SystemExit("[summary] No summary_raw.csv found; nothing to rank/test.")

    print("\n" + "=" * 120)
    print("TRAIN BRUTE FORCE SUMMARY (BID/ASK + ADX BALANCED)")
    print("=" * 120)

    df_all = pd.read_csv(SUMMARY_RAW_PATH)

    # Keep only current engine version
    if "engine_version" in df_all.columns:
        df_all = df_all[df_all["engine_version"] == ENGINE_VERSION].copy()

    # Drop duplicate configs, keep the last occurrence
    df_all = df_all.drop_duplicates(subset=["config"], keep="last").reset_index(drop=True)

    # Filter: minimum trades + return floor
    df_all = df_all[df_all["total_trades"] >= MIN_TRADES].copy()
    df_all = df_all[df_all["return_pct"] > -20].copy()

    if df_all.empty:
        raise SystemExit("[summary] No configs meet filters after TRAIN brute force.")

    # Recompute score if missing
    if "score" not in df_all.columns or df_all["score"].isna().any():
        max_dd = df_all["max_dd_pct"].replace(0, np.nan)
        df_all["score"] = df_all["return_pct"] / max_dd
        df_all["score"] = df_all["score"].fillna(df_all["return_pct"])

    # Rank by robustness score
    df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)
    df_all.insert(0, "rank", df_all.index + 1)

    # Save ranked summary
    df_all.to_csv(SUMMARY_RANKED_PATH, index=False)
    print(f"\n[saved] ranked TRAIN summary -> {SUMMARY_RANKED_PATH}")

    # ===== TEST top N configs on TEST set =====
    print("\n" + "=" * 120)
    print(f"OUT-OF-SAMPLE TEST (BID/ASK + ADX) on last {(1-TRAIN_FRACTION)*100:.1f}% of data")
    print("=" * 120)

    top_for_test = df_all.head(min(TOP_N_TEST, len(df_all))).copy()

    test_results = []
    for _, row in top_for_test.iterrows():
        strategy_mode = str(row["strategy_mode"])
        risk_pct = float(row["risk_pct"])
        sl_atr_mult = float(row["sl_atr_mult"])
        rr = float(row["rr"])
        daily_max_losses = int(row["daily_max_losses"])
        train_config_name = str(row["config"])
        test_config_name = train_config_name + "_TEST"

        print(f"\n[TEST] {test_config_name} ...", end=" ")

        try:
            res_test = simulate_backtest(
                df_test,
                instrument=instr,
                gran=gran,
                start_equity=START_EQUITY,
                risk_pct=risk_pct,
                sl_atr_mult=sl_atr_mult,
                rr=rr,
                daily_max_losses=daily_max_losses,
                strategy_mode=strategy_mode,
                config_name=test_config_name,
            )
            res_test["train_config"] = train_config_name
            test_results.append(res_test)

            print(
                f"✓ trades={res_test['total_trades']} | WR={res_test['win_rate_pct']}% | "
                f"ret={res_test['return_pct']}% | DD={res_test['max_dd_pct']}% | score={res_test['score']}"
            )
        except Exception as e:
            print(f"✗ Error: {e}")

    if not test_results:
        raise SystemExit("[test] No test results produced.")

    df_test_res = pd.DataFrame(test_results)

    print("\n" + "=" * 120)
    print("TEST RESULTS (TOP TRAIN CONFIGS, OUT-OF-SAMPLE, BID/ASK + ADX)")
    print("=" * 120)

    df_test_res = df_test_res.sort_values("score", ascending=False).reset_index(drop=True)
    df_test_res.insert(0, "rank", df_test_res.index + 1)

    print(
        df_test_res[
            [
                "rank",
                "strategy_mode",
                "config",
                "train_config",
                "total_trades",
                "win_rate_pct",
                "total_pl",
                "return_pct",
                "max_dd_pct",
                "score",
                "risk_pct",
                "sl_atr_mult",
                "rr",
                "daily_max_losses",
            ]
        ].to_string(index=False)
    )

    test_out_path = os.path.join(RESULTS_DIR, "summary_test.csv")
    df_test_res.to_csv(test_out_path, index=False)
    print(f"\n[saved] TEST summary -> {test_out_path}")

    # ===== TRAIN vs TEST comparison + overfit flag + overall ranking =====
    print("\n" + "=" * 120)
    print("TRAIN vs TEST COMPARISON (BID/ASK + ADX, OVERALL RANKING & OVERFIT CHECK)")
    print("=" * 120)

    # TRAIN metrics
    train_cols = [
        "config",
        "total_trades",
        "return_pct",
        "max_dd_pct",
        "score",
        "rank",
        "risk_pct",
        "sl_atr_mult",
        "rr",
        "daily_max_losses",
    ]
    df_train_merge = df_all[train_cols].copy()
    df_train_merge = df_train_merge.rename(
        columns={
            "config": "train_config",
            "total_trades": "total_trades_train",
            "return_pct": "return_pct_train",
            "max_dd_pct": "max_dd_pct_train",
            "score": "score_train",
            "rank": "train_rank",
            "risk_pct": "risk_pct_train",
            "sl_atr_mult": "sl_atr_mult_train",
            "rr": "rr_train",
            "daily_max_losses": "daily_max_losses_train",
        }
    )

    # TEST metrics
    test_cols = [
        "train_config",
        "total_trades",
        "return_pct",
        "max_dd_pct",
        "score",
        "rank",
    ]
    df_test_merge = df_test_res[test_cols].copy()
    df_test_merge = df_test_merge.rename(
        columns={
            "total_trades": "total_trades_test",
            "return_pct": "return_pct_test",
            "max_dd_pct": "max_dd_pct_test",
            "score": "score_test",
            "rank": "test_rank",
        }
    )

    df_cmp = pd.merge(df_train_merge, df_test_merge, on="train_config", how="inner")

    if df_cmp.empty:
        raise SystemExit("[cmp] No TRAIN/TEST pairs to compare.")

    df_cmp["score_diff_pct"] = np.where(
        df_cmp["score_train"] != 0,
        (df_cmp["score_test"] - df_cmp["score_train"]) / df_cmp["score_train"] * 100.0,
        np.nan,
    )

    cond_score_drop = df_cmp["score_test"] < df_cmp["score_train"] * 0.5
    cond_ret_drop = df_cmp["return_pct_test"] < df_cmp["return_pct_train"] * 0.4
    cond_dd_worse = df_cmp["max_dd_pct_test"] > df_cmp["max_dd_pct_train"] * 2.0
    cond_trades_drop = df_cmp["total_trades_test"] < df_cmp["total_trades_train"] * 0.5

    df_cmp["overfit"] = np.where(
        cond_score_drop | cond_ret_drop | cond_dd_worse | cond_trades_drop,
        "YES",
        "NO",
    )

    df_cmp["overall_score"] = 0.7 * df_cmp["score_test"] + 0.3 * df_cmp["score_train"]

    df_cmp = df_cmp.sort_values("overall_score", ascending=False).reset_index(drop=True)
    df_cmp.insert(0, "overall_rank", df_cmp.index + 1)

    print(
        df_cmp[
            [
                "overall_rank",
                "train_rank",
                "test_rank",
                "train_config",
                "risk_pct_train",
                "sl_atr_mult_train",
                "rr_train",
                "daily_max_losses_train",
                "total_trades_train",
                "total_trades_test",
                "return_pct_train",
                "return_pct_test",
                "max_dd_pct_train",
                "max_dd_pct_test",
                "score_train",
                "score_test",
                "score_diff_pct",
                "overall_score",
                "overfit",
            ]
        ].to_string(index=False)
    )

    combined_out_path = os.path.join(RESULTS_DIR, "summary_combined.csv")
    df_cmp.to_csv(combined_out_path, index=False)
    print(f"\n[saved] TRAIN/TEST combined summary -> {combined_out_path}")

    print("\n[done] Brute force + test (bid/ask + ADX) complete.")
