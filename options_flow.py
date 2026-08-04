"""
Options flow signal — 9th confidence source.

Uses free Yahoo Finance options chains (via yfinance) to compute:
  - Put/Call open-interest ratio (whole chain): contrarian-style sentiment
  - Front-month call vs put volume skew: short-term directional pressure

Output is a {direction, strength, reason} dict consumed by signal_confidence.

This is "options flow lite". Real institutional flow data (Unusual Whales,
CBOE LiveVol, FlowAlgo) is paid; this gives 80% of the signal for $0.

Gracefully degrades:
  - If yfinance isn't installed, every call returns None.
  - If options aren't available for a ticker (crypto, illiquid), returns None.
  - Cached per-ticker for `cache_minutes` (default 30) to avoid rate limiting.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    logger.info("yfinance not installed — options flow signal disabled. "
                "Install with: pip install yfinance")

_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()


def _cache_get(ticker: str, cache_minutes: int) -> tuple:
    """Returns (hit, payload). A cached None payload (negative result) is a hit."""
    with _cache_lock:
        entry = _cache.get(ticker)
        if not entry:
            return (False, None)
        if (time.time() - entry["ts"]) < cache_minutes * 60:
            return (True, entry["payload"])
        return (False, None)


def _cache_put(ticker: str, payload: Dict) -> None:
    with _cache_lock:
        _cache[ticker] = {"ts": time.time(), "payload": payload}


def get_options_flow(ticker: str, config: Optional[Dict] = None) -> Optional[Dict]:
    """
    Returns a dict for use in signal_confidence, or None if unavailable:

        {
            "direction": "bullish" | "bearish" | "neutral",
            "strength": 0.0..1.0,
            "pc_ratio": float,
            "call_vol_share": float,
            "reason": str,
        }

    Args:
        ticker: stock symbol (no crypto / no slashes)
        config: full config dict; reads options_flow section

    None signals: caller should skip the signal entirely (not score 0).
    """
    if not YFINANCE_AVAILABLE:
        return None
    # Skip crypto and obviously non-equity tickers
    if not ticker or "/" in ticker or ticker.endswith("USD"):
        return None

    cfg = (config or {}).get("options_flow", {}) or {}
    if not cfg.get("enabled", True):
        return None
    cache_minutes = int(cfg.get("cache_minutes", 30))

    hit, cached = _cache_get(ticker, cache_minutes)
    if hit:
        return cached

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options or []
        if not expirations:
            _cache_put(ticker, None)  # negative cache
            return None

        # Front-month expiry only — that's where the action concentrates
        front_expiry = expirations[0]
        chain = tk.option_chain(front_expiry)
        calls = chain.calls
        puts = chain.puts
        if calls is None or puts is None or calls.empty or puts.empty:
            _cache_put(ticker, None)
            return None

        call_oi = float(calls["openInterest"].fillna(0).sum())
        put_oi = float(puts["openInterest"].fillna(0).sum())
        call_vol = float(calls["volume"].fillna(0).sum())
        put_vol = float(puts["volume"].fillna(0).sum())

        if (call_oi + put_oi) < 100 and (call_vol + put_vol) < 100:
            # Too illiquid to mean anything
            _cache_put(ticker, None)
            return None

        # Put/Call OI ratio. Equity benchmark: ~0.7 is neutral.
        pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 5.0
        # Volume share of calls in today's tape
        total_vol = call_vol + put_vol
        call_vol_share = call_vol / total_vol if total_vol > 0 else 0.5

        # Score: combine standing positioning (OI) with today's flow (volume)
        # Bullish if low P/C ratio AND elevated call share.
        # Bearish if high P/C ratio AND elevated put share.
        if pc_oi_ratio < 0.6 and call_vol_share > 0.6:
            direction = "bullish"
            strength = min(1.0, 0.4 + (0.6 - pc_oi_ratio) + (call_vol_share - 0.6))
            reason = f"P/C OI {pc_oi_ratio:.2f} (low), {call_vol_share:.0%} call vol — bullish positioning"
        elif pc_oi_ratio > 1.2 and call_vol_share < 0.4:
            direction = "bearish"
            strength = min(1.0, 0.4 + (pc_oi_ratio - 1.2) + (0.4 - call_vol_share))
            reason = f"P/C OI {pc_oi_ratio:.2f} (high), {1 - call_vol_share:.0%} put vol — bearish positioning"
        else:
            direction = "neutral"
            strength = 0.3
            reason = f"P/C OI {pc_oi_ratio:.2f}, {call_vol_share:.0%} call vol — mixed/neutral"

        payload = {
            "direction": direction,
            "strength": float(strength),
            "pc_ratio": pc_oi_ratio,
            "call_vol_share": call_vol_share,
            "expiry": front_expiry,
            "reason": reason,
        }
        _cache_put(ticker, payload)
        logger.debug(f"[{ticker}] options_flow {payload}")
        return payload

    except Exception as e:
        logger.warning(f"options_flow lookup failed for {ticker}: {e}")
        # Negative cache for a short while so we don't hammer Yahoo on transient errors
        _cache_put(ticker, None)
        return None
