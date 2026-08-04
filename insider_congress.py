"""
Insider & Congressional Trading Data Module

Fetches and scores insider trading activity from two free data sources:
1. Senate Stock Watcher (GitHub JSON) - Congressional trades
2. Finnhub API (free tier) - Corporate insider transactions (Form 4)

Produces a per-ticker insider sentiment score for the confidence scoring system.
"""

import json
import logging
import math
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Finnhub availability
try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

# Module-level caches
_congress_data: Optional[List[Dict]] = None
_congress_cache_time: Optional[datetime] = None
_insider_cache: Dict[str, Dict[str, Any]] = {}
_insider_cache_time: Dict[str, datetime] = {}
_scores_cache: Dict[str, Dict[str, Any]] = {}
_scores_cache_time: Optional[datetime] = None

SENATE_DATA_URL = (
    "https://raw.githubusercontent.com/timothycarambat/senate-stock-watcher-data"
    "/master/aggregate/all_transactions.json"
)


def fetch_senate_trades(
    tickers: List[str],
    lookback_days: int = 90,
    cache_minutes: int = 60,
) -> Dict[str, List[Dict]]:
    """
    Fetch recent congressional trades from Senate Stock Watcher GitHub data.

    Returns a dict mapping ticker -> list of trade dicts.
    """
    global _congress_data, _congress_cache_time

    # Check cache
    if (
        _congress_data is not None
        and _congress_cache_time
        and (datetime.now() - _congress_cache_time).total_seconds() < cache_minutes * 60
    ):
        logger.debug("Using cached Senate trade data.")
    else:
        try:
            logger.info("Fetching Senate Stock Watcher data...")
            resp = requests.get(SENATE_DATA_URL, timeout=30)
            resp.raise_for_status()
            _congress_data = resp.json()
            _congress_cache_time = datetime.now()
            logger.info(f"Fetched {len(_congress_data)} total Senate transactions.")
        except Exception as e:
            logger.warning(f"Failed to fetch Senate data: {e}")
            if _congress_data is None:
                return {t: [] for t in tickers}

    cutoff = datetime.now() - timedelta(days=lookback_days)
    tickers_upper = {t.upper() for t in tickers}
    result: Dict[str, List[Dict]] = {t: [] for t in tickers}

    for txn in _congress_data:
        ticker = (txn.get("ticker") or "").upper().strip()
        if ticker not in tickers_upper or ticker == "--":
            continue

        txn_type = (txn.get("type") or "").strip()
        if not any(kw in txn_type for kw in ("Purchase", "Sale")):
            continue

        # Parse date (multiple formats seen in the data)
        date_str = txn.get("transaction_date") or ""
        txn_date = _parse_date(date_str)
        if txn_date is None or txn_date < cutoff:
            continue

        result[ticker].append({
            "date": txn_date,
            "type": "Purchase" if "Purchase" in txn_type else "Sale",
            "amount": txn.get("amount", ""),
            "senator": f"{txn.get('first_name', '')} {txn.get('last_name', '')}".strip(),
            "owner": txn.get("owner", ""),
        })

    return result


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try multiple date formats."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def fetch_insider_trades(
    ticker: str,
    finnhub_api_key: str,
    lookback_days: int = 90,
    cache_minutes: int = 60,
) -> List[Dict]:
    """
    Fetch insider transactions for a ticker via Finnhub API.
    Falls back to raw requests if finnhub package not installed.
    """
    global _insider_cache, _insider_cache_time

    # Check per-ticker cache
    cached_time = _insider_cache_time.get(ticker)
    if (
        cached_time
        and ticker in _insider_cache
        and (datetime.now() - cached_time).total_seconds() < cache_minutes * 60
    ):
        return _insider_cache[ticker]

    if not finnhub_api_key:
        return []

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    trades = []
    try:
        if FINNHUB_AVAILABLE:
            client = finnhub.Client(api_key=finnhub_api_key)
            data = client.stock_insider_transactions(ticker, from_date, to_date)
        else:
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/insider-transactions",
                params={"symbol": ticker, "from": from_date, "to": to_date, "token": finnhub_api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

        raw_trades = data.get("data", []) if isinstance(data, dict) else []

        for t in raw_trades:
            code = (t.get("transactionCode") or "").upper()
            # Only open-market purchases (P) and sales (S)
            if code not in ("P", "S"):
                continue

            txn_date = _parse_date(t.get("transactionDate", ""))
            if txn_date is None:
                continue

            change = t.get("change", 0) or 0
            trades.append({
                "date": txn_date,
                "type": "Purchase" if code == "P" else "Sale",
                "name": t.get("name", "Unknown"),
                "shares_changed": change,
                "shares_after": t.get("share", 0) or 0,
                "price": t.get("transactionPrice", 0) or 0,
            })

    except Exception as e:
        logger.warning(f"Finnhub insider fetch failed for {ticker}: {e}")

    _insider_cache[ticker] = trades
    _insider_cache_time[ticker] = datetime.now()
    return trades


def compute_insider_sentiment(
    ticker: str,
    congress_trades: List[Dict],
    insider_trades: List[Dict],
) -> Dict[str, Any]:
    """
    Compute a directional score from combined congressional and insider trades.

    Returns:
        {direction: str, strength: float, reason: str, details: dict}
    """
    now = datetime.now()
    congress_buys = 0
    congress_sells = 0
    insider_buys = 0
    insider_sells = 0

    # Score insider trades using a buy-signal approach.
    # Key insight: insider SELLS are mostly routine (executives liquidating options/RSUs).
    # For large-cap stocks, 95%+ of Form 4 filings are sales - that's the normal baseline.
    # Insider BUYS are the real signal (spending their own money = high conviction).
    # Strategy: only insider purchases move the needle bullish; all-sells is NEUTRAL (normal).
    insider_buy_weight = 0.0
    insider_sell_weight = 0.0
    for trade in insider_trades:
        days_ago = max(0, (now - trade["date"]).days)
        recency = math.exp(-days_ago / 30)
        is_buy = trade["type"] == "Purchase"
        # Weight by share count, normalized and capped
        base = min(2.0, abs(trade.get("shares_changed", 0)) / 10000)
        base = max(0.1, base)
        weight = base * recency

        if is_buy:
            insider_buys += 1
            insider_buy_weight += weight
        else:
            insider_sells += 1
            insider_sell_weight += weight

    # Congressional trades scored symmetrically (both buys and sells are meaningful)
    congress_score = 0.0
    congress_total = 0.0
    for trade in congress_trades:
        days_ago = max(0, (now - trade["date"]).days)
        recency = math.exp(-days_ago / 30)
        direction = 1.0 if trade["type"] == "Purchase" else -1.0
        weight = 1.0 * recency

        congress_score += direction * weight
        congress_total += abs(weight)

        if direction > 0:
            congress_buys += 1
        else:
            congress_sells += 1

    total_trades = congress_buys + congress_sells + insider_buys + insider_sells

    if total_trades == 0:
        return {
            "direction": "NEUTRAL",
            "strength": 0.0,
            "reason": f"No insider/congress trades found for {ticker}",
            "details": {},
        }

    # Compute net score using buy-signal approach for insiders:
    # - Insider buy ratio: what fraction of total insider weight is buying?
    #   0% buys (all sells) = NEUTRAL (normal baseline for big companies)
    #   >5% buys = slightly bullish
    #   >20% buys = clearly bullish (unusual insider buying)
    # - Congress: symmetric scoring (both directions meaningful)
    insider_total = insider_buy_weight + insider_sell_weight
    if insider_total > 0:
        buy_ratio = insider_buy_weight / insider_total
        # Map buy ratio to score: 0% -> -0.1, 5% -> 0.0, 20% -> 0.5, 50%+ -> 1.0
        insider_score = (buy_ratio - 0.05) * 2.0  # Centers around 5% as neutral
        insider_score = max(-0.2, min(1.0, insider_score))  # Cap the downside
    else:
        insider_score = 0.0

    # Combine: insider score + congress score
    if congress_total > 0:
        congress_norm = congress_score / congress_total  # [-1, 1]
        # Weight congress and insider equally if both present
        net_score = (insider_score + congress_norm) / 2.0
    else:
        net_score = insider_score

    net_score = max(-1.0, min(1.0, net_score))

    # Map to direction
    if net_score > 0.10:
        direction = "BULLISH"
    elif net_score < -0.15:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    strength = min(1.0, abs(net_score)) if direction != "NEUTRAL" else 0.2

    # Build reason string
    parts = []
    if congress_buys + congress_sells > 0:
        parts.append(f"Congress: {congress_buys}B/{congress_sells}S")
    if insider_buys + insider_sells > 0:
        parts.append(f"Insiders: {insider_buys}B/{insider_sells}S")
    reason = f"{ticker} {', '.join(parts)} (score={net_score:+.2f})"

    return {
        "direction": direction,
        "strength": strength,
        "reason": reason,
        "details": {
            "congress_buys": congress_buys,
            "congress_sells": congress_sells,
            "insider_buys": insider_buys,
            "insider_sells": insider_sells,
            "net_score": round(net_score, 3),
            "total_trades": total_trades,
        },
    }


def get_insider_scores(
    tickers: List[str],
    config: Dict,
    cache_minutes: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """
    Main entry point: fetch and score insider/congress data for all tickers.

    Args:
        tickers: List of ticker symbols
        config: Bot config dict (needs finnhub.api_key, insider_congress.*)

    Returns:
        {ticker: {direction, strength, reason, details}}
    """
    global _scores_cache, _scores_cache_time

    insider_cfg = config.get("insider_congress", {})
    if not insider_cfg.get("enabled", False):
        return {}

    # Check scores cache
    cache_min = insider_cfg.get("cache_minutes", cache_minutes)
    if (
        _scores_cache_time
        and (datetime.now() - _scores_cache_time).total_seconds() < cache_min * 60
    ):
        missing = [t for t in tickers if t not in _scores_cache]
        if not missing:
            return {t: _scores_cache[t] for t in tickers if t in _scores_cache}

    lookback = insider_cfg.get("lookback_days", 90)
    finnhub_key = config.get("finnhub", {}).get("api_key", "")

    # Filter out crypto tickers (contain '/')
    equity_tickers = [t for t in tickers if "/" not in t]

    # Fetch congressional data (single HTTP call for all tickers)
    congress_by_ticker = fetch_senate_trades(equity_tickers, lookback, cache_min)

    # Fetch insider data per ticker (with rate limiting)
    insider_by_ticker: Dict[str, List[Dict]] = {}
    for i, ticker in enumerate(equity_tickers):
        if finnhub_key:
            insider_by_ticker[ticker] = fetch_insider_trades(
                ticker, finnhub_key, lookback, cache_min
            )
            # Rate limit: 50 calls/min on free tier -> ~1.2s between calls
            if i < len(equity_tickers) - 1:
                time.sleep(1.2)
        else:
            insider_by_ticker[ticker] = []

    # Compute scores
    scores: Dict[str, Dict[str, Any]] = {}
    for ticker in equity_tickers:
        scores[ticker] = compute_insider_sentiment(
            ticker,
            congress_by_ticker.get(ticker, []),
            insider_by_ticker.get(ticker, []),
        )

    _scores_cache.update(scores)
    _scores_cache_time = datetime.now()

    logger.info(
        f"Insider/Congress scores computed for {len(scores)} tickers. "
        + ", ".join(f"{t}={s['direction']}" for t, s in scores.items())
    )

    return scores


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "NVDA", "TSLA"]
    print(f"\nTesting insider/congress scoring for: {test_tickers}\n")

    # Test with minimal config (no Finnhub key = congress-only)
    test_config = {
        "insider_congress": {"enabled": True, "lookback_days": 90, "cache_minutes": 60},
        "finnhub": {"api_key": ""},
    }

    scores = get_insider_scores(test_tickers, test_config)
    for ticker, score in scores.items():
        print(f"  {ticker}: {score['direction']} (strength={score['strength']:.2f})")
        print(f"    {score['reason']}")
        if score.get("details"):
            d = score["details"]
            print(f"    Congress: {d.get('congress_buys', 0)} buys, {d.get('congress_sells', 0)} sells")
            print(f"    Insiders: {d.get('insider_buys', 0)} buys, {d.get('insider_sells', 0)} sells")
            print(f"    Net score: {d.get('net_score', 0):+.3f}")
        print()
