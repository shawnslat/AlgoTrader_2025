"""
News + sentiment cache for point-in-time (PIT) backtests.

Every live trading cycle that runs sentiment analysis writes the scored items
to data/news_cache.jsonl (append-only, one JSON object per line). Over time
this accumulates a PIT-correct corpus the backtester can use to look up
sentiment as of any past date — without re-querying paid news APIs.

Each cached record:
    {ticker, label, magnitude, recency_weight, publishedAt, scored_at}

The PIT join uses `publishedAt` as the "knowable as of" timestamp.
`scored_at` is when this bot processed it (forensics only).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / "data" / "news_cache.jsonl"


def append_to_cache(sentiments: Iterable[Dict]) -> int:
    """Append scored sentiment items to the on-disk cache. Returns count written.

    Items without publishedAt are skipped — they can't be used in a PIT join.
    """
    written = 0
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "a") as f:
            for s in sentiments:
                if not s.get("publishedAt"):
                    continue
                record = {
                    "ticker": s.get("ticker"),
                    "label": s.get("label"),
                    "magnitude": s.get("magnitude"),
                    "recency_weight": s.get("recency_weight"),
                    "publishedAt": s.get("publishedAt"),
                    "scored_at": scored_at,
                }
                f.write(json.dumps(record) + "\n")
                written += 1
    except Exception as e:
        logger.warning(f"Could not append to news cache: {e}")
    return written


def load_cache() -> pd.DataFrame:
    """Load the entire cache into a DataFrame. Returns empty if missing."""
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "label", "magnitude", "recency_weight", "publishedAt"])
    try:
        df = pd.read_json(CACHE_PATH, lines=True)
        if df.empty:
            return df
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
        df = df.dropna(subset=["publishedAt"])
        return df
    except Exception as e:
        logger.warning(f"Could not load news cache: {e}")
        return pd.DataFrame()


def pit_sentiment_for_bars(bars_df: pd.DataFrame, lookback_days: int = 3) -> pd.DataFrame:
    """
    For each (ticker, date) row in `bars_df`, compute a weighted sentiment score
    using only cache entries with publishedAt < the bar's date (end-of-day).

    Adds columns: Sentiment_Score, Sentiment_Magnitude (overwrites if present).
    Bars with no matching cached items get 0.0.

    `lookback_days` limits how far back articles are pulled per bar so old news
    doesn't dominate. Recency weight from the original scoring is still applied.

    Returns a NEW DataFrame; does not mutate the input.
    """
    if bars_df is None or bars_df.empty or "ticker" not in bars_df.columns or "date" not in bars_df.columns:
        logger.warning("pit_sentiment_for_bars: bars_df missing required columns; returning input unchanged")
        return bars_df

    cache = load_cache()
    if cache.empty:
        logger.info("News cache is empty — PIT sentiment will be 0 for all bars. Cache will grow as live cycles run.")
        out = bars_df.copy()
        out["Sentiment_Score"] = 0.0
        out["Sentiment_Magnitude"] = 0.0
        return out

    # Normalize bar dates to UTC end-of-day so "before this bar" is unambiguous
    bars = bars_df.copy()
    bars["_bar_dt"] = pd.to_datetime(bars["date"], errors="coerce", utc=True) + pd.Timedelta(hours=23, minutes=59)

    label_to_score = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    cache["_score"] = cache["label"].map(label_to_score).fillna(0.0)
    cache["_weight"] = (
        cache["magnitude"].astype(float).fillna(1.0)
        * cache["recency_weight"].astype(float).fillna(0.5)
    )

    scores: List[float] = []
    magnitudes: List[float] = []
    lookback = pd.Timedelta(days=lookback_days)

    for _, row in bars.iterrows():
        ticker = row["ticker"]
        bar_dt = row["_bar_dt"]
        if pd.isna(bar_dt):
            scores.append(0.0)
            magnitudes.append(0.0)
            continue

        window = cache[
            (cache["ticker"] == ticker)
            & (cache["publishedAt"] < bar_dt)
            & (cache["publishedAt"] >= bar_dt - lookback)
        ]
        if window.empty:
            scores.append(0.0)
            magnitudes.append(0.0)
            continue

        total_w = window["_weight"].sum()
        if total_w <= 0:
            scores.append(0.0)
        else:
            scores.append(float((window["_score"] * window["_weight"]).sum() / total_w))
        magnitudes.append(float(window["magnitude"].astype(float).mean()))

    out = bars.drop(columns=["_bar_dt"])
    out["Sentiment_Score"] = scores
    out["Sentiment_Magnitude"] = magnitudes
    return out


def cache_stats() -> Dict:
    """Quick stats for dashboard display."""
    if not CACHE_PATH.exists():
        return {"exists": False, "records": 0}
    try:
        df = load_cache()
        if df.empty:
            return {"exists": True, "records": 0}
        return {
            "exists": True,
            "records": int(len(df)),
            "tickers": sorted(df["ticker"].dropna().unique().tolist()),
            "earliest": df["publishedAt"].min().isoformat() if not df.empty else None,
            "latest": df["publishedAt"].max().isoformat() if not df.empty else None,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}
