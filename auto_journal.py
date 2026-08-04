#!/usr/bin/env python3
"""Nightly trading journal.

Runs once per trading-day evening. Pulls the day's trades, signal skip log,
and current Alpaca account snapshot, sends a structured digest to Claude,
and appends the response to logs/journal/YYYY-MM.md as a dated entry.

Does NOT change tickers, sizing, or any config -- pure observation. Designed
to fail safely (network blip, API down, etc.) without affecting the bot.

Run manually:    .venv/bin/python auto_journal.py
Run via launchd: scheduled at 6:30 PM ET weekdays (after market close + EOD cycle)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
LOG_DIR = BASE_DIR / "logs"
TRADE_LOG_DIR = LOG_DIR / "trade_logs"
JOURNAL_DIR = LOG_DIR / "journal"
BOT_SERVICE_LOG = LOG_DIR / "bot_service.log"

JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

# Configure script logging (writes to stderr; launchd captures it)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_journal: %(message)s",
)
logger = logging.getLogger(__name__)


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def todays_trades() -> List[Dict[str, str]]:
    """Read all trade_log CSVs touching today's date and return executed trade rows."""
    today = datetime.now().strftime("%Y-%m-%d")
    trades: List[Dict[str, str]] = []
    if not TRADE_LOG_DIR.exists():
        return trades
    for csv_path in sorted(TRADE_LOG_DIR.glob(f"trade_log_{today}_*.csv")):
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip the summary row that the bot appends
                    if row.get("timestamp", "").lower() == "summary":
                        continue
                    if row.get("ticker") and row.get("type"):
                        trades.append(row)
        except Exception as exc:
            logger.warning(f"Failed reading {csv_path}: {exc}")
    return trades


def todays_log_excerpts() -> Dict[str, List[str]]:
    """Scan the bot_service log for today and bucket the interesting lines."""
    today = datetime.now().strftime("%Y-%m-%d")
    buckets: Dict[str, List[str]] = {
        "skipped_low_confidence": [],
        "skipped_other": [],
        "errors": [],
        "cycle_summaries": [],
        "kelly": [],
    }
    if not BOT_SERVICE_LOG.exists():
        return buckets

    try:
        with open(BOT_SERVICE_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2_000_000))  # last ~2 MB is plenty for a day
            tail = f.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.warning(f"Failed reading bot log: {exc}")
        return buckets

    for line in tail.splitlines():
        if today not in line:
            continue
        if "Confidence too low" in line:
            buckets["skipped_low_confidence"].append(line)
        elif "Trade value" in line and "below" in line:
            buckets["skipped_other"].append(line)
        elif "Insufficient" in line or "Skipping" in line:
            buckets["skipped_other"].append(line)
        elif "[ERROR]" in line:
            buckets["errors"].append(line)
        elif "Live trading session completed" in line:
            buckets["cycle_summaries"].append(line)
        elif "Kelly" in line:
            buckets["kelly"].append(line)

    # Cap each bucket so we don't blast Claude with thousands of lines
    for key in buckets:
        buckets[key] = buckets[key][-30:]
    return buckets


def alpaca_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull current account + positions from Alpaca. Best-effort; returns
    empty fields on failure rather than raising."""
    snap = {"equity": None, "cash": None, "buying_power": None, "positions": []}
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            cfg["alpaca"]["api_key"],
            cfg["alpaca"]["api_secret"],
            cfg["alpaca"]["base_url"],
        )
        a = api.get_account()
        snap["equity"] = float(a.equity)
        snap["cash"] = float(a.cash)
        snap["buying_power"] = float(a.buying_power)
        snap["last_equity"] = float(a.last_equity)
        snap["day_pnl"] = float(a.equity) - float(a.last_equity)

        for p in api.list_positions():
            snap["positions"].append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
    except Exception as exc:
        logger.warning(f"Alpaca snapshot failed: {exc}")
        snap["error"] = str(exc)
    return snap


def build_digest(cfg: Dict[str, Any], trades: List[Dict[str, str]],
                 log_buckets: Dict[str, List[str]], snap: Dict[str, Any]) -> str:
    """Assemble the structured prompt for Claude."""
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    tickers = cfg.get("tickers", [])
    crypto = cfg.get("crypto", {}).get("tickers", [])

    # Confidence summary across executed trades
    conf_lines: List[str] = []
    for t in trades:
        conf = t.get("confidence")
        agr = t.get("agreement")
        d = t.get("direction")
        if conf:
            conf_lines.append(
                f"  - {t['ticker']:>10} {t['type']:>4} @ {t['price']:>10}  "
                f"conf={conf} agreement={agr} dir={d}"
            )
        else:
            conf_lines.append(
                f"  - {t['ticker']:>10} {t['type']:>4} @ {t['price']:>10}  qty={t.get('quantity', '?')}"
            )

    parts = [
        f"You are reviewing one trading day for an algorithmic trading bot.",
        f"Date: {today}",
        f"",
        f"## Watchlist",
        f"Stocks: {', '.join(tickers)}",
        f"Crypto: {', '.join(crypto)}",
        f"",
        f"## Account snapshot",
        f"Equity: ${snap.get('equity'):,.2f}" if snap.get("equity") else "Equity: (unavailable)",
        f"Cash: ${snap.get('cash'):,.2f}" if snap.get("cash") else "Cash: (unavailable)",
        f"Day P&L: ${snap.get('day_pnl', 0):+,.2f}" if "day_pnl" in snap else "",
        f"",
        f"## Open positions ({len(snap.get('positions', []))})",
    ]
    for p in snap.get("positions", []):
        parts.append(
            f"  - {p['symbol']:>10}  mv=${p['market_value']:,.2f}  "
            f"upl=${p['unrealized_pl']:+,.2f} ({p['unrealized_plpc']*100:+.1f}%)"
        )
    if not snap.get("positions"):
        parts.append("  (none)")

    parts.append("")
    parts.append(f"## Trades executed today ({len(trades)})")
    if conf_lines:
        parts.extend(conf_lines)
    else:
        parts.append("  (no trades executed)")

    parts.append("")
    parts.append(f"## Skipped: confidence too low ({len(log_buckets['skipped_low_confidence'])})")
    parts.extend([f"  {l}" for l in log_buckets["skipped_low_confidence"][:15]])
    parts.append("")
    parts.append(f"## Skipped: other ({len(log_buckets['skipped_other'])})")
    parts.extend([f"  {l}" for l in log_buckets["skipped_other"][:10]])
    parts.append("")
    parts.append(f"## Errors today ({len(log_buckets['errors'])})")
    parts.extend([f"  {l}" for l in log_buckets["errors"][:10]])
    parts.append("")
    parts.append(f"## Cycle summaries ({len(log_buckets['cycle_summaries'])})")
    parts.extend([f"  {l}" for l in log_buckets["cycle_summaries"]])

    parts.append("")
    parts.append("## Your task")
    parts.append(
        "Write a SHORT (under 250 words) journal entry for this day. Cover:\n"
        "1. Did anything notable happen? (Big P&L moves, error spikes, unusual skip rates)\n"
        "2. Are any tickers consistently failing the confidence filter? Which?\n"
        "3. Is the bot acting as expected, or is something drifting?\n"
        "4. One specific thing to watch tomorrow.\n"
        "\n"
        "DO NOT recommend ticker changes here -- that's the weekly rotation's job. Just observe."
    )
    return "\n".join(parts)


def call_claude(prompt: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Direct call to Anthropic's SDK. Returns the text response, or None on failure."""
    api_key = (cfg.get("claude") or {}).get("api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("No Claude API key available; aborting journal entry.")
        return None
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed.")
        return None

    model = (cfg.get("claude") or {}).get("model", "claude-haiku-4-5-20251001")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        # SDK returns content as a list of TextBlock-like objects
        chunks = []
        for block in msg.content:
            txt = getattr(block, "text", None)
            if txt:
                chunks.append(txt)
        return "\n".join(chunks).strip()
    except Exception as exc:
        logger.error(f"Claude call failed: {exc}", exc_info=True)
        return None


def append_journal(entry: str, prompt_summary: Dict[str, Any]) -> Path:
    """Append the entry to logs/journal/YYYY-MM.md. Returns the path."""
    now = datetime.now()
    journal_path = JOURNAL_DIR / f"{now.strftime('%Y-%m')}.md"
    header = f"\n\n---\n\n## {now.strftime('%Y-%m-%d %A %H:%M ET')}\n\n"
    metadata = (
        f"*Trades today: {prompt_summary['n_trades']} | "
        f"Skipped (low conf): {prompt_summary['n_skipped_low_conf']} | "
        f"Equity: ${prompt_summary['equity']:,.2f}*\n\n"
        if prompt_summary.get("equity") is not None else
        f"*Trades today: {prompt_summary['n_trades']} | "
        f"Skipped (low conf): {prompt_summary['n_skipped_low_conf']} | "
        f"Equity: unavailable*\n\n"
    )
    body = entry.strip() + "\n"
    with open(journal_path, "a") as f:
        f.write(header + metadata + body)
    return journal_path


def main() -> int:
    logger.info("Starting nightly journal run.")
    try:
        cfg = load_cfg()
    except Exception as exc:
        logger.error(f"Couldn't load config: {exc}")
        return 1

    trades = todays_trades()
    log_buckets = todays_log_excerpts()
    snap = alpaca_snapshot(cfg)

    summary = {
        "n_trades": len(trades),
        "n_skipped_low_conf": len(log_buckets["skipped_low_confidence"]),
        "n_errors": len(log_buckets["errors"]),
        "equity": snap.get("equity"),
    }
    logger.info(f"Digest summary: {summary}")

    prompt = build_digest(cfg, trades, log_buckets, snap)
    entry = call_claude(prompt, cfg)
    if not entry:
        # Save a stub so we have a record even if Claude failed
        entry = (
            f"_(Claude call failed; recording raw stats only.)_\n\n"
            f"Trades: {summary['n_trades']}, "
            f"Skipped low-confidence: {summary['n_skipped_low_conf']}, "
            f"Errors: {summary['n_errors']}, "
            f"Equity: {summary['equity']}"
        )

    path = append_journal(entry, summary)
    logger.info(f"Journal entry written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
