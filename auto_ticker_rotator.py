#!/usr/bin/env python3
"""Weekly ticker rotation.

Aggregates the past 7 days of nightly journal entries + per-ticker
performance stats, asks Claude for rotation recommendations, applies them
with safety rails:

  - Never rotates out a ticker with an open position.
  - Always retains an "always include" whitelist (e.g. SPY, QQQ as benchmarks).
  - Caps the rotation diff: if AI suggests changing > MAX_CHURN_PCT of the
    list, the change is staged for manual approval (written to
    pending_ticker_change.json) instead of auto-applied.
  - Writes a full audit log entry with before/after + AI reasoning.

Run manually:    .venv/bin/python auto_ticker_rotator.py
Run via launchd: scheduled at 6:00 PM ET every Sunday
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
TICKERS_AUTO_PATH = BASE_DIR / "tickers_auto.json"
LOG_DIR = BASE_DIR / "logs"
JOURNAL_DIR = LOG_DIR / "journal"
ROTATION_LOG = LOG_DIR / "ticker_rotation.log"
PENDING_CHANGE_PATH = BASE_DIR / "pending_ticker_change.json"

# Safety constants
ALWAYS_INCLUDE_STOCKS: Set[str] = {"SPY", "QQQ"}  # benchmarks always retained
MAX_CHURN_PCT = 0.50  # if > 50% of tickers would change, stage for review instead
MIN_STOCKS = 6
# DEV NOTE (ticker automation fix 2026-06):
# MAX_STOCKS was 12. Rotator log (2026-05-31) showed validation failure:
# "stocks count 13 outside [6,12]" when Claude proposed a slightly larger list.
# Increased to 15 to give AI more flexibility while still bounded.
# Config-driven bounds would be even better in future, but this unblocks auto-rotation.
# Paired with enabling dexter_autofetch in config.yaml so the system can proactively
# research and rotate tickers instead of user manually setting based on Grok/Claude.
MAX_STOCKS = 15
MIN_CRYPTO = 2
MAX_CRYPTO = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto_rotator: %(message)s",
)
logger = logging.getLogger(__name__)


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def held_symbols(cfg: Dict[str, Any]) -> Set[str]:
    """Symbols currently held in Alpaca. We refuse to rotate these out."""
    held: Set[str] = set()
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            cfg["alpaca"]["api_key"],
            cfg["alpaca"]["api_secret"],
            cfg["alpaca"]["base_url"],
        )
        for p in api.list_positions():
            sym = p.symbol
            # Alpaca returns crypto as e.g. "BTCUSD"; our config uses "BTC/USD"
            held.add(sym)
            if "USD" in sym and "/" not in sym and len(sym) <= 8:
                base = sym.replace("USD", "")
                held.add(f"{base}/USD")
        logger.info(f"Currently held: {sorted(held)}")
    except Exception as exc:
        logger.warning(f"Couldn't list positions, treating no symbols as held: {exc}")
    return held


def recent_journal_entries(days: int = 7) -> str:
    """Return concatenated text of journal entries from the last `days` days.
    Empty string if none exist."""
    cutoff = datetime.now() - timedelta(days=days)
    chunks: List[str] = []
    if not JOURNAL_DIR.exists():
        return ""
    # Journal files are monthly: YYYY-MM.md
    for path in sorted(JOURNAL_DIR.glob("*.md")):
        try:
            text = path.read_text()
        except Exception:
            continue
        # Split on "## " headers (each entry begins with "## YYYY-MM-DD ...")
        entries = re.split(r"\n## ", text)
        for entry in entries:
            # Pull date from the start of the chunk
            m = re.match(r"(\d{4}-\d{2}-\d{2})", entry.strip())
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d")
            except ValueError:
                continue
            if d >= cutoff:
                chunks.append("## " + entry.strip())
    return "\n\n".join(chunks)


def per_ticker_stats(days: int = 7) -> Dict[str, Dict[str, float]]:
    """Aggregate trade-log CSVs over the last `days` days into per-ticker stats.
    Returns: {ticker: {"trades": int, "buys": int, "sells": int, "avg_confidence": float}}"""
    stats: Dict[str, Dict[str, Any]] = {}
    cutoff = datetime.now() - timedelta(days=days)
    trade_dir = LOG_DIR / "trade_logs"
    if not trade_dir.exists():
        return stats
    for csv_path in sorted(trade_dir.glob("trade_log_*.csv")):
        # File name encodes the timestamp: trade_log_YYYY-MM-DD_HH-MM-SS.csv
        m = re.search(r"trade_log_(\d{4}-\d{2}-\d{2})", csv_path.name)
        if not m:
            continue
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            continue
        if d < cutoff:
            continue
        try:
            import csv as _csv
            with open(csv_path) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if row.get("timestamp", "").lower() == "summary":
                        continue
                    t = row.get("ticker")
                    if not t:
                        continue
                    rec = stats.setdefault(t, {"trades": 0, "buys": 0, "sells": 0,
                                               "conf_sum": 0.0, "conf_n": 0})
                    rec["trades"] += 1
                    if row.get("type") == "buy":
                        rec["buys"] += 1
                    elif row.get("type") == "sell":
                        rec["sells"] += 1
                    try:
                        c = float(row.get("confidence") or "")
                        rec["conf_sum"] += c
                        rec["conf_n"] += 1
                    except (ValueError, TypeError):
                        pass
        except Exception as exc:
            logger.warning(f"Failed reading {csv_path}: {exc}")

    # Compute averages
    out: Dict[str, Dict[str, float]] = {}
    for t, r in stats.items():
        avg_conf = (r["conf_sum"] / r["conf_n"]) if r["conf_n"] > 0 else 0.0
        out[t] = {
            "trades": r["trades"],
            "buys": r["buys"],
            "sells": r["sells"],
            "avg_confidence": avg_conf,
        }
    return out


def build_prompt(cfg: Dict[str, Any], journal: str, stats: Dict[str, Dict[str, float]],
                 held: Set[str]) -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    stocks = cfg.get("tickers", [])
    crypto = cfg.get("crypto", {}).get("tickers", [])

    stat_lines = []
    for t, s in sorted(stats.items()):
        stat_lines.append(
            f"  {t:>10}: {s['trades']:>3} trades  "
            f"(buys={s['buys']}, sells={s['sells']}, avg_conf={s['avg_confidence']:.2f})"
        )
    if not stat_lines:
        stat_lines.append("  (no trades this week)")

    held_str = ", ".join(sorted(held)) or "(none)"

    prompt = f"""You are reviewing a week of an algorithmic trading bot's behavior to recommend ticker rotations.
Today is {today}.

## Current watchlist
Stocks ({len(stocks)}): {', '.join(stocks)}
Crypto ({len(crypto)}): {', '.join(crypto)}

## CONSTRAINTS YOU MUST RESPECT
- Held positions (CANNOT rotate out): {held_str}
- Always include in stocks: {sorted(ALWAYS_INCLUDE_STOCKS)}
- Min/max stocks: {MIN_STOCKS}-{MAX_STOCKS}
- Min/max crypto: {MIN_CRYPTO}-{MAX_CRYPTO}
- Default to keeping the current list. Only rotate when there's a clear reason.
- Liquid, well-known tickers only. No micro-caps, no penny stocks.

## Per-ticker stats (last 7 days)
{chr(10).join(stat_lines)}

## This week's journal entries
{journal if journal else "(no journal entries available)"}

## Your task
Recommend the watchlist for next week. If no changes are warranted, return the
current list unchanged. If you propose changes, justify each one briefly.

Reply ONLY as JSON in this exact format:
{{
  "stocks": ["AAPL", "MSFT", ...],
  "crypto": ["BTC/USD", "ETH/USD", ...],
  "reasoning": "brief explanation of any changes, or 'no changes' if same",
  "changes": {{
    "added_stocks": [],
    "removed_stocks": [],
    "added_crypto": [],
    "removed_crypto": []
  }}
}}
"""
    return prompt


def call_claude(prompt: str, cfg: Dict[str, Any]) -> Optional[str]:
    api_key = (cfg.get("claude") or {}).get("api_key", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("No Claude API key available; aborting rotation.")
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
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(getattr(b, "text", "") for b in msg.content).strip()
    except Exception as exc:
        logger.error(f"Claude call failed: {exc}", exc_info=True)
        return None


def parse_recommendation(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON block from Claude's response. Returns None if unparseable."""
    if not raw:
        return None
    # Try fenced JSON first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = m.group(1) if m else None
    if not candidate:
        # Fallback: greedy match of the outermost {...}
        m = re.search(r"(\{.*\})", raw, re.DOTALL)
        candidate = m.group(1) if m else None
    if not candidate:
        return None
    cleaned = re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", candidate))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"JSON parse failed: {exc}")
        return None


def validate_recommendation(rec: Dict[str, Any], current_stocks: List[str],
                            current_crypto: List[str], held: Set[str]) -> Tuple[bool, str, Dict[str, Any]]:
    """Apply safety rails to the AI recommendation. Returns (ok, reason, sanitized_rec)."""
    new_stocks_raw = rec.get("stocks") or []
    new_crypto_raw = rec.get("crypto") or []

    new_stocks = sorted({str(s).upper().strip() for s in new_stocks_raw if str(s).strip()})
    new_crypto = sorted({str(c).strip() for c in new_crypto_raw if str(c).strip()})

    # Force-include whitelist + held positions
    for required in ALWAYS_INCLUDE_STOCKS:
        if required not in new_stocks:
            new_stocks.append(required)
    for h in held:
        # If a held symbol is a stock that got removed, restore it
        if h.isalpha() and h not in new_stocks and "/" not in h:
            new_stocks.append(h)
        # Same for crypto (e.g. "BTC/USD")
        if "/" in h and h not in new_crypto:
            new_crypto.append(h)
    new_stocks = sorted(set(new_stocks))
    new_crypto = sorted(set(new_crypto))

    # Bounds checks
    if not (MIN_STOCKS <= len(new_stocks) <= MAX_STOCKS):
        return False, f"stocks count {len(new_stocks)} outside [{MIN_STOCKS},{MAX_STOCKS}]", {}
    if not (MIN_CRYPTO <= len(new_crypto) <= MAX_CRYPTO):
        return False, f"crypto count {len(new_crypto)} outside [{MIN_CRYPTO},{MAX_CRYPTO}]", {}

    # Diff cap: if too much churn vs. current, stage instead of apply
    cur_stocks_set = set(current_stocks)
    cur_crypto_set = set(current_crypto)
    new_stocks_set = set(new_stocks)
    new_crypto_set = set(new_crypto)

    stock_changes = (cur_stocks_set ^ new_stocks_set)  # symmetric diff
    crypto_changes = (cur_crypto_set ^ new_crypto_set)
    total_universe = len(cur_stocks_set | cur_crypto_set | new_stocks_set | new_crypto_set)
    total_changes = len(stock_changes) + len(crypto_changes)
    churn_pct = (total_changes / total_universe) if total_universe else 0.0

    sanitized = {
        "stocks": new_stocks,
        "crypto": new_crypto,
        "reasoning": rec.get("reasoning", ""),
        "changes": {
            "added_stocks": sorted(new_stocks_set - cur_stocks_set),
            "removed_stocks": sorted(cur_stocks_set - new_stocks_set),
            "added_crypto": sorted(new_crypto_set - cur_crypto_set),
            "removed_crypto": sorted(cur_crypto_set - new_crypto_set),
        },
        "churn_pct": round(churn_pct, 3),
    }

    if churn_pct > MAX_CHURN_PCT:
        return False, f"churn {churn_pct:.0%} exceeds max {MAX_CHURN_PCT:.0%}; staging for review", sanitized

    return True, "ok", sanitized


def apply_change(sanitized: Dict[str, Any]) -> None:
    """Atomically rewrite config.yaml + tickers_auto.json with the new lists."""
    new_stocks = sanitized["stocks"]
    new_crypto = sanitized["crypto"]

    # Update config.yaml (preserve all other keys)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["tickers"] = new_stocks
    cfg.setdefault("crypto", {})["tickers"] = new_crypto
    tmp_cfg = CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    os.replace(tmp_cfg, CONFIG_PATH)

    # Update tickers_auto.json
    tmp_auto = TICKERS_AUTO_PATH.with_suffix(".json.tmp")
    with open(tmp_auto, "w") as f:
        json.dump({"tickers": new_stocks}, f)
    os.replace(tmp_auto, TICKERS_AUTO_PATH)


def stage_for_review(sanitized: Dict[str, Any], reason: str) -> None:
    """Write a pending change to disk so the dashboard can prompt for approval."""
    payload = {
        "staged_at": datetime.now().isoformat(timespec="seconds"),
        "reason_staged": reason,
        **sanitized,
    }
    with open(PENDING_CHANGE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.warning(f"Change staged for review: {PENDING_CHANGE_PATH}")


def audit_log(action: str, before: Dict[str, List[str]], after: Dict[str, List[str]],
              sanitized: Dict[str, Any], note: str = "") -> None:
    line = (
        f"\n=== {datetime.now().isoformat(timespec='seconds')} | {action} ===\n"
        f"BEFORE stocks: {before['stocks']}\n"
        f"BEFORE crypto: {before['crypto']}\n"
        f"AFTER  stocks: {after['stocks']}\n"
        f"AFTER  crypto: {after['crypto']}\n"
        f"Changes: {sanitized.get('changes', {})}\n"
        f"Churn: {sanitized.get('churn_pct')}\n"
        f"Reasoning: {sanitized.get('reasoning', '')}\n"
        f"Note: {note}\n"
    )
    with open(ROTATION_LOG, "a") as f:
        f.write(line)


def main() -> int:
    logger.info("Starting weekly ticker rotation.")
    try:
        cfg = load_cfg()
    except Exception as exc:
        logger.error(f"Couldn't load config: {exc}")
        return 1

    current_stocks = list(cfg.get("tickers", []))
    current_crypto = list(cfg.get("crypto", {}).get("tickers", []))
    before = {"stocks": current_stocks, "crypto": current_crypto}

    held = held_symbols(cfg)
    stats = per_ticker_stats(days=7)
    journal = recent_journal_entries(days=7)

    prompt = build_prompt(cfg, journal, stats, held)
    raw = call_claude(prompt, cfg)
    if not raw:
        logger.error("No response from Claude; rotation aborted.")
        audit_log("ABORTED", before, before, {}, note="No Claude response")
        return 2

    rec = parse_recommendation(raw)
    if not rec:
        logger.error("Couldn't parse Claude response as JSON; rotation aborted.")
        audit_log("ABORTED", before, before, {}, note=f"Unparseable response: {raw[:300]}")
        return 3

    ok, reason, sanitized = validate_recommendation(rec, current_stocks, current_crypto, held)

    if not ok:
        logger.warning(f"Validation failed: {reason}")
        if sanitized:
            stage_for_review(sanitized, reason)
            audit_log("STAGED", before, before, sanitized, note=reason)
        else:
            audit_log("REJECTED", before, before, {"reasoning": rec.get("reasoning", "")}, note=reason)
        return 4

    # Apply
    after = {"stocks": sanitized["stocks"], "crypto": sanitized["crypto"]}
    if before == after:
        logger.info("No changes recommended.")
        audit_log("NO_CHANGE", before, after, sanitized, note="AI returned current list unchanged")
        return 0

    try:
        apply_change(sanitized)
        audit_log("APPLIED", before, after, sanitized)
        logger.info(f"Applied rotation. Stocks: {after['stocks']}. Crypto: {after['crypto']}")
        return 0
    except Exception as exc:
        logger.error(f"Apply failed: {exc}", exc_info=True)
        audit_log("APPLY_FAILED", before, before, sanitized, note=str(exc))
        return 5


if __name__ == "__main__":
    sys.exit(main())
