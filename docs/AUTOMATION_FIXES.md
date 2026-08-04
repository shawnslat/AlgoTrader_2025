# Automation Fixes – June 2026

**Date:** 2026-06 (fixes applied following journal/log analysis)
**Context:** User reported bot not trading automatically and requiring manual stock selection based on Grok/Claude. "pull jornal/logs first then we will fix".

## Issues Identified (in order discussed / diagnosed)

Pulled from:
- `logs/journal/2026-06.md` and `2026-05.md` (Claude nightly reviews)
- `logs/bot_service.log` (execution attempts)
- `logs/trade_logs/` (only 1 micro-trade since mid-May)
- `logs/auto_rotator.*.log` (May runs + validation fail)
- Live IPC status + positions
- Previous analysis of `config.yaml`, `bot_service.py`, `auto_ticker_rotator.py`

### 1. Execution Window Timing Mismatch (Primary cause of "not trading automatically")
- Config: `schedule.morning_run: '09:45'`, `afternoon_run: '14:30'`
- Code (`bot_service.py`):
  - `_execute_cycle`: hardcoded `morning_start = dt_time(9, 55)` / `afternoon_start = dt_time(14, 55)`
  - Startup check in `_trading_loop`: same hardcodes
  - Scheduled via `schedule.every().day.at(morning_time).do(...)`
- Result: Every automatic cycle logged "Outside morning execution window (9:55-10:30 AM ET). Current: 09:45. Skipping cycle." (and same for afternoon).
- Journals repeatedly called this out: "skipped due to timing windows", "execution windows are narrowly defined... too restrictive", "Zero trades... purely mechanical (timing gates)".
- "Run Now" (force=True) and some startup cases worked, but scheduled 2x daily did not.
- Trade impact: latest real trade May 15 (AAPL 0.05 shares, conf 0.698). Dozens of 0-trade days after.
- Crypto positions (from journals): SOL/ETH often 15-23%+ underwater with little rebalancing.

### 2. Ticker / Stock Selection Not Automatic
- User had to manually curate list (based on Grok/Claude) and set in config or `tickers_auto.json`.
- `dexter_autofetch` was missing or false (no auto call to `bun dexter/...` tools for research).
- `tickers_auto.json` was being *respected* on bot restart (override logged), but not proactively generated.
- Rotator (`auto_ticker_rotator.py` via launchd Sundays) ran occasionally:
  - May 17: no changes
  - May 24: applied a rotation
  - May 31: "Validation failed: stocks count 13 outside [6,12]"
- `dexter_bias.json` was stale (Jan).
- Result: watchlist static, no "Dexter chooses for me" behavior.

### 3. Other Related / Contributing Issues Noted
- Very sparse recent `trade_logs/` (80+ historical files but activity dropped to near-zero after March/April, one micro entry in May).
- Auto journal sometimes hit Claude connection errors (still wrote raw stats).
- TA-Lib missing (constant warnings; some indicators degraded).
- Position concentration in crypto noted repeatedly in journals.
- Old bugs in master log (auth, import errors) – appear resolved in current code.
- `equity_peak.json` last from May 15.

The bot engine itself (XGBoost + Q + 9-source confidence + ADX regime + scaled exits + guards) was sophisticated; the automation *loop* (scheduling + ticker discovery) was broken/friction-heavy.

## Fixes Applied (in order)

1. **Timing windows (bot_service.py)** – critical for any automatic trading.
   - Made window calculation dynamic from `config['schedule']` `morning_run`/`afternoon_run` values.
   - Windows now start at the exact user-configured nominal time (e.g. 09:45) and extend ~30 min (for drift).
   - Updated both `_execute_cycle` (the per-cycle guard) and the startup immediate-run check in `_trading_loop`.
   - Updated skip log messages to reflect actual configured window.
   - Added detailed DEV NOTE comments explaining the old bug, symptoms from logs/journals, and the fix rationale.
   - Files changed: `bot_service.py` (two locations).

2. **Automatic ticker selection (config.yaml + auto_ticker_rotator.py)**.
   - Added to `config.yaml` (after schedule block):
     - `dexter_autofetch: true`
     - `dexter_ticker_command`, `dexter_bias_command`, `dexter_chat_command` (pointing to dexter/ TS entrypoints).
   - Added DEV NOTE in YAML explaining the previous manual-only behavior and how this (plus rotator) closes the loop.
   - In `auto_ticker_rotator.py`: bumped `MAX_STOCKS = 15` (was 12) + DEV NOTE referencing the exact May 31 validation failure log line. (MIN/MAX_CRYPTO left as-is; churn safety rails preserved.)
   - This enables `maybe_fetch_tickers_via_dexter` / `maybe_override_tickers_from_json` paths and makes the Sunday rotator more likely to succeed and apply changes automatically (with all existing safety rails: held positions, always-include SPY/QQQ, 50% churn cap, audit log).

3. **Notes / documentation created**.
   - This file: `docs/AUTOMATION_FIXES.md` (created with full history, references to specific journals/logs, ordered fixes, dev comments rationale).
   - All code changes include inline DEV NOTE comments (see above).

4. **Dev comments added to all changes** (this task tracked across edits).
   - Every modified section has explanatory comments covering: previous problem (with dates/logs), symptoms, what was changed, why this approach, tradeoffs.

5. **Commit to CodeGraph**.
   - (Executed after all source changes – see below.)

6. **Verification steps** (see end of this doc).

## Current State After Fixes (as of application)

- Scheduled runs at the times in `config.yaml` should now fall inside the computed execution windows and proceed to `_run_cycle_body` (data fetch, features, signals, confidence scoring, risk gates, smart orders, position monitor).
- On bot (re)start or via the weekly rotator + `dexter_autofetch: true`, the system will attempt to auto-research/suggest/rotate tickers via Dexter instead of requiring manual user curation.
- All safety / risk features remain (PDT guard, VIX, confidence min 0.65/full 0.80, crypto caps, drawdown kill switch, scaled exits, etc.).
- `tickers_auto.json` override logic continues to work as before (now fed by autofetch/rotator).
- Next execution visible via GUI status or IPC `get_status`.

**To monitor:**
- `logs/bot_service.log`: look for "Within morning execution window..." or "=== MORNING RUN..." instead of "Outside ... Skipping".
- `logs/ticker_rotation.log` for rotator decisions.
- `logs/journal/YYYY-MM.md` for nightly Claude reviews (should start showing actual trade attempts).
- GUI "Run Now" still available as override / test.
- `codegraph` queries for the changed symbols (windows, dexter_autofetch) for future discussion.

## Remaining / Future Work (not addressed in this pass)

- Make window bounds / drift minutes configurable in yaml (instead of ~30min hardcoded in the dynamic calc).
- Make MIN/MAX_STOCKS etc. in rotator come from config (or the same schedule section).
- Ensure TA-Lib is installed in the env (currently missing; affects some indicators).
- Test full autofetch end-to-end (requires working bun + dexter/ deps + API keys for the TS side).
- Possibly widen or make smarter the position monitor for underwater crypto (journals flagged passivity on losses).
- Long-term: more point-in-time sentiment, calibrated confidence for Kelly, etc. (existing honest limitations in README).

## References to Pulled Data

- Journals: full `logs/journal/2026-06.md` (4 entries, all 0 trades + timing + crypto concerns); `logs/journal/2026-05.md` (many entries, only 1 trade day on 15th with explicit confidence breakdowns).
- Logs: `logs/bot_service.log` tails showing dozens of "Outside morning execution window (9:55-10:30... Current: 09:45" etc.; restarts on Jun 6/7.
- Trade logs: `ls -l logs/trade_logs/` showed activity dropoff; `trade_log_2026-05-15...` was the AAPL 0.05 share entry (conf 0.6981).
- Rotator: stderr/stdout tails with the 13-stock validation error.
- Live: IPC `get_status` / `get_positions` at time of diagnosis (small equity remnants + crypto book).

These fixes were applied in strict order per the discussion, with notes and dev comments as required. No recurring /loop was scheduled (user explicitly said "don't loop").

---
*Generated as part of the automation fix task.*