# Decision Log

## 2026-07-24

- **Phase 1 correctness fixes** applied to the stock trader (14 bugs — see `../CHANGES_2026-07-24.md`). Originals in `../backups/pre_claude_20260724/`.
- **PyQt GUI retired** → everything consolidated into the Streamlit dashboard (port 8502), which gained a Manual Trade tab. Archived (not deleted) in `../archive/gui_retired_20260724/`. dexter/ (TypeScript) kept — it feeds the veto gate until Phase 2 replaces it.
- **Day trading enabled** (`day_trading.enabled: true`, max 100/day). Rationale: FINRA PDT designation retired 2026-06-04; the bot's runtime gate now defers to config.
- **Intraday mode added**: cycle every 30 min, 09:45–15:15 ET, new entries hard-stop at `schedule.no_new_entries_after` (15:30) so nothing straddles the close. Replaces the 2-runs-per-day schedule when `intraday.enabled: true`.
- **Daily pre-market watchlist rotation** (09:00 ET weekdays): bot_service runs `auto_ticker_rotator.py` itself; existing rails apply (held positions protected, whitelist, 50% churn cap, audit log). The bot now picks its own stocks daily — no manual ticker changes needed.
- **SEER quick safety fixes**: live Polymarket US arb path now re-checks profitability at executable prices (was detecting on midpoints, executing blind); incomplete baskets rejected; live positions now count against the exposure cap (were exempt).
- **SEER launches with the trader**: `Trading Bot.command` starts SEER headless (`seer.py --no-ui`) if not already running.
- **NOT changed (needs Shawn's call):** SEER is LIVE with real money (`PAPER_TRADING_MODE = False` in kalshi_v2/config.py) while its loss tracking is fabricated (all trades auto-marked wins → kill switches can never fire). Recommendation on file: flip to paper until SEER hardening items 1–3 in ROADMAP are done. Secrets in kalshi_v2/config.py (Polymarket signing secret, Kalshi key, Telegram token) are hardcoded in a git repo — rotate.

## 2026-07-24 (later) — settings reset + supervision

- **Confidence weights rebalanced to sum to 1.0** (were 1.10). News cluster down-weighted per correlation finding (grok 0.15→0.08, sentiment 0.15→0.10 — they read the same headlines); decorrelated trio kept strong (MA 0.15, IBS 0.15, options 0.10→0.12); fundamental/insider bumped slightly (0.05→0.08/0.07) now their scoring bugs are fixed; technical trimmed 0.15→0.12 (overlaps MA); ml 0.15→0.13 (unproven edge). Weights remain hand-set — Platt calibration after ~200 trades is still the real fix.
- **Crypto allocation cap 50% → 25%** — half the portfolio in crypto was outsized for a $4k paper account with 1.5× risk multipliers.
- Config hot-reloads each cycle — no restart needed for these.
- **Scheduled Cowork supervision created**: (1) weekday 9:20 AM ET pre-market health check — bot process up, 9:00 rotation ran, watchlist sane; (2) weekday 4:30 PM ET post-close audit — pulls the day's logs/trades, verifies cycles ran every 30 min, entries respected the 15:30 cutoff, gates fired, flags bad patterns, writes a dated report to vault/daily_audits/. Both need the desktop app open to reach the Mac; they note-and-exit gracefully if not.
- **Framework evaluation completed** (see FRAMEWORK_EVALUATION.md): adopt-by-extraction only; Tier-1 quick wins = datasieve OOD gate, model-expiry gate, vnpy ts_* factors, turbulence index, protections layer.

## Standing context

- Trader is paper (Alpaca paper API, ~$4,050 equity). Bot was silently dead ~June 19 → July 23 (XGBoost feature mismatch: model wants Momentum/SMA_20, TA-Lib was missing). TA-Lib now present; watch the first intraday cycles.
- Trader API keys in `config.yaml` are plaintext in a GitHub folder — rotate + externalize (still open).
