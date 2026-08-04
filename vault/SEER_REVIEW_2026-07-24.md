# SEER (kalshi_v2) — Deep Code Review, 2026-07-24

Full review of the Kalshi/Polymarket bot. Quick fixes applied same day are marked ✅; everything else is open.

## Headline

**SEER is configured LIVE with real money** (`config.py`: `PAPER_TRADING_MODE = False`, `WATCH_MODE = False`, $200 bankroll cap, $25/order) while the docs describe it as paper-mode. Given D1/D2 below, the recommendation on file is: **flip to paper until hardening items 1–3 in ROADMAP.md are done.**

## What it is

`Seer.command` → `seer.py` → spawns `scanner.py` (60s loop) + Streamlit dashboard. `seer.py --no-ui` = scanner only (this is what the trader launcher now starts).

Strategies live-wired: Kalshi multi-outcome bracket arb; Kalshi YES+NO arb; Polymarket US arb (the live-executing Poly path); Polymarket intl + PredictIt (detect-only, partly fabricated quotes); crypto 5-min Up/Down (two implementations); one-sided Kalshi market making. Directional EV trading intentionally disabled (`probability.py` returns the mid; `LIVE_REQUIRE_ARB_ONLY=True`).

Platforms that actually work: Kalshi fully (RSA auth, real orders). Polymarket US if its SDK + funded account present. Intl Poly/PredictIt read-only.

## Critical defects

- **D1 — Kill switches are inert.** Daily-loss checks measure against a fictional $5,000 bankroll fed by fabricated wins (D2); `LIVE_DAILY_LOSS_LIMIT=$30` only counts resolutions of `_LIVE::` trades, which always resolve as wins. Neither can ever fire on a real loss.
- **D2 — Fabricated 100% win rate.** `resolve_paper_trades` settles every arb/live trade as `{"result":"yes"}` unconditionally (scanner.py ~501-509) → all P&L, win rates, bankroll, and dashboard metrics are fiction, and sizing grows on fake profits. THE most important bug.
- **D3 — Non-atomic arb legs.** All arb paths place sequential GTC limit orders; if leg 1 fills and leg 2 moves, you hold a naked position. Code literally prints "unwind manually." Needs FOK/IOC + true rollback.
- **D4 ✅ FIXED — Live Poly US path executed on midpoint detection with no executable-price re-check.** Now: real-quote-only legs, complete-basket requirement, spread/fee gate (2¢ buffer) before order placement (`polymarket_us_scanner.py`).
- **D5 — Secrets hardcoded in config.py** (Kalshi key id, Polymarket API key + SIGNING SECRET, Telegram token) as `os.getenv(..., "<literal>")` fallbacks, committed to GitHub. Rotate all; make env-only; scrub history.
- **D8 ✅ FIXED — Live positions were exempt from the exposure cap** (`risk_manager.py` excluded `*_LIVE::` ids). Live positions now count.

## Other defects (open)

- D6: N+1 API storm — three Kalshi scans re-fetch the same events/markets every 60s, Poly US BBO per-market per-scan, crypto bot every 15s → 429 risk. Needs a shared per-cycle cache.
- D7: Stale-price execution; crypto near-resolution decides winners from Binance spot while markets settle on their own index (adverse selection).
- D9: Market maker is one-sided (only ever buys YES), no inventory management, `mm_pnl` never updated, 5-min fill detection → adverse-selection collector. Fix or disable `MM_ENABLED`.
- D10: Emergency stop counter only increments in `scan_markets`; arb scans swallow exceptions → 5-error stop rarely triggers.
- D11: RiskManager resets on local midnight, live tracker on UTC.
- D12: AI fake-arb gate fails open and is off (no XAI key) → multi-outcome baskets protected only by the `mutually_exclusive`/`enableNegRisk` flags.

## Strategy quality

Good: arb math + fee model (rate·P·(1−P), taker/maker rates), spread-adjusted profit on Kalshi, honest no-model stance on directional, correct fractional-Kelly helper (unused).
Naive: top-of-book everywhere with no depth check (sizes fetched then ignored in the crypto bot); intl Poly spreads fabricated (mid ±2¢); no order-book imbalance; 5m-vs-15m advertised but unimplemented; arb sizing is an ad-hoc ramp, not Kelly.

## Integration

- No daemonization/PID/lock — two instances would double-submit. Launcher uses `--no-ui` + pgrep guard for now; a launchd service with restart-on-failure is the proper home.
- Health = freshness of `crypto_bot_status.json` / `mm_status.json` (`updated_at`) + `seer.log` tail. Don't delete the status JSONs — bots reload tracked orders from them.
- **Trader integration (to build):** SEER already scans Kalshi macro markets (Fed/CPI/jobs/recession). Export `signals.json` `{market, implied_prob, category, ts}` → stock bot consumes as an event-risk signal. Net-new but small; the highest-value bridge between the two systems.

## Ranked refinements

1. Real resolution tracking (kills D1+D2)  2. Atomic FOK/IOC legs (D3)  3. Kill switch on real balances  4. ✅ spread gate on live Poly  5. Secrets purge + rotation  6. Depth-aware sizing  7. Shared market cache (D6)  8. ✅ live exposure counting  9. Fix/disable market maker  10. Re-enable AI gate + order-book-imbalance filter.
