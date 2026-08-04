# Roadmap

## Done (2026-07-24)

Phase 1 correctness (14 fixes), GUI consolidation into Streamlit, day trading enabled, intraday 30-min loop with pre-close cutoff, daily pre-market AI watchlist rotation, SEER quick safety fixes (executable-price gate, exposure cap), SEER co-launch with the trader, this vault.

## Phase 2 — Claude research layer (next big build)

A scheduled pre-market Claude task (and optional midday refresh) that replaces the thin Grok "guess from ticker list" bias with real research, writing enforced files the bot reads:

- **Inputs:** Bigdata.com + S&P Capital IQ (already connected), Alpha Vantage / FMP MCPs (available to connect: earnings calendar, options data, SEC filings, news), Kalshi macro odds exported from SEER (see below).
- **Outputs (files the bot already knows how to read):**
  - `dexter_bias.json` — per-ticker ok/caution/avoid with REAL earnings dates enforced (no entries within N days of earnings, FOMC, CPI)
  - `tickers_auto.json` — candidate watchlist for the 09:00 rotator to vet
- **Cost control:** route per-cycle sentiment to a local Ollama model (llama/qwen) with Claude as fallback — `sentiment.primary_provider` is already pluggable. Claude stays on the high-value jobs: pre-market brief, journal, rotation.
- **Fix while in there:** sentiment train/serve skew (train point-in-time from news_cache or drop the feature); down-weight the correlated news cluster (Grok+Claude+FinViz ≈ one signal ×3); collapse to one news-sentiment input.

## SEER hardening (before trusting any SEER numbers, and before more live money)

1. **Real resolution tracking** — stop auto-marking arb/live trades as wins; book actual per-leg P&L. Everything downstream (kill switch, metrics, sizing) is fiction until this lands.
2. **Atomic arb legs** — FOK/IOC on both legs + true rollback; today one leg can fill and leave a naked position.
3. **Kill switch on real account equity** — poll actual Kalshi/Polymarket balances; drive daily-loss limits off realized losses.
4. Purge hardcoded secrets from config.py (env-only) + rotate all keys (they're in git history).
5. Depth-aware sizing (size to available contracts at quote; the crypto bot already fetches bid/ask sizes and ignores them).
6. Shared market cache per cycle (three scans re-fetch the same Kalshi events → 429 risk).
7. Fix or disable the market maker (one-sided buy-YES = adverse-selection collector; MM P&L never tracked).
8. Re-enable the AI fake-arb gate (currently fails open with no XAI key set).

## SEER ↔ Trader integration (designed, small build)

SEER scans Kalshi macro markets (Fed, CPI, jobs, recession) every 60s but only logs them. Add a small exporter writing `signals.json` `{market, implied_prob, category, ts}` for macro markets → the stock bot's confidence layer reads it as a true event-risk input (e.g., size down ahead of high-uncertainty FOMC odds). This gives the trader a prediction-market nowcast — a genuinely decorrelated 9th/10th signal.

## Strategy upgrades from research (see RESEARCH_prediction_markets.md)

Priority order for SEER once hardening is done: executable-depth edge math everywhere → order-book imbalance filter on crypto near-resolution → cross-window checks (5m vs 15m implied prob) → fractional Kelly sizing on validated edges → inventory-aware two-sided MM.

## Later / evaluate

- Framework graduation path if the homegrown engine hits limits: NautilusTrader (multi-asset incl. prediction markets) or QuantConnect LEAN; FinRL for RL research to replace the undertrained Q-table.
- Platt-calibrate the confidence score after ~200 logged trades → re-enable fractional Kelly in the stock bot.
- Ollama benchmark: sentiment agreement rate vs Claude on the same headlines before switching.
