# Open-Source Framework Evaluation — 2026-07-24

All 6 repos cloned and code-reviewed against Trader_2025 + SEER (grounded in actual source, file paths verified). Verdict up front: **don't migrate to any of them** — extract specific pieces. Full migration of a working personal bot into any framework is a multi-month rewrite for marginal gain.

## Licensing map (governs HOW we adopt)

- **MIT — copy freely:** FinRL, vnpy, datasieve (the pip package behind FreqAI's best ideas)
- **Apache-2.0 — copy freely:** LEAN (but it's C#; pattern value only)
- **GPL-3.0 — do NOT paste code; port patterns or run as external tool:** Freqtrade, Backtrader
- **LGPL-3.0 — use as a pip dependency, don't vendor source:** NautilusTrader
(Private use is unrestricted for all; these rules only matter if the repo is ever shared/open-sourced — and it's on GitHub, so follow them.)

## Tier 1 — Quick wins (each ~0.5–2 days, adopt soon)

1. **`pip install datasieve` — out-of-distribution gate (from FreqAI, but MIT).** VarianceThreshold → scaler → PCA → SVM outlier removal → **DissimilarityIndex** (flags when today's feature vector is far from the training distribution → "distrust the model today"). This is the missing 10th confidence input: a self-doubt signal. Highest value-for-effort of everything reviewed.
2. **Model expiration + validation gate (PORT from FreqAI).** `check_if_model_expired` + null-prediction-when-stale pattern: an expired or validation-failing model returns *neutral* instead of trading. Directly fixes auto_retrain's min_precision=0.0 hole.
3. **vnpy `ts_*` factors (MIT, port ~15 functions from alpha_158).** Rolling regression slope/R²/residual, price↔volume correlation, up/down-day-count momentum, volume-weighted volatility, window quantile/rank. Pure pandas math, real feature diversity our RSI/MACD set lacks. SKIP the cross-sectional (`cs_*`/alpha_101) factors — meaningless on ~7 tickers.
4. **FinRL turbulence index (Mahalanobis market-stress).** A regime feature for the confidence score + forced-liquidation trigger pattern. Small port from `processor_alpaca.calculate_turbulence`.
5. **Protections layer (PORT pattern from freqtrade/plugins/protections).** Stoploss-guard (N stops in a window → pause), per-ticker cooldown (anti-revenge re-entry on 30-min cycles), windowed max-drawdown locks. Layered circuit breakers vs our single kill switch.
6. **Realistic fill model in the backtester (PORT pattern from backtrader bbroker).** Every simulated fill = price ± slippage clamped to bar high/low + commission. Cheap fix for part of the backtest/live gap.
7. **Quantitative watchlist pre-filter (PORT pattern from freqtrade pairlists).** Volatility-band + min-volume + range-stability filters run BEFORE Claude's 9:00 rotation → AI only picks from liquid, tradeable names.

## Tier 2 — Projects (worth it, scheduled)

8. **Replace the tabular Q-table with an offline-trained PPO (FinRL, MIT).** Train offline on our bars (state = cash/positions/tech per FinRL's env; reward = Δ portfolio value; their fixed-scaling normalization is load-bearing — copy exactly). Export SB3 policy → bot loads it as ONE confidence vote, never sole authority. ~1 week incl. weekly-retrain wiring. Cost: torch + stable-baselines3 deps. Their Alpaca paper-trading module itself is SKIP (deprecated SDK, market-orders only).
9. **Backtrader as external validation oracle (TOOL, separate venv — GPL-safe).** Feed our signals into a bt.Strategy; compare its slippage/commission-aware equity curve vs our homegrown backtester. Divergence localizes OUR bugs. This + #6 is the pragmatic answer to backtest≠live.

## Tier 3 — Strategic (SEER / future)

10. **NautilusTrader for SEER's Polymarket leg.** Production-grade Polymarket CLOB adapter exists (`adapters/polymarket/`): FOK/IOC/GTC/GTD, post-only, batch cancel, WS fill tracking, partial-fill reconciliation. FOK legs directly attack SEER's non-atomic-legs defect (leg fully fills or dies). BUT: no Kalshi adapter (multi-week build), and no framework can make cross-venue arb atomic. Adopt as a pip dependency for the Polymarket side only, IF SEER graduates to serious live trading. Also steal (as patterns): live reconciliation-on-restart, bracket OCO/OUO state machine, order emulator.
11. **LEAN: skip operationally.** Alpaca connector is an external C# plugin; running it solo means .NET builds + proprietary data format. Apache-licensed pattern library at best.

## What NOT to do

Full migration to anything; freqtrade's backtester/Telegram/RPC (coupled, GPL, redundant); vnpy outside vnpy.alpha (China A-share platform); FinRL's live-trading loop and ensemble agents; editing Nautilus's Rust core; LEAN as a local engine.

## Suggested order

datasieve OOD gate (#1) → retrain gate (#2) → ts_* factors (#3) → turbulence (#4) → protections (#5) — that's ~1 week of quick wins that measurably harden live decisions. Then fill model + backtrader oracle (#6/#9). PPO (#8) after Phase 2's research layer, since better inputs beat a better RL layer. Nautilus (#10) when SEER hardening items 1–3 are done.
