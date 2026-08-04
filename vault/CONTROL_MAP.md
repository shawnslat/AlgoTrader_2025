# Control Map — who decides what

Honest map of what is automated, what the AI decides, what needs Shawn, and how the system "learns." Updated 2026-07-24.

## Fully automated, rule-based (no AI judgment, runs itself)

- Scheduling: intraday cycles every 30 min 09:45–15:15 ET weekdays; entries cut off at 15:30
- Data: Polygon/Alpaca bars, feature engineering (RSI, MACD, ATR, ADX, IBS, etc.), live VIX
- Signal: XGBoost model prediction + regime detection (ADX momentum vs mean-reversion)
- Execution: order placement (bracket/limit-IOC/market), position sizing by ATR + confidence, scaled exits (40/30/30), trailing stops, stop-loss enforcement
- Risk rails: 12% drawdown kill switch, VIX threshold, crypto allocation cap, per-trade risk %, exposure caps, day-trade counter
- SEER: arb detection/execution, fee math, exposure caps (rule-based; its AI gate is currently off)

## AI-in-the-loop, automated (AI judges, rails constrain, no human needed)

- Claude sentiment on news headlines → one of 9 confidence inputs each cycle
- Grok directional bias → another confidence input (thin today; Phase 2 enriches it)
- Dexter bias file → veto gate on new buys (fail-open if stale >24h)
- 09:00 ET daily: Claude proposes the day's watchlist → auto-applies only if churn ≤50%; held positions and SPY/QQQ untouchable
- Nightly: Claude writes the trading journal (observational only — cannot change config)
- Weekly: model retrains on fresh data (auto_retrain) — flagged: its quality gate is effectively off (min_precision 0.0)

## Human-in-the-loop (needs Shawn)

- Watchlist changes >50% churn → parked in pending_ticker_change.json for approval
- Any config/risk parameter change; kill-switch reset; manual trades (dashboard tab)
- SEER live-vs-paper mode (OPEN DECISION — currently live with broken loss tracking)
- API key rotation (trader config.yaml + SEER config.py — both overdue)
- Adopting Phase 2 pieces (connectors, Ollama, SEER hardening priorities)

## Is it self-running? 

Yes for the trading day: launcher (or launchd) starts bot + dashboard + SEER + health icon; cycles, rotation, journal, and retrain fire on schedule with no babysitting. No auto-restart on crash yet (launchd plist exists for the bot; enabling restart-on-failure is a small task). Health = menu-bar dot (green/blue/red/gray) + dashboard.

## Is it learning / reward-driven? Honest answer

- **XGBoost**: learns weekly from new market data (supervised, next-day direction). Real learning, weak deployment gate (fix on roadmap: wire walk-forward validation into the gate).
- **Q-learning**: reward-driven by design (reward = realized price move/P&L) and it updates its table live — but with <100 matched trades it has learned nothing statistically meaningful yet, and we removed its 5% random exploration from live trading (it now only exploits learned values; exploration belongs offline).
- **Journal → rotation loop**: the real feedback loop today — nightly AI review of what happened feeds the weekly/daily watchlist decisions. Slow, qualitative, rail-guarded.
- **Confidence weights**: NOT learned — hand-set in config, flagged as unjustified; Platt calibration after ~200 logged trades is the roadmap fix that would let sizing become properly probability/Kelly-driven.
- **SEER**: currently learns nothing (its "results" are fabricated wins — hardening item #1); once real resolution tracking lands, its own trade history becomes usable for strategy validation.

So: self-running yes, rail-guarded AI judgment yes, genuinely reward-driven only partially — the RL layer is real but immature, and the honest learning happens in the weekly retrain + journal/rotation loop.
