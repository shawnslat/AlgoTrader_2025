# AlgoTrader 2025

Automated stock + crypto trading bot with hybrid ML + Q-learning signals, multi-source confidence scoring, a Streamlit dashboard, and supporting tooling for nightly journaling, weekly ticker rotation, and prediction-market arbitrage scanning.

> **Status:** This is paper-trading research software. Treat live results with skepticism. The "Honest limitations" section below lists known issues you should understand before trusting any backtest output.

## Features

- **Hybrid ML signals** -- XGBoost classifier on 30+ engineered features + Q-learning refinement layer
- **8-source confidence scoring** -- Claude sentiment, Grok bias, technical, ML prediction, FinViz fundamentals, insider/congress filings, MA crossover, IBS mean-reversion. Min 65% confidence to trade; 80% for full size.
- **Regime detection** -- ADX-based switching between momentum and mean-reversion strategies
- **Scaled exits** -- 3-tier partial profit taking (40% at +3%, 30% at +6%, 30% trailing)
- **Smart orders** -- Limit-first with IOC fallback to market
- **Crypto support** -- BTC/USD, ETH/USD, SOL/USD with separate risk params and a portfolio-level allocation cap
- **Risk gates** -- VIX threshold, PDT guard with regulatory cutover handling, confidence floor, dollar-minimum trade size
- **Kill switch** -- Two-click "Flatten All" button in the dashboard cancels every open order, market-sells every position, and halts the bot
- **Streamlit dashboard** -- Portfolio, positions, charts, market overview, SEER (prediction markets), AI chat, ticker manager, scaled exits, backtest, logs, income portfolio, and trading settings
- **Nightly AI journal** -- Claude reviews each trading day's activity and writes an entry to `logs/journal/YYYY-MM.md`
- **Weekly AI ticker rotation** -- Sunday job aggregates the journal + per-ticker stats and proposes watchlist changes, with safety rails (held positions never rotated out, 50% max churn, audit log)
- **macOS GUI** -- PyQt6 menu-bar tray app for bot control

## Quick Start

```bash
# 1. Set up the venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Edit config.yaml with your API keys (Alpaca, Polygon, Claude, Grok, NewsAPI, Finnhub)

# 3. Start everything
python trading_bot_app.py

# Or run components separately:
python bot_service.py                                  # trading service
.venv/bin/streamlit run dashboard.py --server.port 8502  # dashboard
python gui/menu_bar.py                                 # macOS tray icon
```

The bot trades twice on weekdays (defaults: **9:45 AM ET** for gap trades, **2:30 PM ET** for EOD signals). Both windows have ±15-30 min tolerance. Configurable in `config.yaml -> schedule`.

Dashboard runs on [http://localhost:8502](http://localhost:8502).

## Automated AI Workflows

Two scheduled jobs make the bot reactive to its own behavior without daily babysitting:

| Job | Frequency | Purpose |
| --- | --- | --- |
| `auto_journal.py` | Weekdays 6:30 PM ET | Sends today's trades, skips, errors, and account snapshot to Claude. Writes a dated entry to `logs/journal/YYYY-MM.md`. **Does not change tickers or config.** |
| `auto_ticker_rotator.py` | Sundays 6:00 PM ET | Aggregates the past 7 days of journal entries + per-ticker stats. Proposes a new watchlist. Auto-applies if churn ≤ 50%; stages for manual approval otherwise. |

Both run via macOS launchd. Install once with:

```bash
./launchd/install.sh           # install + load
./launchd/install.sh --remove  # uninstall
```

Both can also be triggered manually:

```bash
.venv/bin/python auto_journal.py
.venv/bin/python auto_ticker_rotator.py
```

The rotator's safety rails:

- Never rotates out a ticker with an open position
- Always retains an "always include" whitelist (`SPY`, `QQQ` by default)
- Bounds: 6-12 stocks, 2-5 crypto
- Diff cap: if AI proposes >50% churn, writes to `pending_ticker_change.json` for manual approval instead of auto-applying
- Atomic writes (temp file + rename) so a crash mid-write can't corrupt config
- Full audit log at `logs/ticker_rotation.log`

## Repository Layout

```text
Trader_2025/
├── trading_bot_app.py                # Unified launcher (spawns bot_service + dashboard)
├── Trader_main_Grok4_20250731.py     # Core trading engine + feature pipeline + backtest
├── bot_service.py                    # Background service, IPC server, scheduler
├── dashboard.py                      # Streamlit dashboard
├── signal_confidence.py              # 8-source confidence scoring
├── insider_congress.py               # Insider + congressional trade signals
├── finviz_enrichment.py              # FinViz fundamentals + risk scoring
├── dexter_gate.py                    # AI trade veto gate
├── pdt_guard.py                      # PDT day-trade limiter
├── position_monitor.py               # Scaled exits + stop-loss enforcement
├── trade_analyzer.py                 # FIFO trade matching + P&L
├── backtest_validation.py            # Sharpe, Sortino, walk-forward, Monte Carlo
├── auto_journal.py                   # Nightly AI journal (cron)
├── auto_ticker_rotator.py            # Weekly AI ticker rotation (cron)
├── config.yaml                       # Configuration (API keys, tickers, risk params)
├── tickers_auto.json                 # Active watchlist (overrides config.yaml.tickers)
├── launchd/                          # macOS LaunchAgent plists + installer
├── gui/                              # PyQt6 menu-bar app
├── seer/ + dexter/                   # Optional sub-projects
├── income_portfolio/                 # Passive income tracker
├── artifacts/                        # ML models, Q-table, backtest results (auto-generated)
├── data/                             # OHLCV CSVs (auto-generated)
└── logs/                             # Bot logs, trade logs, journal, audit logs
```

## Signal Flow

```text
Market Data (Polygon stocks, Alpaca crypto)
  -> Feature Engineering (RSI, MACD, BB, ATR, Stoch, ADX, IBS, vol regime, ...)
  -> Sentiment + FinViz Fundamentals (live cycles only -- skipped in backtest, see below)
  -> XGBoost Model -> Buy/Sell/Hold
  -> Q-Learning Refinement
  -> 8-Source Confidence Score
       < 0.65 = SKIP   |   0.65-0.79 = scaled position   |   >= 0.80 = full position
  -> Risk Gates (VIX, PDT cutover, day-trade count, crypto allocation cap)
  -> Smart Order (limit-first IOC, market fallback)
  -> Position Monitor (scaled exits, stops, trailing)
  -> Trade Log (with confidence + agreement + direction recorded)
```

## Backtest Cost Model

Backtests apply realistic retail friction. Defaults in `config.yaml -> backtest`:

| Field | Default | Note |
| --- | --- | --- |
| `slippage_pct` | 0.0005 (5 bps/side) | 10 bps round-trip |
| `transaction_cost_pct` | 0.001 (10 bps/side) | Stocks are 0 on Alpaca; crypto ~15-25 bps. Blended retail floor. |

Total round-trip drag is ~30 bps. Strategy must clear that floor before any backtest "edge" is real.

## Honest Limitations

Read this before believing any backtest result.

1. **Confidence is not a probability.** The 8-source score is a heuristic weighted vote, not a calibrated win probability. Kelly sizing was implemented and then **disabled** for this reason. After ~200+ trades with logged confidence (now logged in every trade CSV), Platt scaling can calibrate it and Kelly can be re-enabled.

2. **Sentiment + fundamentals leak in backtests.** `add_sentiment_features` and `add_fundamental_features` fetch *current* values and broadcast them across every historical bar. Backtests now neutralize these features (`is_backtest=True` flag) so models don't train on time-traveled data. Live cycles are unaffected -- today's bar correctly gets today's sentiment. The proper fix (point-in-time joins on `publishedAt` timestamps) is a future project.

3. **PDT rule cutover.** FINRA Rule 4210 amendments take effect **June 4, 2026**, eliminating the Pattern Day Trader designation. Both `bot_service.py` and the standalone `Trader_main` path enforce a runtime gate: before that date, `day_trading_enabled` is forced to `False` regardless of config to prevent PDT violations. After the date, config takes over automatically.

4. **Q-learning state space is undertrained.** The Q-table is stored as a CSV with no atomic-write guarantees. With <100 matched trades of history, RL has not learned anything statistically meaningful. Confidence + regime detection do the actual decision-making.

5. **Multi-AI consensus is correlated.** Claude and Grok are not independent voters; they read overlapping web data. The 15% + 15% weighting in the confidence score overestimates the second opinion's information content.

6. **Hardcoded confidence weights.** The 8 weights in `config.yaml -> confidence` are unjustified by data. They're a v1 starting point and should be stress-tested via sensitivity sweep.

## API Keys Required

| Key | Source | Cost |
| --- | --- | --- |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Free (paper) / Live |
| Polygon | [polygon.io](https://polygon.io) | Free tier OK |
| Claude | [console.anthropic.com](https://console.anthropic.com) | Paid (used for sentiment + journal + rotator) |
| Grok (xAI) | [console.x.ai](https://console.x.ai) | Paid (secondary AI bias) |
| NewsAPI | [newsapi.org](https://newsapi.org) | Free tier OK |
| Finnhub | [finnhub.io](https://finnhub.io) | Free tier OK |

## Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. The known limitations above are real -- the bot's backtests are subject to multiple sources of bias that have not yet been fully fixed. Do not deploy live capital based on backtest results without first running 3-6 months of paper trading and verifying the bot's live equity tracks (or beats) a passive index benchmark.

Past performance does not guarantee future results.

## License

MIT
