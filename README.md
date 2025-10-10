# AlgoTrader 2025 — Hybrid ML + Q‑Learning Trading Bot

A production‑leaning, paper‑trading bot that combines:
- ML classification (XGBoost) on daily bars
- Technical indicators (ta, optional TA‑Lib)
- News sentiment via xAI Grok and NewsAPI
- A lightweight Q‑Learning layer to refine buy/sell/hold
- Live execution and positions via Alpaca, market data via Polygon
- 30‑minute continuous scheduling during US market hours

The code lives in `Trader_main_Grok4_20250731.py` and is driven by `config.yaml`.

> Important: This project is configured for paper trading by default (`https://paper-api.alpaca.markets`). Do not use with real funds without thorough evaluation and risk controls.


## What it does

- Downloads/loads recent daily OHLCV for configured tickers (Polygon)
- Engineers a rich feature set (MA, RSI, MACD, Bollinger, ATR, StochRSI, lags, volume change; optional TA‑Lib indicators)
- Pulls recent news and classifies sentiment per ticker using xAI Grok (via OpenAI SDK) and NewsAPI
- Trains/tunes an XGBoost classifier to predict next‑day up/down (Target)
- Evaluates with a confusion matrix and classification report
- Generates hybrid signals (ML + Q‑Learning + risk gates like VIX)
- Backtests with simple position/risk rules and outputs charts/CSV
- Runs a continuous live loop on market days: refresh data, recompute features, generate signals, and place paper orders via Alpaca


## Repository structure (key files)

- `Trader_main_Grok4_20250731.py` — Main bot (training, backtesting, live trading, scheduler)
- `config.yaml` — API keys, tickers, risk settings, schedule
- `valid_tickers.yaml` — Optional reference list of symbols
- `data/` — Daily CSVs per ticker (auto‑downloaded if missing)
- Generated artifacts (after runs):
  - `final_model.pkl` — Trained XGBoost model
  - `confusion_matrix.png` — Model evaluation heatmap
  - `backtesting_results.csv` and `backtesting_results.png` — Backtest equity curve
  - `q_table.csv`, `last_decisions.csv` — Q‑Learning state/action memory
  - `master_trading_bot.log` — Run logs
  - `trade_log_*.csv` — Live session trade log(s)


## Requirements

- Python 3.10+ (tested on macOS)
- Accounts/keys:
  - Alpaca (paper): trading and latest bars
  - Polygon: historical OHLCV (daily)
  - NewsAPI: recent headlines
  - xAI Grok API key (via OpenAI SDK with base_url override)
- Python packages (install via pip):
  - core: `pandas numpy requests pytz pyyaml joblib schedule aiohttp`
  - data/TA: `ta` (optional) `TA-Lib` (optional, see note)
  - plotting: `matplotlib seaborn`
  - ML: `xgboost scikit-learn`
  - brokers/data: `alpaca-trade-api`
  - LLM/news/social: `openai tweepy`

Optional TA‑Lib (native lib) on macOS:
- Install library: `brew install ta-lib`
- Then pip package: `pip install TA-Lib`
If TA‑Lib isn’t installed, the bot will continue with a reduced feature set and log a warning.


## Setup

1) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install pandas numpy requests pytz pyyaml joblib schedule aiohttp ta matplotlib seaborn xgboost scikit-learn alpaca-trade-api openai tweepy
# Optional: if you want additional indicators
# brew install ta-lib
# pip install TA-Lib
```

3) Configure your keys and settings

Edit `config.yaml` and replace placeholders with your keys. Example fields:

```yaml
alpaca:
  api_key:    "<YOUR_ALPACA_KEY>"
  api_secret: "<YOUR_ALPACA_SECRET>"
  base_url:   "https://paper-api.alpaca.markets"

polygon:
  api_key: "<YOUR_POLYGON_KEY>"

newsapi_token: "<YOUR_NEWSAPI_KEY>"
grok_api_key:  "<YOUR_XAI_GROK_API_KEY>"  # used via OpenAI SDK with x.ai base_url

tickers: [AAPL, TSLA, MSFT, NVDA]

risk_per_trade_pct: 0.2   # risk sizing input for position calc
max_position_pct:   5.0   # max position (informational)
stop_loss_pct:      0.05
take_profit_pct:    0.10
buying_power_pct:   50
vix_threshold:      25    # skip trades if VIX above this
market_hours:
  start: "09:30"
  end:   "16:00"
job_interval_minutes: 30
```

Security note: never commit real API keys. Consider using environment variables and template configs for production.


## How to run

From the project root:

```bash
python Trader_main_Grok4_20250731.py
```

On start, you’ll be prompted:
- “Have you downloaded, trained, and backtested the model today? (yes/no)”

Choose a path:

- no — Full pipeline
  1) Download/refresh daily data (Polygon) into `data/`
  2) Feature engineering + sentiment
  3) Train/tune XGBoost (RandomizedSearchCV, TimeSeriesSplit)
  4) Evaluate and save metrics (`confusion_matrix.png`), persist model (`final_model.pkl`)
  5) Backtest and produce `backtesting_results.csv/.png`

- yes — Skip training/backtest
  - Load `final_model.pkl` and start the continuous 30‑minute live loop using Alpaca paper trading


## Live (paper) trading loop

Once running, every ~30 minutes during US market hours the bot:
- Pulls any latest bars from Alpaca and merges with recent history from Polygon
- Recomputes indicators and sentiment
- Generates hybrid signals (ML + Q‑Learning, with VIX and risk screens)
- Places/monitors market orders via Alpaca paper API
- Logs to `master_trading_bot.log` and appends per‑session `trade_log_*.csv`

Notes:
- Market open check uses US/Eastern timezone (09:30–16:00)
- macOS notifications are sent for fills using AppleScript (`osascript`)


## Outputs and logs

- `final_model.pkl` — persisted XGBoost model
- `confusion_matrix.png` — evaluation
- `backtesting_results.csv` / `backtesting_results.png` — equity curve
- `q_table.csv` — tabular Q values (states x actions)
- `last_decisions.csv` — last state/action/price per ticker
- `master_trading_bot.log` — run‑wide log
- `trade_log_*.csv` — per run/session trade events and a summary row


## Troubleshooting

- TA‑Lib warning: If you see “TA‑Lib is not installed,” either install it (see above) or proceed without it. The bot will still run.
- 401 Unauthorized (Polygon/Alpaca/NewsAPI/Grok): Check keys in `config.yaml` and account status. Paper vs live base URLs must match account type.
- 429 Too Many Requests: The code waits between requests, but free tiers rate‑limit aggressively. Reduce tickers or increase intervals.
- `final_model.pkl` not found: Run the training path (`no`) first to produce the model file.
- NaN errors during evaluation: The script drops NaNs and aligns indices; ensure sufficient history per ticker (at least 50–200 daily bars recommended).
- Empty/insufficient data: Make sure `tickers` exist on Polygon/Alpaca and that `data/` contains recent files if you’re offline.
- macOS notifications fail: AppleScript requires macOS. On other OSes, disable or ignore notifications.


## How it works (architecture)

- Data: Daily OHLCV is pulled from Polygon (`/v2/aggs/ticker/{symbol}/range/1/day/...`). Latest one‑day bars are fetched from Alpaca during live mode.
- Features: `ta` indicators (MA10/MA50, RSI, MACD + signal + hist diff, Bollinger bands, ATR, StochRSI), lags, volume change, optional TA‑Lib (Momentum, SMA20). A simple VIX feature is supported if `^VIX` is included and present.
- Target: `Target = close(t+1) > close(t)` for binary classification.
- Model: XGBoost classifier tuned via `RandomizedSearchCV` over a small grid using `TimeSeriesSplit` and `precision` scoring.
- Sentiment: News headlines via NewsAPI are classified by xAI Grok (OpenAI SDK with `base_url=https://api.x.ai/v1`) into Positive/Neutral/Negative; per‑ticker net score is mapped into a `Sentiment_Score` feature.
- Hybrid signals: ML probabilities are thresholded, then adjusted by rules (skip if high VIX) and refined by a small Q‑Learning policy that learns from recent action outcomes.
- Risk & execution: Simple sizing using ATR, stop‑loss/take‑profit gates, basic transaction cost and slippage model in backtest, market orders in live/paper.
- Scheduling: `schedule` runs a job at start and every 30 minutes within configured market hours.


## Disclaimers

This software is for educational and research purposes. Markets involve substantial risk. Past performance does not guarantee future results. Use paper trading and validate thoroughly before considering any real capital.


## Nice‑to‑have next steps

- Add a `requirements.txt`/lock file and container image for fully reproducible runs
- Improve VIX/volatility regime handling and macro features
- Expand unit/integration tests around data, features, and signal gen
- Add position sizing via Kelly/vol targeting and portfolio constraints
- Switch secrets to environment variables and add a `.env.example` template