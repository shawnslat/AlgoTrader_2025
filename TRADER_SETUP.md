# AlgoTrader 2025 — Complete Setup Guide

Everything you need to run the full stack: the **stock/crypto trading bot**, the
**Streamlit dashboard** (portfolio, positions, charts, AI chat, manual trades,
backtests, income tracker), and the optional **SEER prediction-market scanner**
([github.com/shawnslat/SEER](https://github.com/shawnslat/SEER)).

> **Everything runs in PAPER mode by default. Keep it that way until the system
> has proven itself to you.** Live-switching instructions are at the bottom,
> with the warnings they deserve.

---

## 1. Accounts & API keys you need

Create your **own** accounts — never share keys between people (two bots on one
brokerage account will fight over the same positions).

| Service | What it's for | Cost | Required? |
|---|---|---|---|
| [Alpaca](https://alpaca.markets) | Brokerage — paper trading account + market data for crypto | Free (paper) | **Yes** |
| [Polygon.io](https://polygon.io) | Stock price history/bars | Free tier OK | **Yes** |
| [Anthropic](https://console.anthropic.com) | Claude: news sentiment, nightly journal, daily watchlist rotation, dashboard AI chat | Paid (~$ a few/month at Haiku pricing) | **Yes** (core AI) |
| [NewsAPI](https://newsapi.org) | News headlines feeding sentiment | Free tier OK | Yes |
| [Finnhub](https://finnhub.io) | Insider (Form 4) trade data | Free tier OK | Yes |
| [xAI / Grok](https://console.x.ai) | Second-opinion bias vote + sentiment fallback | Paid | Optional (bot degrades gracefully without it) |

For SEER (prediction markets), see [its own setup guide](https://github.com/shawnslat/SEER) —
paper mode needs **no accounts at all**; live needs Kalshi and/or Polymarket US.

## 2. Install (macOS)

```bash
# TA-Lib C library (required by the ta-lib python package)
brew install ta-lib

git clone https://github.com/shawnslat/AlgoTrader_V1.git
cd AlgoTrader_V1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Linux/Windows: everything except the menu-bar icon and `.command` launcher
works; install ta-lib via your package manager and run the components manually
(step 4).

## 3. Configure

```bash
cp config.example.yaml config.yaml
# open config.yaml and paste YOUR keys into the placeholder fields
```

Key settings worth knowing (all in `config.yaml`, hot-reloaded every cycle):

- `alpaca.base_url: https://paper-api.alpaca.markets` ← **this is what makes it paper**
- `intraday:` — cycle every 30 min, 09:45–15:15 ET; `schedule.no_new_entries_after: '15:30'`
- `confidence.min_confidence_to_trade: 0.65` — the trade floor
- `watchlist_rotation.daily_premarket: true` — AI picks tickers each morning at 09:00
- `risk_per_trade_pct`, `stop_loss_pct`, `max_position_pct`, `crypto.allocation_cap_pct` — risk knobs
- `drawdown_kill_switch` — halts new entries at −12% from peak equity

## 4. Run

```bash
# Option A (macOS): one double-click
open "Trading Bot.command"     # bot + dashboard + health dot (+ SEER if installed)

# Option B: components manually
python bot_service.py                                     # the trading engine
.venv/bin/streamlit run dashboard.py --server.port 8502   # dashboard → http://localhost:8502
python tray_health.py                                     # macOS menu-bar dot (optional)
```

**Menu-bar dot:** green = running · blue = trading cycle in progress · red =
error/stopped · gray = service not started.

## 5. The dashboard (localhost:8502)

| Tab | What it does |
|---|---|
| Portfolio | Equity curve, daily P&L, recent matched trades (FIFO) |
| Positions / Orders | Live holdings with P&L, allocation, order history, TradingView charts |
| Charts / Market | Per-ticker charts, market overview |
| SEER | Prediction-market scanner status (if SEER is installed) |
| Scaled Exits | Tier state for partial profit-taking |
| AI Chat | Ask Claude about your portfolio (read-only advisor) |
| Backtest | Run/inspect backtests (read the README's Honest Limitations first) |
| Logs | Live service logs |
| Income Portfolio | Passive-income tracker — fill `income_portfolio/Income_Portfolio_Tracker.xlsx` to activate |
| Manual Trade | Hand-place a market order through the bot service (confirm-guarded) |
| Trading Settings | Risk parameters |

The red **Flatten All** control cancels every order, sells every position, and
halts the bot — the big red button.

## 6. What runs automatically (weekdays)

| Time (ET) | Job |
|---|---|
| 09:00 | Claude rewrites the watchlist (SPY/QQQ locked, held positions protected, >50% churn needs your approval) |
| 09:45–15:15 | Trading cycle every 30 min |
| 15:30 | Hard cutoff — no new entries |
| 16:15 | FIFO trade analysis refresh |
| 18:30 | Claude writes the nightly journal (`logs/journal/`) |
| Weekly | XGBoost model retrain |

## 7. Switching to LIVE money (read this twice)

**Don't, until:** you've run 3–6 months of paper, the live equity curve tracks
or beats SPY, and you've read `README.md → Honest Limitations` and
`vault/DECISIONS.md`. The lifetime stats in this repo's history include months
of known-buggy behavior — they are a "before" baseline, not a track record.

When you truly want live:

1. In Alpaca, generate **live** API keys (real account, real money).
2. In `config.yaml`: swap in the live keys and set
   `alpaca.base_url: https://api.alpaca.markets`
3. Start small: cut `buying_power_pct`, `max_position_pct`, and
   `risk_per_trade_pct` well below the paper defaults.
4. Watch the first week daily. The kill switch and stops are rails, not guarantees.

Nothing else changes — same code path, which is the point of paper-first.

## 8. Troubleshooting

`TROUBLESHOOTING.md` covers common failures. Quick hits: "no module named
yfinance/talib" → activate the venv & reinstall requirements; dashboard shows
no bot → is `bot_service.py` running (check the menu-bar dot)?; zero trades →
normal on low-confidence days, check the Logs tab for "Confidence too low"
lines; crypto symbol errors → use `BTC/USD` form, never `BTCUSD`.
