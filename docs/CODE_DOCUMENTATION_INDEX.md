# Trading Bot Code Documentation Index

**Project:** Trader_2025
**Purpose:** Comprehensive code documentation for all core components
**Date Generated:** 2025-12-18

---

## Overview

This directory contains detailed code documentation for the entire trading bot system. Each file has been documented with function-by-function explanations, programming notes, and examples.

---

## Quick Navigation

### Core Trading System
1. **[Trader_main_Grok4_20250731.py](CODE_DOCUMENTATION_Trader_main.md)** - Main trading bot (1,354 lines)
   - ML model training (XGBoost)
   - Q-Learning reinforcement
   - Signal generation
   - Live trading execution
   - Backtesting engine

2. **[bot_service.py](CODE_DOCUMENTATION_bot_service.md)** - Bot wrapper service (966 lines)
   - IPC server for GUI communication
   - Job management (backtest, signals, Dexter)
   - Trading cycle orchestration
   - State machine (stopped → idle → running → trading)

### Communication & GUI
3. **[ipc_protocol.py](CODE_DOCUMENTATION_ipc_protocol.md)** - IPC system (201 lines)
   - Unix domain socket server/client
   - Log streaming
   - Command protocol

4. **[launch_gui_proper.py](CODE_DOCUMENTATION_launch_gui_proper.md)** - macOS GUI launcher (280 lines)
   - System tray application
   - Menu bar icon
   - Status monitoring
   - Window management

### Risk Management
5. **[pdt_guard.py](CODE_DOCUMENTATION_pdt_guard.md)** - PDT protection (58 lines)
   - Day trade tracking
   - 5-day rolling window
   - Round-trip monitoring

6. **[dexter_gate.py](CODE_DOCUMENTATION_dexter_gate.md)** - AI sentiment gate (42 lines)
   - Dexter bias reader
   - Trade veto logic
   - Fail-safe design

---

## Documentation Files

| File | Lines | Purpose | Key Features |
|------|-------|---------|--------------|
| **[CODE_DOCUMENTATION_Trader_main.md](CODE_DOCUMENTATION_Trader_main.md)** | 1,354 | Core trading algorithm | XGBoost ML, Q-Learning RL, 25+ indicators, Alpaca execution |
| **[CODE_DOCUMENTATION_bot_service.md](CODE_DOCUMENTATION_bot_service.md)** | 966 | Service wrapper | IPC server, background jobs, state machine, scheduling |
| **[CODE_DOCUMENTATION_ipc_protocol.md](CODE_DOCUMENTATION_ipc_protocol.md)** | 201 | Inter-process communication | Socket protocol, log streaming, thread-safe |
| **[CODE_DOCUMENTATION_launch_gui_proper.md](CODE_DOCUMENTATION_launch_gui_proper.md)** | 280 | macOS GUI | PyQt6, system tray, menu actions, window lifecycle |
| **[CODE_DOCUMENTATION_pdt_guard.md](CODE_DOCUMENTATION_pdt_guard.md)** | 58 | Pattern day trader guard | Day trade tracking, SEC compliance, round-trip detection |
| **[CODE_DOCUMENTATION_dexter_gate.md](CODE_DOCUMENTATION_dexter_gate.md)** | 42 | AI sentiment veto | Dexter bias integration, risk filtering, fail-safe mode |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                                                               │
│  launch_gui_proper.py (PyQt6 System Tray)                   │
│  - Start/Stop Bot                                            │
│  - Dashboard, Logs, Settings, Backtest, Chat                │
└────────────┬─────────────────────────────────────────────────┘
             │ IPC Commands
             ↓
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│                                                               │
│  bot_service.py (Service Wrapper)                            │
│  ├─ IPC Server (ipc_protocol.py)                            │
│  ├─ Job Management (backtest, signals, Dexter)              │
│  ├─ Trading Cycle Orchestration                              │
│  └─ State Machine (stopped → idle → running → trading)      │
└────────────┬─────────────────────────────────────────────────┘
             │ Function Calls
             ↓
┌─────────────────────────────────────────────────────────────┐
│                     Trading Algorithm                         │
│                                                               │
│  Trader_main_Grok4_20250731.py (Core Logic)                 │
│  ├─ Data Fetching (Polygon, Alpaca)                         │
│  ├─ Feature Engineering (25+ indicators)                     │
│  ├─ ML Training (XGBoost)                                   │
│  ├─ Q-Learning (Reinforcement)                              │
│  ├─ Signal Generation (Hybrid ML+RL)                        │
│  └─ Live Execution (Alpaca API)                             │
└────────────┬─────────────────────────────────────────────────┘
             │
             ├────────────┬────────────┐
             ↓            ↓            ↓
    ┌─────────────┐ ┌─────────┐ ┌────────────┐
    │ pdt_guard   │ │ dexter  │ │ External   │
    │             │ │ _gate   │ │ APIs       │
    │ PDT         │ │         │ │            │
    │ Protection  │ │ AI Gate │ │ • Polygon  │
    │             │ │         │ │ • Alpaca   │
    │ Max 2 day   │ │ Veto    │ │ • NewsAPI  │
    │ trades/day  │ │ Risky   │ │ • xAI Grok │
    └─────────────┘ └─────────┘ └────────────┘
```

---

## System Components

### 1. Data Layer
**Files:** Trader_main_Grok4_20250731.py (data fetching functions)
- **Polygon.io:** Historical market data (OHLCV)
- **Alpaca:** Real-time data and broker API
- **NewsAPI:** Sentiment data source
- **xAI Grok:** LLM for sentiment analysis

### 2. Feature Engineering
**Files:** Trader_main_Grok4_20250731.py (feature engineering)
- **Technical Indicators:** RSI, MACD, Bollinger Bands, ATR, Stochastic RSI, MA10/50
- **Sentiment Scores:** Grok-based news sentiment (-1 to +1)
- **Lag Features:** Previous close prices
- **Volume Analysis:** Volume change ratios
- **VIX:** Market volatility indicator

### 3. Machine Learning
**Files:** Trader_main_Grok4_20250731.py (ML functions)
- **Model:** XGBoost binary classifier
- **Objective:** Predict next-day price direction (up/down)
- **Training:** RandomizedSearchCV with TimeSeriesSplit
- **Features:** 25+ technical + sentiment indicators
- **Evaluation:** Precision, recall, F1-score, confusion matrix

### 4. Reinforcement Learning
**Files:** Trader_main_Grok4_20250731.py (Q-Learning functions)
- **Algorithm:** Q-Learning with epsilon-greedy policy
- **State Space:** 162 discrete states (sentiment × RSI × MACD × volume × position)
- **Actions:** Buy, Sell, Hold
- **Learning:** Bellman equation updates after each trade
- **Exploration:** 5% random actions for discovery

### 5. Signal Generation
**Files:** Trader_main_Grok4_20250731.py (hybrid signals)
- **Hybrid Approach:** XGBoost predictions + Q-Learning refinement
- **ML Component:** Price direction probability
- **RL Component:** Action selection based on learned policy
- **Filters:** VIX threshold, PDT guard, Dexter gate

### 6. Risk Management
**Files:** pdt_guard.py, dexter_gate.py
- **PDT Guard:** Prevents exceeding day trade limits (3 per 5 days for <$25k accounts)
- **Dexter Gate:** AI veto for risky trades based on sentiment analysis
- **Position Sizing:** ATR-based volatility adjustment
- **Stop Loss:** Automatic exit at 5% loss
- **Take Profit:** Automatic exit at 10% gain

### 7. Execution
**Files:** Trader_main_Grok4_20250731.py (trading functions)
- **Broker:** Alpaca Markets (paper and live trading)
- **Order Type:** Market orders (immediate execution)
- **Timing:** 3:55 PM ET (5 min before close)
- **Monitoring:** Order status polling until filled/rejected
- **Logging:** Complete trade logs to CSV

### 8. Service Management
**Files:** bot_service.py
- **Mode:** Background daemon service
- **IPC:** Unix domain socket for GUI communication
- **Jobs:** Backtest, signal generation, Dexter bias
- **Scheduling:** Daily execution at 3:55 PM ET
- **State:** Stopped → Idle → Running → Trading

### 9. User Interface
**Files:** launch_gui_proper.py
- **Framework:** PyQt6 native macOS application
- **Display:** System tray icon with context menu
- **Features:** Start/stop bot, dashboard, logs, settings, backtest, chat
- **Updates:** Real-time status polling (5-second intervals)
- **Windows:** Dashboard, logs, settings, backtest, Dexter chat

---

## Key Workflows

### Training Workflow
```
1. Download historical data (Polygon.io)
   ↓
2. Engineer features (25+ indicators)
   ↓
3. Add sentiment (Grok + NewsAPI)
   ↓
4. Split data (80/20 train/test)
   ↓
5. Train XGBoost model (RandomizedSearchCV)
   ↓
6. Evaluate model (confusion matrix, metrics)
   ↓
7. Backtest strategy (simulate trades)
   ↓
8. Save artifacts (model.pkl, q_table.csv)
```

### Live Trading Workflow
```
1. Load trained model + Q-table
   ↓
2. Schedule daily job (3:55 PM ET)
   ↓
3. Fetch latest data (Alpaca + Polygon)
   ↓
4. Engineer features
   ↓
5. Add sentiment
   ↓
6. Generate signals (ML + RL)
   ↓
7. Apply filters (PDT guard, Dexter gate, VIX)
   ↓
8. Calculate position sizes (ATR-based)
   ↓
9. Place orders (Alpaca API)
   ↓
10. Monitor fills
   ↓
11. Update Q-table (learn from outcomes)
   ↓
12. Log trades
```

### GUI Interaction Workflow
```
User clicks "Start Bot" in menu
   ↓
GUI sends IPC command {"command": "start"}
   ↓
bot_service.py receives command
   ↓
Changes state: idle → running
   ↓
Schedules trading job (3:55 PM)
   ↓
Sends IPC response {"success": true}
   ↓
GUI polls status every 5 seconds
   ↓
Displays "🟢 Bot Status: Running"
```

---

## File Dependencies

### Trader_main_Grok4_20250731.py Dependencies
```python
# Core
import pandas as pd
import numpy as np
import yaml
import json

# ML/RL
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Trading
import alpaca_trade_api as tradeapi

# Technical Analysis
import ta
import talib  # Optional

# APIs
import requests  # Polygon, NewsAPI
from openai import OpenAI  # xAI Grok

# Scheduling
import schedule

# Risk Management
from pdt_guard import DayTradeGuard
from dexter_gate import DexterGate
```

### bot_service.py Dependencies
```python
# Core
import yaml
import json
import logging

# IPC
from ipc_protocol import IPCServer, LogStreamer

# Trading
from Trader_main_Grok4_20250731 import (
    load_configuration,
    initialize_alpaca_api,
    fetch_current_market_data,
    engineer_features,
    add_sentiment_features,
    generate_signals,
    execute_trading_logic_live,
    backtest_strategy,
    # ... and many more functions
)

# Risk Management
from pdt_guard import DayTradeGuard
from dexter_gate import DexterGate

# Scheduling
import schedule

# Concurrency
import threading
import subprocess
```

### launch_gui_proper.py Dependencies
```python
# GUI Framework
from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu
)
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer, Qt

# IPC
from ipc_protocol import IPCClient

# Windows
from gui.dashboard_window import DashboardWindow
from gui.logs_window import LogsWindow
from gui.settings_window import SettingsWindow
from gui.backtest_window import BacktestWindow
from gui.chat_window import ChatWindow
```

---

## Configuration Files

### config.yaml
```yaml
# Broker API
alpaca:
  api_key: "YOUR_ALPACA_KEY"
  api_secret: "YOUR_ALPACA_SECRET"
  base_url: "https://paper-api.alpaca.markets"

# Market Data
polygon:
  api_key: "YOUR_POLYGON_KEY"

# Tickers
tickers:
  - AAPL
  - MSFT
  - GOOGL
  - AMZN
  - TSLA

# Risk Management
max_day_trades: 2
risk_per_trade_pct: 0.01
stop_loss_pct: 0.05
take_profit_pct: 0.10
vix_threshold: 25

# Sentiment Analysis
newsapi_token: "YOUR_NEWSAPI_KEY"
grok_api_key: "YOUR_XAI_KEY"

sentiment:
  enabled: true
  max_items: 30
  llm_timeout_seconds: 20

# Dexter Integration
dexter_autofetch: false
dexter_ticker_command: "bun dexter/main.ts --mode tickers"
dexter_chat_command: "bun dexter/main.ts --mode chat"
dexter_bias_command: "bun dexter/main.ts --mode bias"
```

### tickers_auto.json (Generated by Dexter)
```json
{
  "tickers": ["NVDA", "AMD", "INTC"],
  "source": "dexter_research",
  "timestamp": "2025-12-18T14:30:00Z",
  "reasoning": "High volume tech stocks with strong momentum"
}
```

### dexter_bias.json (Generated by Dexter)
```json
{
  "AAPL": {
    "bias": "positive",
    "score": 0.75,
    "reasoning": "Strong earnings beat, positive analyst upgrades",
    "recommendation": "allow"
  },
  "TSLA": {
    "bias": "negative",
    "score": -0.60,
    "reasoning": "Regulatory concerns, production delays",
    "recommendation": "avoid"
  }
}
```

---

## Artifact Files

### Generated During Training
```
artifacts/
├── final_model.pkl              # Trained XGBoost model
├── q_table.csv                  # Q-Learning state-action values
├── last_decisions.csv           # Previous decisions for Q-updates
├── classification_report.txt    # Model evaluation metrics
├── confusion_matrix.png         # Visual confusion matrix
├── backtesting_results.csv      # Backtest performance data
└── backtesting_results.png      # Equity curve plot
```

### Generated During Live Trading
```
logs/
├── master_trading_bot.log       # Main application log
└── trade_logs/
    └── trade_log_2025-12-18_14-30-00.csv  # Trade execution log
```

---

## API Keys Required

| Service | Purpose | Free Tier | Link |
|---------|---------|-----------|------|
| **Alpaca Markets** | Broker API (paper/live trading) | Yes (paper trading unlimited) | https://alpaca.markets |
| **Polygon.io** | Historical market data | Yes (5 calls/min) | https://polygon.io |
| **NewsAPI** | News articles for sentiment | Yes (100 req/day) | https://newsapi.org |
| **xAI** | Grok LLM for sentiment analysis | Paid (~$5/month) | https://x.ai |

---

## Common Tasks

### Start the Bot
```bash
# Terminal 1: Start bot service
./1_start_bot_service.sh

# Terminal 2: Start GUI
./2_start_gui.sh
```

### Train New Model
```bash
python Trader_main_Grok4_20250731.py
# Answer "no" when asked if trained today
# This will:
# 1. Download data
# 2. Train model
# 3. Run backtest
# 4. Start live trading
```

### Run Backtest Only
```python
from Trader_main_Grok4_20250731 import *

config = load_configuration()
data = load_historical_data("./data", config)
data = engineer_features(data)
data = add_sentiment_features(data, config)

model = joblib.load("artifacts/final_model.pkl")
q_table = load_q_table()

backtest_strategy(model, data, selected_features, config, q_table)
```

### Generate Dexter Tickers
```bash
bun dexter/main.ts --mode tickers --query "Most volatile tech stocks today"
# Generates tickers_auto.json
```

### Generate Dexter Bias
```bash
bun dexter/main.ts --mode bias
# Generates dexter_bias.json for current tickers
```

---

## Troubleshooting

### Bot Won't Start
**Problem:** `can't open file 'bot_service_simple.py'`
**Solution:** Check `1_start_bot_service.sh` line 24 points to `bot_service.py`

### PDT Violations
**Problem:** Exceeding day trade limits
**Solution:** Check `max_day_trades` in config.yaml, verify `pdt_guard.py` is working

### GUI Missing Option
**Problem:** Chat window not in menu
**Solution:** Verify `launch_gui_proper.py` has all menu actions connected

### IPC Connection Failed
**Problem:** GUI can't connect to bot service
**Solution:** Check `/tmp/trader_bot.sock` exists, restart bot service

### Model Loading Error
**Problem:** `final_model.pkl` not found
**Solution:** Train model first by running Trader_main and answering "no" to skip training

---

## Performance Benchmarks

### Training Time (MacBook Pro M1)
- **Data Download:** ~2 minutes (3 tickers, 2 years)
- **Feature Engineering:** ~5 seconds
- **Sentiment Analysis:** ~30 seconds (30 articles)
- **Model Training:** ~45 seconds (RandomizedSearchCV, 100 iterations)
- **Backtesting:** ~10 seconds (2 years of data)
- **Total:** ~4 minutes for complete training pipeline

### Live Trading Performance
- **Data Fetch:** ~3 seconds (Polygon + Alpaca)
- **Feature Engineering:** ~1 second
- **Sentiment Analysis:** ~20 seconds
- **Signal Generation:** <1 second
- **Order Placement:** ~2 seconds (Alpaca API)
- **Total Execution Time:** ~30 seconds (within 3:55-4:00 PM window)

### Model Accuracy (Typical)
- **Precision:** 52-55%
- **Recall:** 50-58%
- **F1-Score:** 51-56%
- **Accuracy:** 52-54%

*Note: Accuracy above 50% in financial markets is considered good. The goal is consistent slight edge over many trades.*

---

## Next Steps

### Improvements to Consider
1. **Add more features:** Order flow, options data, social media sentiment
2. **Ensemble models:** Combine XGBoost with LSTM or Transformer
3. **Multi-timeframe analysis:** Add 15-min, hourly signals
4. **Portfolio optimization:** Kelly criterion for position sizing
5. **Walk-forward testing:** Regular retraining schedule
6. **Feature importance analysis:** Remove low-value features
7. **Deep Q-Learning:** Replace tabular Q-Learning with neural network

### Maintenance Tasks
1. **Weekly:** Retrain model with latest data
2. **Monthly:** Review Q-table learning (check for convergence)
3. **Quarterly:** Backtest on recent data, validate performance
4. **Yearly:** Full system audit, dependency updates

---

## Documentation Standards

Each documentation file follows this structure:
1. **Overview:** Purpose and high-level description
2. **Architecture:** Data flow and component relationships
3. **Function Documentation:** Detailed explanation of each function
   - Purpose
   - Parameters
   - Returns
   - How it works (step-by-step)
   - Programming notes
   - Examples
4. **Usage Examples:** Real-world usage patterns
5. **Error Handling:** Common errors and solutions

---

## Contact & Support

For questions about this documentation:
- Review the individual documentation files for detailed information
- Check function docstrings in the source code
- Refer to external API documentation (Alpaca, Polygon, etc.)

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-18 | 1.0 | Initial comprehensive documentation |

---

## License

This documentation is provided as-is for the Trader_2025 project. Refer to the main project LICENSE file for terms.
