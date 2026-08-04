# Trader_2025 Directory Structure

Clean, organized directory structure as of December 17, 2025.

## 📁 Root Directory (Production Files Only)

```
Trader_2025/
├── 1_start_bot_service.sh          # Start trading bot service
├── 2_start_gui.sh                  # Start GUI (menu bar app)
├── README.md                       # Main project documentation
├── Trader_main_Grok4_20250731.py   # Core trading bot (1,350 lines)
├── bot_service.py                  # Bot service wrapper for GUI (966 lines)
├── config.yaml                     # Configuration (API keys, tickers, risk params)
├── dexter_gate.py                  # Dexter AI veto gate
├── ipc_protocol.py                 # Inter-process communication for GUI
├── launch_gui_proper.py            # GUI launcher (macOS-native)
├── pdt_guard.py                    # Pattern Day Trader compliance
├── valid_tickers.yaml              # Valid ticker symbols
├── tickers_auto.json               # Auto-generated tickers from Dexter
└── dexter_bias.json                # Dexter trading bias data
```

**Essential Files:**
- ✅ `Trader_main_Grok4_20250731.py` - Main trading logic
- ✅ `bot_service.py` - Service wrapper for GUI
- ✅ `config.yaml` - All configuration
- ✅ `1_start_bot_service.sh` & `2_start_gui.sh` - Startup scripts

---

## 📂 Subdirectories

### `/data` - Historical Market Data
```
data/
├── AAPL.csv                        # Apple stock history
├── MSFT.csv                        # Microsoft stock history
├── NVDA.csv                        # NVIDIA stock history
├── TSLA.csv                        # Tesla stock history
├── AMZN.csv                        # Amazon stock history
└── ...                             # Other configured tickers
```

**Format:** CSV files with columns: `time, open, high, low, close, volume`
**Source:** Polygon.io API
**Coverage:** ~2 years of daily bars per ticker

---

### `/artifacts` - Model & Analysis Outputs
```
artifacts/
├── final_model.pkl                 # Current production model (XGBoost)
├── improved_model_v2.pkl           # Improved model (55.4% accuracy)
├── q_table.csv                     # Q-Learning state-action values
├── last_decisions.csv              # Most recent trading decisions
├── backtesting_results.csv         # Backtest equity curve
├── backtesting_results.png         # Backtest visualization
├── confusion_matrix.png            # Model evaluation
├── classification_report.txt       # Model metrics
├── improved_model_v2_confusion.png # Improved model confusion matrix
├── model_improvement_visualization.png  # Performance comparison charts
└── model_analysis/                 # Model analysis reports
    ├── feature_importance.png
    ├── data_quality_report.txt
    └── recommendations.csv
```

**Key Files:**
- ✅ `final_model.pkl` - Active trading model
- ✅ `improved_model_v2.pkl` - Better model ready to deploy
- ✅ `q_table.csv` - Reinforcement learning values

---

### `/logs` - Runtime Logs
```
logs/
├── master_trading_bot.log          # Main bot execution log
├── bot_service.log                 # Service wrapper log (278 KB)
├── trader_gui.log                  # GUI application log
└── trade_logs/                     # Trade execution history
    ├── trade_log_2025-12-16_15-57-26.csv
    ├── trade_log_2025-12-15_15-57-32.csv
    └── ...                         # Daily trade logs
```

**Purpose:**
- Debug issues
- Monitor trading activity
- Audit trade history

**Rotation:** Logs grow indefinitely, manually archive old logs

---

### `/gui` - GUI Components
```
gui/
├── menu_bar.py                     # macOS menu bar application
├── dashboard_window.py             # Portfolio dashboard
├── logs_window.py                  # Log viewer
├── settings_window.py              # Config editor
├── backtest_window.py              # Backtest/training interface
├── chat_window.py                  # Dexter AI chat
├── manual_trade_dialog.py          # Manual trade placement
└── widgets/                        # Reusable UI components
```

**Entry Point:** `launch_gui_proper.py` (not `trader_gui.py`)

---

### `/dexter` - AI Assistant Integration
```
dexter/
├── dexter-chat.ts                  # Chat with Dexter AI
├── dexter-tickers.ts               # Auto-suggest tickers
├── dexter-bias.ts                  # Generate trade bias
├── package.json                    # Bun dependencies
└── ...                             # TypeScript utilities
```

**Runtime:** Bun (JavaScript runtime)
**Output:** `tickers_auto.json`, `dexter_bias.json`

---

### `/docs` - Documentation (16 files)
```
docs/
├── START_HERE.md                   # Quick start guide ⭐
├── MODEL_IMPROVEMENT_REPORT.md     # Model accuracy improvement (Dec 2025) ⭐
├── SENTIMENT_FIX_SUMMARY.md        # Sentiment fallback fix (Dec 2025) ⭐
├── NO_TRADES_ANALYSIS.md           # Root cause analysis (Dec 17, 2025)
├── QUICK_START_IMPROVED_MODEL.md   # Deploy improved model guide
├── GUI_README.md                   # GUI user guide
├── LAUNCH_INSTRUCTIONS.md          # Startup procedures
├── SETUP_DEPENDENCIES.md           # Environment setup
├── TROUBLESHOOTING.md              # Common issues
├── ANALYSIS_REPORT.md              # Architecture analysis
├── BUGFIX_TALIB.md                 # TA-Lib installation
├── DAILY_CLOSE_EXECUTION_CHANGES.md # Execution timing changes
├── GUI_IMPLEMENTATION_SUMMARY.md   # GUI technical details
├── SIMPLE_SOLUTION.md              # Simplified deployment
├── TEST_GUI_NOW.md                 # GUI testing guide
└── QUICK_START_GUIDE.md            # Alternative quick start
```

**Start Reading:**
1. `START_HERE.md` - Project overview
2. `MODEL_IMPROVEMENT_REPORT.md` - Recent improvements
3. `GUI_README.md` - How to use the GUI

---

### `/archive` - Development Artifacts
```
archive/
├── README.md                       # Archive guide
├── analysis/                       # Model improvement scripts
│   ├── improved_model_v2.py        # Training script (55.4% accuracy)
│   ├── model_analysis_and_improvement.py
│   ├── quick_model_diagnosis.py
│   └── visualize_improvements.py
├── test_scripts/                   # One-time utilities
│   ├── fix_sentiment_fallback.py
│   ├── test_sentiment_fix.py
│   └── update_tickers_from_json.py
└── old_gui_scripts/                # Deprecated GUI code
    ├── trader_gui.py               # Old launcher (don't use)
    ├── bot_service_simple.py       # Mock service (don't use)
    └── run_gui.sh                  # Old script (don't use)
```

**Purpose:** Historical reference, not needed for production

---

### `/scripts` - Utility Scripts
```
scripts/
├── setup_venv.sh                   # Create Python virtual environment
├── install_dependencies.sh         # Install Python packages
└── ...                             # Other helper scripts
```

---

### `/launch_agent` - macOS LaunchAgent
```
launch_agent/
├── com.trader.bot.plist            # macOS LaunchAgent config
└── install.sh                      # Install as system service
```

**Purpose:** Auto-start bot on macOS login (optional)

---

### `/backups` - Configuration Backups
```
backups/
└── config_backups/                 # Timestamped config.yaml backups
```

---

## 🎯 File Count Summary

| Category | Count | Purpose |
|----------|-------|---------|
| **Root Python files** | 5 | Core trading logic |
| **Root shell scripts** | 2 | Startup scripts |
| **Root config files** | 3 | Configuration |
| **GUI components** | 8 | User interface |
| **Documentation** | 16 | Guides and reports |
| **Data files** | 12 | Historical market data |
| **Archived files** | 13 | Development artifacts |

**Total active files:** ~60
**Total archived files:** 13

---

## 🚀 Quick Navigation

### Want to...
- **Start the bot:** Run `./1_start_bot_service.sh`
- **Open GUI:** Run `./2_start_gui.sh`
- **Change settings:** Edit `config.yaml`
- **View logs:** Check `logs/bot_service.log`
- **See trades:** Check `logs/trade_logs/`
- **Read docs:** Start with `docs/START_HERE.md`
- **Improve model:** See `docs/MODEL_IMPROVEMENT_REPORT.md`
- **Troubleshoot:** See `docs/TROUBLESHOOTING.md`

### Where is...
- **Trading logic:** `Trader_main_Grok4_20250731.py`
- **API keys:** `config.yaml` (lines 1-24)
- **Tickers list:** `config.yaml` (lines 28-34)
- **Model file:** `artifacts/final_model.pkl`
- **Better model:** `artifacts/improved_model_v2.pkl`
- **Trade history:** `logs/trade_logs/`

---

## 📊 Disk Usage

Approximate sizes:
- `/data` - 300 KB (12 tickers × 2 years)
- `/artifacts` - 10 MB (models, charts, analysis)
- `/logs` - 1-5 MB (grows over time)
- `/dexter` - 500 KB (Node modules not counted)
- `/docs` - 200 KB
- `/archive` - 100 KB (code only)
- **Total:** ~15-20 MB

---

## 🧹 Maintenance

### Weekly
- Check `logs/` size, archive old logs if > 100 MB
- Review `logs/trade_logs/` for trading activity

### Monthly
- Backup `config.yaml` to `backups/`
- Review `artifacts/backtesting_results.csv` for performance
- Consider retraining model with fresh data

### As Needed
- Update `data/*.csv` for more historical data
- Archive old logs to external storage
- Clean up `archive/` if not needed

---

**Last Updated:** December 17, 2025
**Purpose:** Directory reference for Trader_2025 project
