# 🚀 Launch Instructions - Trading Bot GUI

## Quick Start (3 Easy Steps)

### Step 1: Navigate to Directory
```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
```

### Step 2: Launch the GUI
```bash
./START_GUI.sh
```
*Or manually:*
```bash
python trader_gui.py
```

### Step 3: Look for Menu Bar Icon
Look at the **top-right** of your screen for the colored circle icon:
- 🔴 Red = Stopped
- 🟡 Yellow = Idle
- 🟢 Green = Running
- 🔵 Blue = Trading

Click it to access all features!

---

## What Happens When You Launch

```
1. Check dependencies ✓
   - PyQt6 installed?
   - Model file exists?

2. Start bot service ✓
   - Launches bot_service.py in background
   - Opens IPC socket at /tmp/trader_bot.sock
   - Loads model and configuration

3. Launch GUI ✓
   - Menu bar icon appears
   - Connects to bot service
   - Ready for interaction!
```

---

## Menu Bar Features

**Click the icon to see:**

```
🟢 Bot Status: Running (Next: 3:55 PM)
─────────────────────────────────
▶️  Start Bot
⏸️  Stop Bot
─────────────────────────────────
📊 Dashboard
📈 Positions
📋 Today's Signals
📄 Live Logs
⚙️  Settings
─────────────────────────────────
❌ Quit
```

### What Each Option Does:

| Option | Action |
|--------|--------|
| **Start Bot** | Begin trading (auto-executes at 3:55 PM ET) |
| **Stop Bot** | Pause trading |
| **Dashboard** | Opens window with positions, P&L, account info |
| **Positions** | Quick view of current holdings |
| **Today's Signals** | See generated buy/sell signals |
| **Live Logs** | Real-time log viewer with filtering |
| **Settings** | Edit tickers, risk parameters, etc. |
| **Quit** | Exit application (asks if bot should stop) |

---

## Dashboard Window

When you open the dashboard, you'll see:

### 📊 Top Section: Account Summary
```
Cash: $100,000.00  |  Buying Power: $100,000.00
Portfolio Value: $100,000.00  |  Equity: $100,000.00
```

### 📈 Middle Section: Current Positions
```
┌────────┬─────┬─────────────┬───────────────┬──────────────┬──────────┬─────────┐
│ Ticker │ Qty │ Entry Price │ Current Price │ Market Value │   P&L    │  P&L %  │
├────────┼─────┼─────────────┼───────────────┼──────────────┼──────────┼─────────┤
│ AAPL   │ 10  │   $180.00   │    $185.00    │  $1,850.00   │ +$50.00  │ +2.78%  │
│ MSFT   │ 5   │   $370.00   │    $365.00    │  $1,825.00   │ -$25.00  │ -1.35%  │
└────────┴─────┴─────────────┴───────────────┴──────────────┴──────────┴─────────┘
```
*Colors: Green = profit, Red = loss*

### 📋 Bottom Section: Today's Signals
```
┌────────┬────────┬────────────┬────────────┐
│ Ticker │ Signal │ Confidence │    Time    │
├────────┼────────┼────────────┼────────────┤
│ NVDA   │  BUY   │    0.85    │ 15:55:00   │
│ AAPL   │  SELL  │    0.72    │ 15:55:00   │
└────────┴────────┴────────────┴────────────┘
```

---

## Live Logs Window

Real-time log viewer with:
- **Filter dropdown**: All, INFO, WARNING, ERROR, DEBUG
- **Color coding**: Red (errors), Orange (warnings), Black (info)
- **Auto-scroll**: Stays at bottom as new logs appear
- **Refresh button**: Manually reload logs
- **Clear button**: Clear display (doesn't delete log file)

Example log:
```
2025-12-17 15:55:00 [INFO] Job running at 15:55:00 EDT
2025-12-17 15:55:01 [INFO] Market close window. Proceeding with trading logic...
2025-12-17 15:55:02 [INFO] Fetching current market data with Polygon.io...
2025-12-17 15:55:05 [INFO] Generated 2 signals for trading
2025-12-17 15:55:06 [INFO] BUY 10 shares of NVDA at $520.00
```

---

## Settings Window

Edit configuration without touching YAML files:

### Tickers Section
```
┌─────────────────────────────┐
│ Tickers (one per line):     │
│                             │
│ AAPL                        │
│ MSFT                        │
│ NVDA                        │
│                             │
└─────────────────────────────┘
```

### Risk Parameters
```
Risk Per Trade (%):      [0.20  ]
Max Position (%):        [5.0   ]
Stop Loss (%):           [0.05  ]
Take Profit (%):         [0.10  ]
Buying Power Usage (%):  [50    ]
```

**Remember**: Bot must be restarted for changes to take effect!

---

## Auto-Start Setup (Optional)

Want the GUI to launch automatically at login?

### Install Auto-Start
```bash
cd launch_agent
./install.sh
```

You'll see:
```
Installing Trading Bot LaunchAgent...
✓ Copied plist to ~/Library/LaunchAgents/com.trader2025.bot.plist
✓ LaunchAgent loaded

Installation complete!
The Trading Bot GUI will now start automatically at login.
```

### Remove Auto-Start
```bash
cd launch_agent
./uninstall.sh
```

---

## Trading Schedule

The bot executes **once per day** at:
- ⏰ **3:55 PM ET** (5 minutes before market close)
- **Execution Window**: 3:50 PM - 4:00 PM ET
- **Days**: Monday - Friday (weekdays only)

**Why 3:55 PM?**
- Daily bar is 99% complete (only missing last 5 min)
- Aligns with model training data (daily bars)
- Ensures higher prediction accuracy

---

## Troubleshooting

### Problem: "Bot service not running"

**Cause**: Stale IPC socket file

**Fix**:
```bash
rm /tmp/trader_bot.sock
python trader_gui.py
```

---

### Problem: "Model not found"

**Cause**: `final_model.pkl` doesn't exist

**Fix**:
```bash
python Trader_main_Grok4_20250731.py
# Answer 'no' when prompted to train model
```

---

### Problem: Menu bar icon not visible

**Cause**: Too many menu bar icons (macOS hides overflow)

**Fix**:
1. Close other menu bar apps
2. Increase screen resolution
3. Check System Settings > Control Center > Menu Bar

---

### Problem: PyQt6 not installed

**Error**: `ModuleNotFoundError: No module named 'PyQt6'`

**Fix**:
```bash
pip3 install --break-system-packages PyQt6
```

---

## File Locations

### Logs
- **GUI logs**: `trader_gui.log`
- **Bot logs**: `master_trading_bot.log`
- **LaunchAgent logs**: `launch_agent.log`, `launch_agent.error.log`

### Model & Data
- **Model**: `final_model.pkl`
- **Q-Table**: `q_table.pkl`
- **Config**: `config.yaml`

### IPC
- **Socket**: `/tmp/trader_bot.sock`

---

## Command Reference

### Start GUI
```bash
python trader_gui.py
# or
./START_GUI.sh
```

### Start Bot Service Only (no GUI)
```bash
python bot_service.py
```

### Check Bot Status (CLI)
```bash
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```

### View Live Logs (Terminal)
```bash
tail -f master_trading_bot.log
```

### Kill Bot Service
```bash
rm /tmp/trader_bot.sock
pkill -f bot_service.py
```

---

## First Time Checklist

Before launching for the first time:

- [ ] Model trained (`final_model.pkl` exists)
- [ ] PyQt6 installed (`python3 -c "import PyQt6"`)
- [ ] In correct directory (`/Users/shawnslat/Documents/GitHub/Trader_2025`)
- [ ] Config file exists (`config.yaml`)

If all checked, you're ready to launch! 🚀

---

## What to Expect on First Launch

1. **Startup (5-10 seconds)**
   - Dependency checks run
   - Bot service starts in background
   - IPC socket created
   - Model loaded

2. **Menu bar icon appears**
   - Look for colored circle in top-right
   - Initially 🟡 yellow (idle)

3. **Click icon → Start Bot**
   - Icon turns 🟢 green (running)
   - Status shows "Next: 15:55"

4. **Wait until 3:55 PM**
   - Icon turns 🔵 blue (trading)
   - Trades execute
   - Icon returns to 🟢 green

5. **Check dashboard**
   - View new positions
   - See P&L
   - Review signals

---

## Security Notes

- ✅ Uses Alpaca **paper trading** account (safe, not real money)
- ✅ Manual trades require confirmation dialog
- ✅ Dexter gate rules enforced (blocks "avoid" tickers)
- ✅ All trades logged to `master_trading_bot.log`
- ✅ API keys remain in `config.yaml` (not exposed in GUI)

---

## Questions?

**Read full documentation**:
- [GUI_README.md](GUI_README.md) - Complete user guide
- [GUI_IMPLEMENTATION_SUMMARY.md](GUI_IMPLEMENTATION_SUMMARY.md) - Technical details
- [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Trading bot basics
- [DAILY_CLOSE_EXECUTION_CHANGES.md](DAILY_CLOSE_EXECUTION_CHANGES.md) - Schedule details

---

## 🎉 Ready to Launch!

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
./START_GUI.sh
```

**Enjoy your native macOS trading bot!** 📈🚀

