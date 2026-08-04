# Trading Bot - Native macOS GUI

A native macOS menu bar application for controlling and monitoring your trading bot.

## 🎯 Features

- **Menu Bar Icon**: Live status indicator (green/yellow/red)
- **Quick Controls**: Start/stop bot from menu bar
- **Dashboard**: View positions, P&L, and account summary
- **Live Logs**: Real-time log viewer with filtering
- **Settings**: Edit configuration without touching YAML files
- **Manual Trades**: Execute trades manually with confirmation
- **Auto-Start**: Optional launch at login (systemd-style)

## 📦 Installation

### 1. Install Dependencies

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
source .venv/bin/activate
pip install PyQt6
```

### 2. Ensure Model is Trained

The bot requires `final_model.pkl` to exist. If you haven't trained the model yet:

```bash
python Trader_main_Grok4_20250731.py
# Answer 'no' when prompted to train the model
```

## 🚀 Usage

### Quick Start

Simply run:

```bash
python trader_gui.py
```

This will:
1. Check dependencies
2. Start the bot service in background
3. Launch the menu bar GUI

### Menu Bar Icon Colors

- 🔴 **Red**: Bot stopped or service not running
- 🟡 **Yellow**: Bot idle (waiting for execution time)
- 🟢 **Green**: Bot running (active)
- 🔵 **Blue**: Trading in progress

### Menu Options

**From the menu bar icon, you can:**

- ▶️ **Start Bot** - Begin trading
- ⏸️ **Stop Bot** - Pause trading
- 📊 **Dashboard** - View positions and portfolio
- 📈 **Positions** - Quick view of current positions
- 📋 **Today's Signals** - See generated trading signals
- 📄 **Live Logs** - Real-time log viewer
- ⚙️ **Settings** - Edit configuration
- ❌ **Quit** - Exit application

## 📊 Dashboard Features

The dashboard window shows:

### Account Summary
- Cash available
- Buying power
- Portfolio value
- Total equity

### Positions Table
- Ticker symbols
- Quantity held
- Entry price
- Current price
- Market value
- P&L (profit/loss) with color coding
- P&L percentage

### Signals Table
- Generated trading signals
- Buy/Sell indicators
- Confidence scores
- Timestamps

### Manual Trading
- Execute manual trades with confirmation
- Respects Dexter gate rules
- Real-time order submission

## 📄 Live Logs

The logs window provides:
- Real-time log streaming
- Filter by level (INFO, WARNING, ERROR, DEBUG)
- Color-coded entries
- Auto-scroll to latest
- Search functionality

## ⚙️ Settings

Edit bot configuration without manually editing YAML:
- Trading tickers (add/remove)
- Risk parameters (stop loss, take profit)
- Position sizing
- Buying power allocation

**Note:** Bot must be restarted for changes to take effect.

## 🔄 Auto-Start at Login

To make the bot GUI start automatically when you log in:

```bash
cd launch_agent
./install.sh
```

This installs a macOS LaunchAgent that will:
- Launch the GUI at login
- Keep the bot service running
- Restart on crashes (if enabled)

### Disable Auto-Start

```bash
cd launch_agent
./uninstall.sh
```

Or manually:

```bash
launchctl unload ~/Library/LaunchAgents/com.trader2025.bot.plist
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     trader_gui.py (Launcher)        │
└─────────────┬───────────────────────┘
              │
              ├─────────────────┐
              │                 │
              ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│  bot_service.py  │  │ gui/menu_bar.py  │
│ (Background Bot) │  │  (Menu Bar GUI)  │
└─────────┬────────┘  └────────┬─────────┘
          │                    │
          │  IPC (Unix Socket) │
          └────────────────────┘
                   │
          /tmp/trader_bot.sock
```

### Component Breakdown

1. **trader_gui.py**: Main launcher
   - Checks dependencies
   - Starts bot service
   - Launches GUI

2. **bot_service.py**: Background trading service
   - Runs trading logic
   - Communicates via IPC
   - Non-blocking operations

3. **ipc_protocol.py**: Inter-process communication
   - Unix socket protocol
   - Command/response pattern
   - Thread-safe

4. **gui/**: GUI package
   - `menu_bar.py`: Menu bar icon and popup
   - `dashboard_window.py`: Main dashboard
   - `logs_window.py`: Log viewer
   - `settings_window.py`: Config editor
   - `manual_trade_dialog.py`: Manual trade UI

## 🔧 Troubleshooting

### Bot Service Won't Start

Check if another instance is running:

```bash
# Check for socket file
ls -la /tmp/trader_bot.sock

# If it exists and bot isn't running, remove it:
rm /tmp/trader_bot.sock
```

Then restart:

```bash
python trader_gui.py
```

### Menu Bar Icon Not Showing

macOS sometimes hides menu bar icons when space is limited. Try:
1. Close other menu bar apps
2. Increase screen resolution
3. Check System Settings > Control Center

### Model Not Found Error

Train the model first:

```bash
python Trader_main_Grok4_20250731.py
# Answer 'no' to the training prompt
```

### Logs Not Updating

The log viewer reads from `master_trading_bot.log`. Ensure the bot has write permissions:

```bash
ls -la master_trading_bot.log
```

## 📝 Files Created

```
Trader_2025/
├── trader_gui.py                    # Main launcher
├── bot_service.py                   # Background bot service
├── ipc_protocol.py                  # IPC communication
├── gui/                             # GUI package
│   ├── __init__.py
│   ├── menu_bar.py                 # Menu bar icon
│   ├── dashboard_window.py         # Dashboard UI
│   ├── logs_window.py              # Log viewer
│   ├── settings_window.py          # Settings editor
│   └── manual_trade_dialog.py      # Manual trade dialog
└── launch_agent/                    # Auto-start scripts
    ├── com.trader2025.bot.plist    # LaunchAgent config
    ├── install.sh                  # Install auto-start
    └── uninstall.sh                # Remove auto-start
```

## 🔐 Security Notes

- Bot uses Alpaca **paper trading** account (safe)
- Manual trades require confirmation dialog
- Dexter gate rules still apply
- All API keys remain in `config.yaml`

## 🚦 Trading Schedule

The bot executes at **3:55 PM ET** daily (market close strategy):
- Execution window: 3:50 PM - 4:00 PM ET
- Weekdays only (Mon-Fri)
- Skips market holidays

## 📞 Support

For issues or questions:
1. Check `trader_gui.log` for GUI errors
   - In venv mode, GUI logs are written to `logs/trader_gui.log`
2. Check `logs/master_trading_bot.log` for bot errors
3. Review `QUICK_START_GUIDE.md` for trading details
4. See `DAILY_CLOSE_EXECUTION_CHANGES.md` for schedule info

## ⚡ Quick Commands

```bash
# Start GUI
python trader_gui.py

# Start bot service only (no GUI)
python bot_service.py

# View live logs
tail -f logs/master_trading_bot.log

# Check bot status via IPC
python -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"

# Install auto-start
cd launch_agent && ./install.sh

# Remove auto-start
cd launch_agent && ./uninstall.sh
```

## 🎉 Enjoy Your Native Trading Bot GUI!

You can now:
- ✅ Run the bot without keeping VS Code open
- ✅ Monitor your portfolio from the menu bar
- ✅ Control trading with a native macOS app
- ✅ Auto-start at login (optional)
- ✅ View live logs and signals
- ✅ Execute manual trades safely

**No more terminal windows!** 🚀
