# Simple Solution - Run Components Separately

Since the integrated launcher is having issues, let's run the bot service and GUI separately.

## Step-by-Step Instructions

### Terminal 1: Start Bot Service (Leave Running)

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
python3 bot_service_simple.py
```

**You should see:**
```
2025-12-17 08:00:14,274 [INFO] IPC server started on /tmp/trader_bot.sock
2025-12-17 08:00:14,274 [INFO] Bot service running (simple wrapper mode)
2025-12-17 08:00:14,274 [INFO] Note: This is a simplified version for GUI testing
2025-12-17 08:00:14,274 [INFO] For full functionality, ensure all dependencies are installed
```

**Leave this terminal running!** Don't close it.

---

### Terminal 2: Start GUI

Open a **new terminal** and run:

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
python3 -c "
from gui.menu_bar import TradingBotMenuBar
import sys

app = TradingBotMenuBar()
sys.exit(app.run())
"
```

**You should see:**
- Menu bar icon appears (should be 🟡 yellow or 🟢 green now, not red)
- Click it to see the menu
- All options should work

---

## What Each Terminal Does

**Terminal 1 (Bot Service):**
- Runs the backend
- Handles data requests
- Must stay open while using GUI

**Terminal 2 (GUI):**
- Displays menu bar icon
- Shows windows and dashboards
- Connects to Terminal 1

---

## To Stop

**Stop GUI:**
- Click menu bar icon → "Quit"
- Or press Ctrl+C in Terminal 2

**Stop Bot Service:**
- Press Ctrl+C in Terminal 1

---

## If Icon is Still Red

The bot service isn't running or can't be reached.

**Check:**
```bash
# In Terminal 3, check if service is running
ls -la /tmp/trader_bot.sock

# If it exists, test connection
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```

**If no socket:**
- Go back to Terminal 1
- Make sure bot_service_simple.py is actually running
- Check for error messages

---

## Why This Works Better

Running components separately means:
- ✅ You can see bot service logs in real-time
- ✅ You can see GUI logs in real-time
- ✅ If one crashes, the other keeps running
- ✅ Easier to debug issues

Once this works, we can fix the integrated launcher.

---

## Quick Test

**Terminal 1:**
```bash
python3 bot_service_simple.py
```

Wait for "IPC server started" message.

**Terminal 2:**
```bash
python3 -c "from gui.menu_bar import TradingBotMenuBar; import sys; app = TradingBotMenuBar(); sys.exit(app.run())"
```

**Look for menu bar icon!** Should be 🟡 yellow or 🟢 green.

