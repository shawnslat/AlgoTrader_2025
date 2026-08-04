# Troubleshooting Guide

## Current Situation

You're seeing a **red icon** in the menu bar but can't interact with the GUI properly.

---

## Solution: Use the New Launcher

```bash
./run_gui.sh
```

This script:
1. Cleans up any old processes
2. Starts bot service properly
3. Waits for it to be ready
4. Launches GUI

---

##  What Each Icon Color Means

| Color | Meaning | What's Happening |
|-------|---------|------------------|
| 🔴 Red | Error or Stopped | Bot service not running or connection failed |
| 🟡 Yellow | Idle | Bot ready but not actively trading |
| 🟢 Green | Running | Bot active, scheduled for 3:55 PM |
| 🔵 Blue | Trading | Currently executing trades |

---

## If Icon is Red and Unresponsive

**This means the GUI can't connect to the bot service.**

### Quick Fix:
```bash
# Stop everything
killall -9 Python

# Remove socket
rm /tmp/trader_bot.sock

# Restart properly
./run_gui.sh
```

---

## If You See No Icon At All

### Possible Causes:
1. **Too many menu bar icons** - macOS hides overflow
2. **System tray disabled** - Check System Settings
3. **Process crashed** - Check logs

### Solutions:
```bash
# Check if process is running
ps aux | grep trader_gui

# Check logs
cat trader_gui.log
cat bot_service.log

# Try simple test
python3 test_menubar_simple.py
```

---

## If Icon Shows But Menu Won't Open

### macOS Permission Issue

The GUI might need accessibility permissions.

**Fix:**
1. Open System Settings
2. Go to Privacy & Security → Accessibility
3. Add Terminal or Python to allowed apps

---

## Common Errors and Fixes

### Error: "Bot service not running"

**Symptom:** Red icon, no interaction

**Fix:**
```bash
# Check if service is actually running
ps aux | grep bot_service

# If not running, start it
python3 bot_service_simple.py &

# Wait 2 seconds
sleep 2

# Test connection
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```

### Error: "Connection refused"

**Symptom:** Socket exists but can't connect

**Fix:**
```bash
# Remove stale socket
rm /tmp/trader_bot.sock

# Kill all processes
killall Python

# Restart
./run_gui.sh
```

### Error: Windows don't open

**Symptom:** Icon works but clicking Dashboard/Logs does nothing

**Fix:**
This is likely a PyQt6 display issue.

```bash
# Test if PyQt6 windows work at all
python3 -c "
from PyQt6.QtWidgets import QApplication, QMessageBox
import sys
app = QApplication(sys.argv)
QMessageBox.information(None, 'Test', 'If you see this, PyQt6 windows work!')
"
```

---

## Checking System Status

### Is bot service running?
```bash
ls -la /tmp/trader_bot.sock
```
- If exists: Bot service is running
- If not: Bot service needs to start

### Is GUI process running?
```bash
ps aux | grep trader_gui | grep -v grep
```
- If shows PID: GUI is running
- If empty: GUI crashed or not started

### Can IPC connect?
```bash
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```
- If returns `{status...}`: Working!
- If returns `{error...}`: Connection problem

---

## Nuclear Option: Complete Reset

If nothing works:

```bash
# 1. Kill everything
killall -9 Python
pkill -9 -f trader_gui
pkill -9 -f bot_service

# 2. Clean up
rm -f /tmp/trader_bot.sock
rm -f bot_service_subprocess.log
rm -f trader_gui.log
rm -f bot_service.log

# 3. Start fresh
./run_gui.sh
```

---

## Alternative: Run Components Separately

If the integrated launcher doesn't work, run components manually:

### Terminal 1: Bot Service
```bash
python3 bot_service_simple.py
```
Leave this running.

### Terminal 2: GUI
```bash
python3 trader_gui.py
```

This way you can see errors in real-time.

---

## Logs to Check

| Log File | Contains |
|----------|----------|
| `trader_gui.log` | GUI launcher logs |
| `bot_service.log` | Bot service logs |
| `bot_service_subprocess.log` | Bot service started by GUI |
| `master_trading_bot.log` | Original bot logs |

```bash
# View all logs
tail -20 trader_gui.log bot_service.log bot_service_subprocess.log
```

---

## Known Limitations

### Simplified Mode (Current)
- ✅ GUI interface works
- ✅ Menu bar icon visible
- ✅ Windows open and close
- ⚠️ Shows mock data
- ❌ No real trading

### To Enable Full Mode
See [SETUP_DEPENDENCIES.md](SETUP_DEPENDENCIES.md)

---

## Getting Help

If still not working, collect this info:

```bash
# System info
sw_vers
python3 --version

# Process status
ps aux | grep -E "trader_gui|bot_service"

# Socket status
ls -la /tmp/trader_bot.sock

# IPC test
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"

# Recent logs
tail -50 trader_gui.log bot_service.log
```

---

## Quick Reference

**Start GUI:**
```bash
./run_gui.sh
```

**Stop GUI:**
```bash
killall Python
rm /tmp/trader_bot.sock
```

**Check Status:**
```bash
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```

**View Logs:**
```bash
tail -f trader_gui.log
```

