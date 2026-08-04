# Setup Dependencies for Full GUI Functionality

## Current Status

Your GUI is working in **simplified mode** with mock data. To enable full trading functionality, you need to install the bot dependencies.

---

## Quick Fix: Install Dependencies

The original trading bot (`Trader_main_Grok4_20250731.py`) has these dependencies installed somewhere. We just need to make them available to the GUI.

### Option 1: Use the Same Python Environment (Recommended)

Find which Python the original bot uses:

```bash
# Check if you have multiple Python installations
which -a python python3

# Try to find where dependencies are installed
python3 -c "import sys; print(sys.path)"
```

Then update the shebang in `trader_gui.py` to use that Python.

### Option 2: Install Dependencies for Current Python

```bash
pip3 install --break-system-packages \
    pytz \
    schedule \
    pandas \
    numpy \
    alpaca-trade-api \
    requests \
    ta \
    xgboost \
    scikit-learn \
    pyyaml \
    aiohttp
```

### Option 3: Create a Virtual Environment (Clean Approach)

```bash
# Create venv
python3 -m venv trader_venv

# Activate it
source trader_venv/bin/activate

# Install dependencies
pip install PyQt6 pytz schedule pandas numpy alpaca-trade-api \
    requests ta xgboost scikit-learn pyyaml aiohttp joblib seaborn matplotlib

# Run GUI from venv
python trader_gui.py
```

---

## What's Different Between Modes?

### Simplified Mode (Current)
- ✅ GUI works
- ✅ Menu bar icon
- ✅ Dashboard displays
- ⚠️ Shows mock data
- ❌ No real trading
- ❌ No live position updates
- ❌ No signal generation

### Full Mode (After Installing Dependencies)
- ✅ Everything from simplified mode
- ✅ Real trading logic
- ✅ Live position updates from Alpaca
- ✅ Signal generation from ML model
- ✅ Manual trades execution
- ✅ Scheduled trading at 3:55 PM

---

## Testing Which Mode You're In

```bash
# Test dependencies
python3 -c "import pytz, schedule, pandas, alpaca_trade_api; print('✅ Full mode available')" 2>&1
```

If you see `✅ Full mode available`, restart the GUI and it will use full mode.

If you see `ModuleNotFoundError`, you're in simplified mode.

---

## Recommended: Find Existing Dependencies

Your original bot works, so the dependencies exist somewhere. Let's find them:

```bash
# Find all Python installations
find /usr -name "python*" -type f 2>/dev/null | head -10
find /opt -name "python*" -type f 2>/dev/null | head -10
find ~/Library -name "python*" -type f 2>/dev/null | head -10

# Check if original bot has a shebang
head -5 Trader_main_Grok4_20250731.py

# Try running original bot to see which Python it uses
python Trader_main_Grok4_20250731.py --version 2>&1 | head -5
```

---

## Quick Test

After installing dependencies, test the full bot service:

```bash
# Remove old socket
rm /tmp/trader_bot.sock

# Start full bot service
python3 bot_service.py &

# Test it
sleep 3
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"

# Stop it
pkill -f bot_service.py
```

If this works without errors, your GUI will now use full mode!

---

## Alternative: Keep Using Simplified Mode

If you just want to **test the GUI interface** without actual trading:

1. Simplified mode is perfect for this
2. You can see the UI, test buttons, view layouts
3. When ready for real trading, install dependencies later

---

## Summary

**Right Now**: GUI works in simplified mode (mock data)

**To Enable Full Trading**:
1. Install dependencies (Option 2 above)
2. Restart GUI: `./START_GUI.sh`
3. Full bot service will load automatically

**Status Check**:
```bash
python3 trader_gui.py 2>&1 | grep -i "service"
# Look for "Full bot service" vs "simplified bot service"
```

---

## Need Help?

The GUI will automatically detect which mode to use. If dependencies are missing, it uses simplified mode. Install dependencies anytime and restart to upgrade to full mode.

