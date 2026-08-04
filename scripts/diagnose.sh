#!/bin/bash

echo "================================"
echo "Trading Bot GUI Diagnostics"
echo "================================"
echo ""

echo "1. Checking for Python processes..."
ps aux | grep -E "trader_gui|bot_service|menu_bar" | grep -v grep | grep -v diagnose

echo ""
echo "2. Checking socket..."
if [ -S /tmp/trader_bot.sock ]; then
    echo "   ✅ Socket exists: /tmp/trader_bot.sock"
    ls -la /tmp/trader_bot.sock
else
    echo "   ❌ Socket does NOT exist"
fi

echo ""
echo "3. Testing IPC connection..."
python3 -c "from ipc_protocol import IPCClient; import json; print('   Result:', json.dumps(IPCClient().send_command({'command': 'get_status'})))" 2>&1

echo ""
echo "4. Checking logs..."
if [ -f trader_gui.log ]; then
    echo "   Last 5 lines of trader_gui.log:"
    tail -5 trader_gui.log | sed 's/^/   /'
fi

if [ -f bot_service.log ]; then
    echo "   Last 5 lines of bot_service.log:"
    tail -5 bot_service.log | sed 's/^/   /'
fi

echo ""
echo "5. PyQt6 system tray test..."
python3 -c "
from PyQt6.QtWidgets import QSystemTrayIcon
if QSystemTrayIcon.isSystemTrayAvailable():
    print('   ✅ System tray available')
else:
    print('   ❌ System tray NOT available')
" 2>&1

echo ""
echo "================================"
echo "Summary:"
echo "================================"

if [ -S /tmp/trader_bot.sock ]; then
    echo "✅ Bot service appears to be running"
else
    echo "❌ Bot service is NOT running - start with ./1_start_bot_service.sh"
fi

PYPROCS=$(ps aux | grep -E "trader_gui|menu_bar" | grep -v grep | grep -v diagnose | wc -l)
if [ "$PYPROCS" -gt 0 ]; then
    echo "✅ GUI process(es) running: $PYPROCS"
else
    echo "❌ No GUI processes found - start with ./2_start_gui.sh"
fi

echo ""
