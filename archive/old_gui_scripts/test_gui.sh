#!/bin/bash

echo "Cleaning up..."
rm -f /tmp/trader_bot.sock bot_service_subprocess.log trader_gui.log logs/bot_service_subprocess.log logs/trader_gui.log

echo "Starting GUI..."
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
  echo "⚠️  Warning: .venv not found; using system python3"
fi

$PYTHON trader_gui.py > gui_test.log 2>&1 &
GUI_PID=$!

echo "Waiting for bot service to start..."
sleep 6

echo ""
echo "=== Bot Service Log ==="
if [ -f logs/bot_service_subprocess.log ]; then
    cat logs/bot_service_subprocess.log
else
    echo "No bot service log found"
fi

echo ""
echo "=== Socket Status ==="
ls -la /tmp/trader_bot.sock 2>&1

echo ""
echo "=== Testing IPC Connection ==="
$PYTHON -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))" 2>&1

echo ""
echo "=== GUI Process ==="
ps aux | grep $GUI_PID | grep -v grep

echo ""
echo "✅ If you see the menu bar icon, the GUI is working!"
echo "   Look for a colored circle in the top-right of your screen"
echo ""
echo "To stop: kill $GUI_PID"
