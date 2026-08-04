#!/bin/bash
# Proper GUI launcher with process management

echo "🤖 Trading Bot GUI Launcher"
echo "=============================="
echo ""

# Choose python (prefer venv)
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
  echo "⚠️  Warning: .venv not found; using system python3"
fi

# Kill any existing processes
echo "Cleaning up old processes..."
killall -9 Python 2>/dev/null || true
sleep 1
rm -f /tmp/trader_bot.sock bot_service_subprocess.log
rm -f logs/bot_service_subprocess.log

# Start bot service
echo "Starting bot service..."
$PYTHON bot_service_simple.py > logs/bot_service.log 2>&1 &
BOT_PID=$!
echo "Bot service PID: $BOT_PID"

# Wait for socket
echo "Waiting for bot service..."
for i in {1..10}; do
    if [ -S /tmp/trader_bot.sock ]; then
        echo "✅ Bot service ready"
        break
    fi
    sleep 1
done

if [ ! -S /tmp/trader_bot.sock ]; then
    echo "❌ Bot service failed to start"
    cat logs/bot_service.log
    exit 1
fi

# Test IPC
echo "Testing IPC connection..."
$PYTHON -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))" 2>&1 | head -1

# Start GUI
echo ""
echo "Starting GUI..."
echo "Look for the menu bar icon in the top-right of your screen!"
echo ""

$PYTHON trader_gui.py

# GUI exited, clean up
echo ""
echo "GUI exited. Cleaning up..."
kill $BOT_PID 2>/dev/null
rm -f /tmp/trader_bot.sock
