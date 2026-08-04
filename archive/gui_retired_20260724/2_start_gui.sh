#!/bin/bash
# Step 2: Start GUI (run AFTER bot service is running)

echo "╔════════════════════════════════════════╗"
echo "║  Trading Bot GUI                       ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check if bot service is running
if [ ! -S /tmp/trader_bot.sock ]; then
    echo "❌ ERROR: Bot service is not running!"
    echo ""
    echo "Please run './1_start_bot_service.sh' first in another terminal"
    echo ""
    exit 1
fi

echo "✅ Bot service detected"
echo ""
echo "Starting GUI..."
echo "Look for the menu bar icon in the top-right of your screen!"
echo ""

# Start GUI with proper macOS launcher
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
  echo "⚠️  Warning: .venv not found; using system python3"
fi

mkdir -p logs
$PYTHON launch_gui_proper.py

echo ""
echo "GUI closed."
