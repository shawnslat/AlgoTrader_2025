#!/bin/bash
# Step 1: Start Bot Service (keep this terminal open)

echo "╔════════════════════════════════════════╗"
echo "║  Trading Bot Service                   ║"
echo "║  Keep this terminal window open!      ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Starting bot service..."
echo ""

# Clean up old socket
rm -f /tmp/trader_bot.sock

# Start bot service
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
  echo "⚠️  Warning: .venv not found; using system python3"
fi

mkdir -p logs
$PYTHON bot_service.py

# If it exits, show error
echo ""
echo "⚠️  Bot service stopped!"
echo "Check the error messages above."
