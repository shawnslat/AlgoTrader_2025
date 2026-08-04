#!/bin/bash
# Trading Bot Launcher (headless) - Runs bot service + Streamlit dashboard, closes terminal
# PyQt GUI retired 2026-07-24 — everything now lives in the Streamlit dashboard (port 8502)

cd "/Users/shawnslat/Documents/GitHub/Trader_2025"

# Use virtual environment Python/streamlit if available
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    STREAMLIT=".venv/bin/streamlit"
else
    PYTHON="python3"
    STREAMLIT="streamlit"
fi

mkdir -p logs

# Don't double-start: skip bot service if already running
if pgrep -f "bot_service.py" > /dev/null 2>&1; then
    echo "Bot service already running."
else
    rm -f /tmp/trader_bot.sock
    nohup $PYTHON bot_service.py >> logs/bot_launchd.log 2>&1 &
fi

# Don't double-start the dashboard either
if pgrep -f "streamlit run dashboard.py" > /dev/null 2>&1; then
    echo "Dashboard already running."
else
    nohup $STREAMLIT run dashboard.py --server.port 8502 --server.headless true > /dev/null 2>&1 &
fi

# Start SEER (Kalshi/Polymarket prediction-market bot) alongside the trader
SEER_DIR="/Users/shawnslat/Documents/Programming/kalshi_v2"
if [ -d "$SEER_DIR" ]; then
    if pgrep -f "scanner\.py" > /dev/null 2>&1; then
        echo "SEER already running."
    else
        # Launch scanner.py DIRECTLY under nohup — seer.py spawns it as a child,
        # which dies on terminal close (child doesn't inherit nohup immunity)
        (cd "$SEER_DIR" && nohup python3 scanner.py >> seer_launch.log 2>&1 &)
        echo "SEER started (scanner, headless)."
    fi
fi

# Menu-bar health dot (green = running, blue = trading, red = error, gray = off)
if pgrep -f "tray_health.py" > /dev/null 2>&1; then
    echo "Health icon already running."
else
    nohup $PYTHON tray_health.py > /dev/null 2>&1 &
fi

# Wait a moment for the server to start, then open in browser
sleep 3
open "http://localhost:8502"

# Close this terminal window
osascript -e 'tell application "Terminal" to close front window' &
exit 0
