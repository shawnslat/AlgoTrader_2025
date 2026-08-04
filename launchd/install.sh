#!/bin/bash
# Install / reinstall the AlgoTrader launchd agents.
#
# Usage: ./launchd/install.sh
#
# After running this once, the journal will run weekdays at 18:30 local time
# and the ticker rotator will run Sundays at 18:00 local time. They survive
# reboots and run automatically. To uninstall: ./launchd/install.sh --remove

set -euo pipefail

LA_DIR="$HOME/Library/LaunchAgents"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

JOURNAL_PLIST="com.shawnslat.algotrader.journal.plist"
ROTATOR_PLIST="com.shawnslat.algotrader.rotator.plist"
RETRAIN_PLIST="com.shawnslat.algotrader.retrain.plist"

mkdir -p "$LA_DIR"

unload_if_loaded() {
    local label="$1"
    if launchctl list | grep -q "$label"; then
        echo "Unloading existing $label..."
        launchctl unload "$LA_DIR/${label}.plist" 2>/dev/null || true
    fi
}

if [ "${1:-}" = "--remove" ]; then
    unload_if_loaded "com.shawnslat.algotrader.journal"
    unload_if_loaded "com.shawnslat.algotrader.rotator"
    unload_if_loaded "com.shawnslat.algotrader.retrain"
    rm -f "$LA_DIR/$JOURNAL_PLIST" "$LA_DIR/$ROTATOR_PLIST" "$LA_DIR/$RETRAIN_PLIST"
    echo "Removed."
    exit 0
fi

# Unload first (no-op if not loaded) so we can replace cleanly
unload_if_loaded "com.shawnslat.algotrader.journal"
unload_if_loaded "com.shawnslat.algotrader.rotator"
unload_if_loaded "com.shawnslat.algotrader.retrain"

# Copy and load
cp "$SRC_DIR/$JOURNAL_PLIST" "$LA_DIR/"
cp "$SRC_DIR/$ROTATOR_PLIST" "$LA_DIR/"
cp "$SRC_DIR/$RETRAIN_PLIST" "$LA_DIR/"
launchctl load "$LA_DIR/$JOURNAL_PLIST"
launchctl load "$LA_DIR/$ROTATOR_PLIST"
launchctl load "$LA_DIR/$RETRAIN_PLIST"

echo "Installed:"
launchctl list | grep -E "shawnslat\.algotrader" || echo "  (none found -- something is wrong)"
echo ""
echo "Test the journal manually before relying on the schedule:"
echo "  .venv/bin/python auto_journal.py"
echo ""
echo "Test the rotator manually:"
echo "  .venv/bin/python auto_ticker_rotator.py"
echo ""
echo "Test the retrain manually (takes minutes — runs a full hyperparameter sweep):"
echo "  .venv/bin/python auto_retrain.py"
