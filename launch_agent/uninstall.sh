#!/bin/bash
# Uninstall LaunchAgent

PLIST_FILE="com.trader2025.bot.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS_DIR/$PLIST_FILE"

echo "Uninstalling Trading Bot LaunchAgent..."

# Unload the agent
launchctl unload "$PLIST_DEST" 2>/dev/null
echo "✓ LaunchAgent unloaded"

# Remove plist file
rm -f "$PLIST_DEST"
echo "✓ Removed plist file"

echo ""
echo "Uninstallation complete!"
echo "The Trading Bot GUI will no longer start at login."
