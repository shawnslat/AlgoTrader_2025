#!/bin/bash
# Install LaunchAgent for auto-start at login

PLIST_FILE="com.trader2025.bot.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DEST="$LAUNCH_AGENTS_DIR/$PLIST_FILE"

echo "Installing Trading Bot LaunchAgent..."

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Copy plist file
cp "$PLIST_FILE" "$PLIST_DEST"
echo "✓ Copied plist to $PLIST_DEST"

# Unload existing agent if running
launchctl unload "$PLIST_DEST" 2>/dev/null

# Load the agent
launchctl load "$PLIST_DEST"
echo "✓ LaunchAgent loaded"

echo ""
echo "Installation complete!"
echo "The Trading Bot GUI will now start automatically at login."
echo ""
echo "To disable auto-start, run:"
echo "  launchctl unload $PLIST_DEST"
echo ""
echo "To re-enable auto-start, run:"
echo "  launchctl load $PLIST_DEST"
