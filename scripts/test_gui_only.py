#!/usr/bin/env python3
"""
Test GUI connecting to already-running bot service
"""
import sys
from gui.menu_bar import TradingBotMenuBar

print("=" * 60)
print("Testing GUI (bot service should already be running)")
print("=" * 60)
print("")

# Check if bot service is running
from ipc_protocol import IPCClient
client = IPCClient()

if not client.is_running():
    print("❌ ERROR: Bot service is not running!")
    print("")
    print("Start it first:")
    print("  ./1_start_bot_service.sh")
    print("")
    sys.exit(1)

print("✅ Bot service is running")

# Test connection
result = client.send_command({"command": "get_status"})
print(f"✅ IPC connection works: {result}")
print("")

# Create GUI
print("Creating GUI menu bar...")
app = TradingBotMenuBar()
print("✅ GUI created")
print("")
print("Starting GUI event loop...")
print("👀 Look for the menu bar icon in top-right of screen!")
print("")
print("Press Ctrl+C to exit")
print("=" * 60)
print("")

sys.exit(app.run())
