#!/usr/bin/env python3
"""
Proper GUI launcher for macOS
Ensures PyQt6 runs in correct application context
"""
import sys
import os

# Important: Set Qt environment variables before importing PyQt6
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# Check bot service BEFORE importing PyQt6
from ipc_protocol import IPCClient
client = IPCClient()

if not client.is_running():
    print("❌ ERROR: Bot service is not running!")
    print("")
    print("Start it first in another terminal:")
    print("  ./1_start_bot_service.sh")
    print("")
    sys.exit(1)

print("✅ Bot service detected")
print("✅ Starting GUI...")
print("👀 Look for menu bar icon in top-right of screen!")
print("")

# Now import PyQt6 and create proper application
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer, Qt

app = QApplication(sys.argv)
app.setApplicationName("Trading Bot")
app.setQuitOnLastWindowClosed(False)

# Create tray icon with color
def create_icon(color_name):
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    colors = {
        "green": QColor(76, 175, 80),
        "yellow": QColor(255, 193, 7),
        "red": QColor(244, 67, 54),
        "gray": QColor(158, 158, 158),
    }

    painter.setBrush(colors.get(color_name, colors["gray"]))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()

    return QIcon(pixmap)

# Check if system tray is available
if not QSystemTrayIcon.isSystemTrayAvailable():
    print("❌ ERROR: System tray not available on this system")
    sys.exit(1)

# Create tray icon
tray = QSystemTrayIcon()
tray.setIcon(create_icon("yellow"))
tray.setToolTip("Trading Bot - Loading...")

# Create menu
menu = QMenu()

status_action = QAction("🟡 Bot Status: Loading...")
status_action.setEnabled(False)
menu.addAction(status_action)

heartbeat_action = QAction("⏱ Last heartbeat: —")
heartbeat_action.setEnabled(False)
menu.addAction(heartbeat_action)

menu.addSeparator()

start_action = QAction("▶️  Start Bot")
stop_action = QAction("⏸️  Stop Bot")
stop_action.setEnabled(False)

menu.addAction(start_action)
menu.addAction(stop_action)

run_now_action = QAction("⚡ Run Now")
run_now_action.setEnabled(False)
menu.addAction(run_now_action)

menu.addSeparator()

dashboard_action = QAction("📊 Dashboard")
logs_action = QAction("📄 Live Logs")
settings_action = QAction("⚙️  Settings")
backtest_action = QAction("🧪 Train/Backtest")
chat_action = QAction("💬 AI Chat")

menu.addAction(dashboard_action)
menu.addAction(logs_action)
menu.addAction(settings_action)
menu.addAction(backtest_action)
menu.addAction(chat_action)

menu.addSeparator()

quit_action = QAction("❌ Quit")
quit_action.triggered.connect(app.quit)
menu.addAction(quit_action)

tray.setContextMenu(menu)
tray.show()

# Status update function
def update_status():
    try:
        response = client.send_command({"command": "get_status"})

        # IPC errors are returned as {"error": "..."}; normal status includes error=None
        if response.get("error"):
            status_action.setText("🔴 Bot Status: Error")
            tray.setIcon(create_icon("red"))
            tray.setToolTip(f"Trading Bot - Error: {response['error']}")
            start_action.setEnabled(False)
            stop_action.setEnabled(False)
            run_now_action.setEnabled(False)
        else:
            status = response.get("status", "unknown")
            running = response.get("running", False)
            next_exec = response.get("next_execution", "N/A")
            hb = response.get("last_heartbeat")
            heartbeat_action.setText(f"⏱ Last heartbeat: {hb or '—'}")

            if status == "running":
                status_action.setText(f"🟢 Bot Status: Running (Next: {next_exec})")
                tray.setIcon(create_icon("green"))
                tray.setToolTip(f"Trading Bot - Running (Next: {next_exec})")
                start_action.setEnabled(False)
                stop_action.setEnabled(True)
                run_now_action.setEnabled(True)
            elif status == "trading":
                status_action.setText("🔵 Bot Status: Trading")
                tray.setIcon(create_icon("blue"))
                tray.setToolTip("Trading Bot - Trading")
                start_action.setEnabled(False)
                stop_action.setEnabled(True)
                run_now_action.setEnabled(True)
            else:
                status_action.setText("🟡 Bot Status: Idle")
                tray.setIcon(create_icon("yellow"))
                tray.setToolTip("Trading Bot - Idle")
                start_action.setEnabled(True)
                stop_action.setEnabled(False)
                run_now_action.setEnabled(False)
    except Exception as e:
        print(f"Error updating status: {e}")

# Start/stop handlers
def start_bot():
    response = client.send_command({"command": "start"})
    if response.get("success"):
        tray.showMessage("Trading Bot", "Bot started successfully", QSystemTrayIcon.MessageIcon.Information, 3000)
    update_status()

def stop_bot():
    response = client.send_command({"command": "stop"})
    if response.get("success"):
        tray.showMessage("Trading Bot", "Bot stopped successfully", QSystemTrayIcon.MessageIcon.Information, 3000)
    update_status()

def run_now():
    response = client.send_command({"command": "run_now"})
    if response.get("success"):
        tray.showMessage("Trading Bot", "Run-now triggered", QSystemTrayIcon.MessageIcon.Information, 3000)
    else:
        msg = response.get("message") or response.get("error") or "Unknown error"
        tray.showMessage("Trading Bot", f"Run-now failed: {msg}", QSystemTrayIcon.MessageIcon.Critical, 5000)
    update_status()

# Dashboard handler
def show_dashboard():
    try:
        from gui.dashboard_window import DashboardWindow
        global dashboard_window
        if 'dashboard_window' not in globals() or dashboard_window is None:
            dashboard_window = DashboardWindow(client)
        dashboard_window.show()
        dashboard_window.raise_()
        dashboard_window.activateWindow()
    except Exception as e:
        print(f"Error opening dashboard: {e}")
        import traceback
        traceback.print_exc()

# Logs handler
def show_logs():
    try:
        from gui.logs_window import LogsWindow
        global logs_window
        if 'logs_window' not in globals() or logs_window is None:
            logs_window = LogsWindow()
        logs_window.show()
        logs_window.raise_()
        logs_window.activateWindow()
    except Exception as e:
        print(f"Error opening logs: {e}")
        import traceback
        traceback.print_exc()

# Settings handler
def show_settings():
    try:
        from gui.settings_window import SettingsWindow
        global settings_window
        if 'settings_window' not in globals() or settings_window is None:
            settings_window = SettingsWindow()
        settings_window.show()
        settings_window.raise_()
        settings_window.activateWindow()
    except Exception as e:
        print(f"Error opening settings: {e}")
        import traceback
        traceback.print_exc()

# Backtest handler
def show_backtest():
    try:
        from gui.backtest_window import BacktestWindow
        global backtest_window
        if 'backtest_window' not in globals() or backtest_window is None:
            backtest_window = BacktestWindow(client)
        backtest_window.show()
        backtest_window.raise_()
        backtest_window.activateWindow()
    except Exception as e:
        print(f"Error opening backtest window: {e}")
        import traceback
        traceback.print_exc()

# Chat handler
def show_chat():
    try:
        from gui.chat_window import ChatWindow
        global chat_window
        if 'chat_window' not in globals() or chat_window is None:
            chat_window = ChatWindow(client)
        chat_window.show()
        chat_window.raise_()
        chat_window.activateWindow()
    except Exception as e:
        print(f"Error opening chat window: {e}")
        import traceback
        traceback.print_exc()

# Connect actions
start_action.triggered.connect(start_bot)
stop_action.triggered.connect(stop_bot)
run_now_action.triggered.connect(run_now)
dashboard_action.triggered.connect(show_dashboard)
logs_action.triggered.connect(show_logs)
settings_action.triggered.connect(show_settings)
backtest_action.triggered.connect(show_backtest)
chat_action.triggered.connect(show_chat)

# Setup status timer
status_timer = QTimer()
status_timer.timeout.connect(update_status)
status_timer.start(5000)  # Update every 5 seconds

# Initial update
update_status()

print("✅ Menu bar icon should now be visible!")
print("")

# Run app
sys.exit(app.exec())
