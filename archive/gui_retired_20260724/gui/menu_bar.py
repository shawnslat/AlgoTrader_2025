"""
Menu Bar Application - Lightweight tray icon for trading bot control
"""
import os
import sys
import socket
import subprocess
import webbrowser
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer, Qt
from ipc_protocol import IPCClient
import logging

logger = logging.getLogger(__name__)

DASHBOARD_PORT = 8502


class TradingBotMenuBar:
    """Lightweight menu bar icon with bot controls and dashboard launcher."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.ipc_client = IPCClient()
        self.tray_icon = None
        self.menu = None
        self.status_timer = QTimer()
        self.bot_status = "stopped"

        self.setup_ui()

    def setup_ui(self):
        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(self._create_status_icon("gray"))
        self.tray_icon.setToolTip("Trading Bot - Stopped")

        self.create_menu()
        self.tray_icon.show()

        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)
        self.update_status()

    def create_menu(self):
        self.menu = QMenu()

        # Status (non-clickable)
        self.status_action = QAction("Bot Status: Stopped")
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)

        self.heartbeat_action = QAction("Last heartbeat: --")
        self.heartbeat_action.setEnabled(False)
        self.menu.addAction(self.heartbeat_action)

        self.menu.addSeparator()

        # Bot controls
        self.start_action = QAction("Start Bot")
        self.start_action.triggered.connect(self.start_bot)
        self.menu.addAction(self.start_action)

        self.stop_action = QAction("Stop Bot")
        self.stop_action.triggered.connect(self.stop_bot)
        self.stop_action.setEnabled(False)
        self.menu.addAction(self.stop_action)

        self.restart_action = QAction("Restart Bot")
        self.restart_action.triggered.connect(self.restart_bot)
        self.menu.addAction(self.restart_action)

        self.menu.addSeparator()

        # Dashboard
        self.open_dashboard_action = QAction("Open Dashboard")
        self.open_dashboard_action.triggered.connect(self.open_streamlit_dashboard)
        self.menu.addAction(self.open_dashboard_action)

        self.menu.addSeparator()

        # Quit
        self.quit_action = QAction("Quit")
        self.quit_action.triggered.connect(self.quit_application)
        self.menu.addAction(self.quit_action)

        self.tray_icon.setContextMenu(self.menu)

    def _create_status_icon(self, color: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = {
            "green": QColor(76, 175, 80), "yellow": QColor(255, 193, 7),
            "red": QColor(244, 67, 54), "gray": QColor(158, 158, 158),
            "blue": QColor(33, 150, 243),
        }
        painter.setBrush(colors.get(color, colors["gray"]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()
        return QIcon(pixmap)

    def update_status(self):
        if not self.ipc_client.is_running():
            self.bot_status = "stopped"
            self.status_action.setText("Bot Service: Not Running")
            self.heartbeat_action.setText("Last heartbeat: --")
            self.tray_icon.setIcon(self._create_status_icon("red"))
            self.tray_icon.setToolTip("Trading Bot - Not Running")
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.restart_action.setEnabled(False)
            return

        response = self.ipc_client.send_command({"command": "get_status"})
        if response.get("error"):
            self.bot_status = "error"
            self.status_action.setText("Bot Status: Error")
            self.tray_icon.setIcon(self._create_status_icon("red"))
            return

        status = response.get("status", "unknown")
        self.bot_status = status
        running = response.get("running", False)
        next_exec = response.get("next_execution")
        last_heartbeat = response.get("last_heartbeat")

        self.heartbeat_action.setText(f"Last heartbeat: {last_heartbeat or '--'}")

        if status == "trading":
            self.status_action.setText("Bot Status: Trading")
            self.tray_icon.setIcon(self._create_status_icon("blue"))
            self.tray_icon.setToolTip("Trading Bot - Executing Trades")
        elif status == "running":
            self.status_action.setText(f"Bot Status: Running (Next: {next_exec or 'N/A'})")
            self.tray_icon.setIcon(self._create_status_icon("green"))
            self.tray_icon.setToolTip(f"Trading Bot - Running")
        elif status == "idle":
            self.status_action.setText("Bot Status: Idle")
            self.tray_icon.setIcon(self._create_status_icon("yellow"))
            self.tray_icon.setToolTip("Trading Bot - Idle")
        elif status == "error":
            self.status_action.setText("Bot Status: Error")
            self.tray_icon.setIcon(self._create_status_icon("red"))
            self.tray_icon.setToolTip("Trading Bot - Error")
        else:
            self.status_action.setText("Bot Status: Stopped")
            self.tray_icon.setIcon(self._create_status_icon("gray"))
            self.tray_icon.setToolTip("Trading Bot - Stopped")

        self.start_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        self.restart_action.setEnabled(True)

    def start_bot(self):
        response = self.ipc_client.send_command({"command": "start"})
        msg = "Bot started" if response.get("success") else response.get("message", "Failed")
        self.tray_icon.showMessage("Trading Bot", msg, QSystemTrayIcon.MessageIcon.Information, 3000)
        self.update_status()

    def stop_bot(self):
        response = self.ipc_client.send_command({"command": "stop"})
        msg = "Bot stopped" if response.get("success") else response.get("message", "Failed")
        self.tray_icon.showMessage("Trading Bot", msg, QSystemTrayIcon.MessageIcon.Information, 3000)
        self.update_status()

    def restart_bot(self):
        self.ipc_client.send_command({"command": "stop"})
        time.sleep(2)
        self.ipc_client.send_command({"command": "start"})
        self.tray_icon.showMessage("Trading Bot", "Bot restarted", QSystemTrayIcon.MessageIcon.Information, 3000)
        self.update_status()

    def open_streamlit_dashboard(self):
        # Check if already running
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(("localhost", DASHBOARD_PORT))
            s.close()
            webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
            return
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass

        # Start Streamlit
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.py")
        subprocess.Popen(
            ["streamlit", "run", dashboard_path, "--server.port", str(DASHBOARD_PORT), "--server.headless", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        time.sleep(3)
        webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")

    def quit_application(self):
        from PyQt6.QtWidgets import QMessageBox
        if self.bot_status in ["running", "trading"]:
            reply = QMessageBox.question(
                None, "Quit", "Bot is running. Stop and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_bot()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    app = TradingBotMenuBar()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
