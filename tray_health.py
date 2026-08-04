#!/usr/bin/env python3
"""
Menu-bar health dot for AlgoTrader (slim replacement for the retired PyQt GUI tray).

Colors:
  GREEN = bot service running (idle between cycles)
  BLUE  = trading cycle in progress
  RED   = error state or service stopped
  GRAY  = bot service unreachable (not started)

Menu: live status details, Open Dashboard, SEER status, Quit.
Display-only — start/stop/trade actions live in the Streamlit dashboard.
"""
import json
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer

SOCKET_PATH = "/tmp/trader_bot.sock"
DASHBOARD_URL = "http://localhost:8502"
POLL_SECONDS = 10

COLORS = {
    "green": "#00c853",
    "blue": "#2196f3",
    "red": "#ff1744",
    "gray": "#9e9e9e",
}


def ipc_status():
    """Query bot_service status over its Unix socket. Returns dict or None."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(SOCKET_PATH)
        s.sendall(json.dumps({"command": "get_status"}).encode("utf-8") + b"\n")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break
        s.close()
        if not data:
            return None
        return json.loads(data.decode("utf-8").strip())
    except Exception:
        return None


def seer_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"scanner\.py"],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0
    except Exception:
        return False


def make_dot_icon(color_hex: str) -> QIcon:
    pm = QPixmap(22, 22)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color_hex)
    p.setBrush(c)
    p.setPen(c)
    p.drawEllipse(4, 4, 14, 14)
    p.end()
    return QIcon(pm)


class HealthTray:
    def __init__(self, app: QApplication):
        self.app = app
        self.tray = QSystemTrayIcon()
        self.menu = QMenu()

        self.status_action = QAction("Checking bot status...")
        self.status_action.setEnabled(False)
        self.menu.addAction(self.status_action)

        self.next_run_action = QAction("")
        self.next_run_action.setEnabled(False)
        self.menu.addAction(self.next_run_action)

        self.seer_action = QAction("SEER: checking...")
        self.seer_action.setEnabled(False)
        self.menu.addAction(self.seer_action)

        self.menu.addSeparator()

        open_dash = QAction("Open Dashboard")
        open_dash.triggered.connect(lambda: webbrowser.open(DASHBOARD_URL))
        self.menu.addAction(open_dash)
        self._open_dash = open_dash  # keep reference

        self.menu.addSeparator()

        quit_action = QAction("Quit Health Icon")
        quit_action.triggered.connect(self.app.quit)
        self.menu.addAction(quit_action)
        self._quit = quit_action

        self.tray.setContextMenu(self.menu)
        self.tray.setIcon(make_dot_icon(COLORS["gray"]))
        self.tray.setVisible(True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(POLL_SECONDS * 1000)
        self.refresh()

    def refresh(self):
        st = ipc_status()
        if st is None:
            color, text = "gray", "Bot service: unreachable (not started?)"
            next_text = ""
        else:
            status = (st.get("status") or "").lower()
            running = bool(st.get("running"))
            error = st.get("error")
            if status == "trading":
                color, text = "blue", "Bot: trading cycle in progress"
            elif status == "error" or error:
                color, text = "red", f"Bot ERROR: {str(error)[:60]}"
            elif running:
                color, text = "green", "Bot: running (idle)"
            else:
                color, text = "red", "Bot: stopped"
            nxt = st.get("next_execution")
            last = st.get("last_execution")
            next_text = f"Next: {nxt or '—'}   Last: {last or '—'}"

        self.tray.setIcon(make_dot_icon(COLORS[color]))
        self.tray.setToolTip(text)
        self.status_action.setText(text)
        self.next_run_action.setText(next_text)
        self.seer_action.setText(
            "SEER: running" if seer_running() else "SEER: not running"
        )


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    tray = HealthTray(app)  # noqa: F841 — must stay referenced
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
