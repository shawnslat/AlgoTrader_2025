"""
Chat Window - Ask questions to Grok or Claude (via bot_service IPC).
"""
import re
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QPushButton, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt

from ipc_protocol import IPCClient

# Import markdown converter from main_window (or define locally)
try:
    from gui.main_window import _md_to_html
except ImportError:
    try:
        from main_window import _md_to_html
    except ImportError:
        def _md_to_html(text):
            """Fallback: basic markdown to HTML."""
            import re as _re
            lines = text.split('\n')
            out = []
            for line in lines:
                s = line.strip()
                if s.startswith('### '):
                    out.append(f'<h4>{s[4:]}</h4>')
                elif s.startswith('## '):
                    out.append(f'<h3>{s[3:]}</h3>')
                elif s.startswith('# '):
                    out.append(f'<h2>{s[2:]}</h2>')
                elif s in ('---', '***', '___'):
                    out.append('<hr>')
                elif _re.match(r'^[-*]\s', s):
                    content = _re.sub(r'^[-*]\s+', '', s)
                    content = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
                    out.append(f'  {content}<br>')
                elif not s:
                    out.append('<br>')
                else:
                    s = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
                    out.append(f'{s}<br>')
            return '\n'.join(out)


class _ChatSignals(QObject):
    finished = pyqtSignal(dict)


class ChatWindow(QMainWindow):
    def __init__(self, ipc_client: IPCClient):
        super().__init__()
        self.ipc_client = ipc_client
        self.signals = _ChatSignals()
        self.signals.finished.connect(self._on_response)
        self.setWindowTitle("Trading Bot - AI Chat")
        self.setGeometry(140, 140, 900, 650)

        self._build_ui()

    def _build_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        top = QHBoxLayout()
        self.status_label = QLabel("Ready")
        top.addWidget(self.status_label)
        top.addStretch()
        self.include_context = QCheckBox("Include account/positions/signals")
        self.include_context.setChecked(True)
        top.addWidget(self.include_context)
        layout.addLayout(top)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(self.chat_history)

        bottom = QHBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(90)
        bottom.addWidget(self.input_box, stretch=1)

        right = QVBoxLayout()
        self.send_btn = QPushButton("Send to Grok")
        self.send_btn.clicked.connect(self.send_grok)
        right.addWidget(self.send_btn)

        self.claude_btn = QPushButton("Send to Claude")
        self.claude_btn.clicked.connect(self.send_claude)
        self.claude_btn.setToolTip("Ask Claude directly with full portfolio context")
        right.addWidget(self.claude_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.chat_history.clear)
        right.addWidget(self.clear_btn)

        right.addStretch()
        bottom.addLayout(right)

        layout.addLayout(bottom)

    def _append(self, role: str, text: str):
        self.chat_history.append(f"<b>{role}:</b> {text}")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def _set_busy(self, busy: bool, status: str):
        self.status_label.setText(status)
        self.send_btn.setEnabled(not busy)
        self.claude_btn.setEnabled(not busy)

    def send_grok(self):
        query = (self.input_box.toPlainText() or "").strip()
        if not query:
            return
        self.input_box.clear()
        self._append("You", query)
        self._set_busy(True, "Asking Grok...")

        def worker():
            try:
                import requests as _req
                import yaml
                import os

                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                api_key = cfg.get('grok_api_key', '')
                if not api_key:
                    self.signals.finished.emit({"error": "No Grok API key in config.yaml"})
                    return
                resp = _req.post(
                    'https://api.x.ai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
                    json={'model': 'grok-4', 'messages': [{'role': 'user', 'content': query}],
                          'temperature': 0.3, 'max_tokens': 1000},
                    timeout=60,
                )
                resp.raise_for_status()
                answer = resp.json()['choices'][0]['message']['content'].strip()
                self.signals.finished.emit({"answer": answer, "_source": "grok"})
            except Exception as e:
                self.signals.finished.emit({"error": str(e)})

        threading.Thread(target=worker, daemon=True).start()

    def send_claude(self):
        query = (self.input_box.toPlainText() or "").strip()
        if not query:
            return
        self.input_box.clear()
        self._append("You", query)
        self._set_busy(True, "Asking Claude...")

        include_ctx = self.include_context.isChecked()

        def worker():
            resp = self.ipc_client.send_command(
                {"command": "claude_chat", "query": query, "include_context": include_ctx},
                timeout=120.0,
            )
            resp["_source"] = "claude"
            self.signals.finished.emit(resp)

        threading.Thread(target=worker, daemon=True).start()

    def _on_response(self, resp: dict):
        if resp.get("error"):
            self._set_busy(False, "Error")
            self._append("Error", str(resp.get("error")))
            return
        self._set_busy(False, "Ready")
        source = resp.get("_source", "grok")
        label = "Claude" if source == "claude" else "Grok"
        answer = str(resp.get("answer", ""))
        formatted = _md_to_html(answer)
        self.chat_history.append(f"<b>{label}:</b><br>{formatted}")
        sb = self.chat_history.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        event.ignore()
        self.hide()
