#!/usr/bin/env python3
"""
Simple Bot Service Wrapper
Provides IPC interface without reimplementing trading logic
"""
import sys
import os
import json
import socket
import threading
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_service_simple.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SOCKET_PATH = "/tmp/trader_bot.sock"
os.makedirs("logs", exist_ok=True)


class SimpleBotService:
    """Simple service that provides status and control"""

    def __init__(self):
        self.running = False
        self.status = "idle"
        self.last_heartbeat = None
        self.last_execution = None
        self.trade_count = 0
        self.server_socket = None
        self.accept_thread = None
        self.heartbeat_thread = None

    def start_ipc_server(self):
        """Start the IPC server"""
        # Remove existing socket
        if Path(SOCKET_PATH).exists():
            Path(SOCKET_PATH).unlink()

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(SOCKET_PATH)
        self.server_socket.listen(5)

        self.accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.accept_thread.start()

        logger.info(f"IPC server started on {SOCKET_PATH}")

    def start_heartbeat(self):
        """Update heartbeat timestamp periodically so GUI can confirm liveness."""
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        while True:
            self.last_heartbeat = datetime.now().isoformat(timespec="seconds")
            time.sleep(1)

    def _simulate_run_now(self):
        """Simulate a trading cycle so the GUI can test status transitions."""
        self.status = "trading"
        time.sleep(2)
        self.last_execution = datetime.now().isoformat(timespec="seconds")
        self.trade_count += 1
        self.status = "running" if self.running else "idle"

    def _accept_connections(self):
        """Accept client connections"""
        while True:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                ).start()
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                break

    def _handle_client(self, client_socket):
        """Handle a client request"""
        try:
            # Receive data
            data = client_socket.recv(4096)
            if not data:
                return

            message = json.loads(data.decode('utf-8').strip())
            command = message.get('command')

            logger.debug(f"Received command: {command}")

            # Handle command
            response = self._handle_command(command, message)

            # Send response
            response_data = json.dumps(response).encode('utf-8') + b"\n"
            client_socket.sendall(response_data)

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
            error_response = json.dumps({"error": str(e)}).encode('utf-8') + b"\n"
            try:
                client_socket.sendall(error_response)
            except:
                pass
        finally:
            client_socket.close()

    def _handle_command(self, command: str, message: dict) -> dict:
        """Handle incoming commands"""

        if command == 'start':
            self.running = True
            self.status = "running"
            return {"success": True, "message": "Bot started (simulation mode)"}

        elif command == 'stop':
            self.running = False
            self.status = "stopped"
            return {"success": True, "message": "Bot stopped"}

        elif command == 'run_now':
            if not self.running:
                return {"success": False, "message": "Bot is not running. Click Start Bot first."}
            threading.Thread(target=self._simulate_run_now, daemon=True).start()
            return {"success": True, "message": "Triggered run-now (simulation mode)"}

        elif command == 'get_status':
            return {
                "status": self.status,
                "running": self.running,
                "next_execution": "15:55:00",
                "last_execution": self.last_execution,
                "trade_count": self.trade_count,
                "last_heartbeat": self.last_heartbeat,
                "error": None
            }

        elif command == 'get_positions':
            # Read from log or return empty
            return {"positions": self._get_mock_positions()}

        elif command == 'get_signals':
            return {"signals": []}

        elif command == 'get_account':
            return self._get_mock_account()

        elif command == 'manual_trade':
            return {
                "error": "Manual trading not available in wrapper mode. Use original bot."
            }

        elif command == 'refresh_dexter':
            return {"success": True, "message": "Dexter refresh not available in wrapper mode"}

        else:
            return {"error": f"Unknown command: {command}"}

    def _get_mock_positions(self):
        """Return mock positions for testing"""
        return [
            {
                "ticker": "AAPL",
                "qty": 10,
                "avg_entry_price": 180.00,
                "current_price": 185.00,
                "market_value": 1850.00,
                "unrealized_pl": 50.00,
                "unrealized_plpc": 2.78
            }
        ]

    def _get_mock_account(self):
        """Return mock account info"""
        return {
            "cash": 100000.00,
            "buying_power": 100000.00,
            "portfolio_value": 101850.00,
            "equity": 101850.00,
            "pattern_day_trader": False
        }

    def run(self):
        """Run the service"""
        self.start_ipc_server()
        self.start_heartbeat()

        logger.info("Bot service running (simple wrapper mode)")
        logger.info("Note: This is a simplified version for GUI testing")
        logger.info("For full functionality, ensure all dependencies are installed")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.server_socket:
                self.server_socket.close()
            if Path(SOCKET_PATH).exists():
                Path(SOCKET_PATH).unlink()


if __name__ == "__main__":
    service = SimpleBotService()
    service.run()
