#!/usr/bin/env python3
"""
Trading Bot GUI Launcher
Starts the bot service and GUI menu bar application
"""
import sys
import subprocess
import time
import logging
from pathlib import Path
from ipc_protocol import IPCClient
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/trader_gui.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import PyQt6
        logger.info("PyQt6 is installed")
    except ImportError:
        logger.error("PyQt6 is not installed. Run: source .venv/bin/activate && pip install PyQt6")
        return False

    return True


def check_bot_dependencies():
    """Check if bot service dependencies are available"""
    try:
        import pytz
        import schedule
        import pandas
        import alpaca_trade_api
        return True
    except ImportError:
        return False


def is_bot_service_running():
    """Check if bot service is already running"""
    client = IPCClient()
    return client.is_running()


def start_bot_service():
    """Start the bot service in background"""
    logger.info("Starting bot service...")

    # Prefer full bot_service.py (real Alpaca + model) if available, otherwise fall back to simple wrapper mode.
    bot_script = "bot_service.py"
    if not Path("artifacts/final_model.pkl").exists():
        bot_script = "bot_service_simple.py"
        logger.info("artifacts/final_model.pkl not found; using simplified bot service (mock data mode)")
    else:
        logger.info("Using full bot service for GUI (real mode)")

    # Start bot service as a subprocess (redirect output to file)
    with open('logs/bot_service_subprocess.log', 'w') as log_file:
        process = subprocess.Popen(
            [sys.executable, bot_script],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
        )

    logger.info(f"Started {bot_script} with PID {process.pid}")
    logger.info("Bot service logs: logs/bot_service_subprocess.log")

    # Wait for service to start
    max_wait = 10
    for i in range(max_wait):
        time.sleep(1)
        if is_bot_service_running():
            # Basic sanity check: service should respond to get_status
            try:
                resp = IPCClient().send_command({"command": "get_status"})
                if resp.get("error"):
                    raise RuntimeError(resp.get("error"))
                logger.info("Bot service started successfully")
                return True
            except Exception as e:
                logger.error(f"Bot service started but is unhealthy: {e}")
                break

    logger.error("Bot service failed to start within timeout")
    # Check the log
    try:
        with open('logs/bot_service_subprocess.log', 'r') as f:
            logger.error(f"Bot service log: {f.read()}")
    except:
        pass
    return False


def start_gui():
    """Start the GUI menu bar application"""
    logger.info("Starting GUI menu bar...")

    from gui.menu_bar import TradingBotMenuBar

    app = TradingBotMenuBar()
    return app.run()


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Trading Bot GUI Launcher")
    logger.info("=" * 60)

    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)

    # Check if bot service is already running
    if is_bot_service_running():
        logger.info("Bot service is already running")
    else:
        # Start bot service
        if not start_bot_service():
            logger.error("Failed to start bot service")
            logger.error("Check bot_service_subprocess.log for details")
            sys.exit(1)

    # Start GUI
    try:
        exit_code = start_gui()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running GUI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
