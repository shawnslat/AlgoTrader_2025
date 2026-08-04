"""
Trading Bot Service - Non-blocking background service for GUI integration
Wraps the core trading logic from Trader_main_Grok4_20250731.py
"""
import os
import signal
import sys
import threading
import time
import logging
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import shlex
import pytz
import schedule

# Logging (write to logs/bot_service.log for GUI troubleshooting)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import core bot components
from Trader_main_Grok4_20250731 import (
    load_configuration,
    initialize_alpaca_api,
    fetch_current_market_data_with_crypto,
    fetch_latest_data,
    engineer_features,
    generate_signals,
    execute_trading_logic_live,
    fetch_current_positions,
    load_q_table,
    download_historical_data_with_crypto,
    load_historical_data,
    add_sentiment_features,
    prepare_train_test_data,
    tune_and_train_model,
    evaluate_model,
    backtest_strategy,
    maybe_override_tickers_from_json,
    maybe_add_crypto_tickers,
    is_alpaca_auth_error,
    is_transient_alpaca_error,
)
from pdt_guard import DayTradeGuard
from ipc_protocol import IPCServer, LogStreamer
from position_monitor import (
    check_and_enforce_stops,
    check_drawdown_kill_switch,
    is_kill_switch_tripped,
    clear_kill_switch,
)
import joblib

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


class TradingBotService:
    """Background service that runs the trading bot and responds to GUI commands"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = None
        self.api = None
        self.model = None
        self.q_table = None
        self.selected_features = []
        self.guard = None

        # Runtime state
        self.running = False
        self.bot_thread = None
        self.raw_historical = None
        self.last_execution_time = None
        self.next_execution_time = None
        self.trade_count = 0
        self.current_signals = []
        self.last_signals_refresh = None
        self.signals_refresh_running = False
        self.signals_refresh_error = None
        self.backtest_running = False
        self.backtest_last_run = None
        self.backtest_last_error = None
        self.backtest_phase = "idle"
        self.backtest_started_at = None
        self.backtest_last_update = None
        self.backtest_progress = None
        self._backtest_thread = None
        self.status = "stopped"  # stopped, idle, running, trading, error
        self.error_message = None
        self.last_heartbeat = None
        self._last_cycle_time = {}  # Track last execution per session type

        # IPC
        self.ipc_server = None
        self.log_streamer = LogStreamer()

        # Thread safety
        self.state_lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.signal_lock = threading.Lock()
        self.heartbeat_thread = None
        self.dexter_gate = None

    def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("Initializing trading bot service...")

            # Load configuration
            self.config = load_configuration(self.config_path)
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            # Initialize Alpaca API
            self.api = initialize_alpaca_api(
                self.config['alpaca']['api_key'],
                self.config['alpaca']['api_secret'],
                self.config['alpaca']['base_url']
            )

            # Initialize guards
            # PDT regulatory cutover: FINRA Rule 4210 amendments take effect 2026-06-04.
            # Before that date, force day_trading_enabled=False regardless of config to
            # avoid PDT violations. After that date, honor whatever config says.
            day_trading_cfg = self.config.get('day_trading', {})
            from datetime import date as _date
            PDT_RETIRES_ON = _date(2026, 6, 4)
            cfg_dt_enabled = bool(day_trading_cfg.get('enabled', False))
            if _date.today() < PDT_RETIRES_ON:
                effective_dt_enabled = False
                if cfg_dt_enabled:
                    logger.warning(
                        f"day_trading.enabled=true in config but PDT rule is still active "
                        f"until {PDT_RETIRES_ON}. Forcing day_trading_enabled=False to "
                        f"prevent PDT violations. Config will take effect on/after that date."
                    )
            else:
                effective_dt_enabled = cfg_dt_enabled
                logger.info(
                    f"PDT rule retired ({PDT_RETIRES_ON} reached). "
                    f"day_trading_enabled = {effective_dt_enabled} (from config)."
                )
            self.guard = DayTradeGuard(
                max_day_trades=day_trading_cfg.get('max_day_trades', self.config.get('max_day_trades', 2)),
                day_trading_enabled=effective_dt_enabled,
            )

            # Dexter gate (optional trade veto based on dexter_bias.json)
            try:
                from dexter_gate import DexterGate
                self.dexter_gate = DexterGate()
                logger.info("Dexter gate initialized")
            except Exception as e:
                self.dexter_gate = None
                logger.warning(f"Dexter gate unavailable: {e}")

            # Load model if exists
            model_path = Path("artifacts/final_model.pkl")
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Loaded existing model from artifacts/final_model.pkl")
            else:
                logger.warning("No model found at artifacts/final_model.pkl - bot will need training")
                self.status = "error"
                self.error_message = "Model not found. Please train the model first."
                return False

            # Load Q-table
            self.q_table = load_q_table()  # uses artifacts/q_table.csv
            logger.info("Loaded/initialized Q-table from artifacts/q_table.csv")

            # Set selected features
            self.selected_features = [
                'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_Upper',
                'Bollinger_Lower', 'Lag1_Close', 'Lag2_Close', 'ATR', 'Stochastic_RSI', 'Volume_Change'
            ]

            # Add TA-Lib features if available (keep consistent with Trader_main module)
            try:
                import Trader_main_Grok4_20250731 as core_bot
                if getattr(core_bot, "talib", None):
                    self.selected_features.extend(['Momentum', 'SMA_20'])
                    logger.info("TA-Lib available - added Momentum and SMA_20 features")
                else:
                    logger.info("TA-Lib not available - using core features only")
            except Exception:
                logger.info("TA-Lib not available - using core features only")

            # Start IPC server
            self.ipc_server = IPCServer(self._handle_command)
            self.ipc_server.start()

            self._start_heartbeat()
            self.status = "idle"
            logger.info("Trading bot service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing bot service: {e}", exc_info=True)
            self.status = "error"
            self.error_message = str(e)
            return False

    def _handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming commands from GUI"""
        cmd = command.get('command')

        try:
            if cmd == 'start':
                return self.start_trading()
            elif cmd == 'stop':
                return self.stop_trading()
            elif cmd == 'run_now':
                return self.run_now()
            elif cmd == 'get_status':
                return self.get_status()
            elif cmd == 'get_positions':
                return self.get_positions()
            elif cmd == 'get_signals':
                return self.get_signals()
            elif cmd == 'refresh_signals':
                return self.refresh_signals()
            elif cmd == 'get_signals_status':
                return self.get_signals_status()
            elif cmd == 'get_account':
                return self.get_account_info()
            elif cmd == 'run_backtest':
                return self.run_backtest()
            elif cmd == 'get_backtest_status':
                return self.get_backtest_status()
            elif cmd == 'claude_chat':
                return self.claude_chat(command.get('query', ''), command.get('include_context', True))
            elif cmd == 'manual_trade':
                return self.execute_manual_trade(
                    command.get('ticker'),
                    command.get('action'),
                    command.get('quantity')
                )
            elif cmd == 'flatten_all':
                return self.flatten_all(halt_after=bool(command.get('halt_after', True)))
            elif cmd == 'reset_kill_switch':
                cleared = clear_kill_switch()
                logger.warning(f"Drawdown kill switch reset via IPC (cleared={cleared}). Equity peak reset.")
                return {"success": True, "cleared": cleared}
            elif cmd == 'reload_model':
                return self.reload_model()
            else:
                return {"error": f"Unknown command: {cmd}"}

        except Exception as e:
            logger.error(f"Error handling command {cmd}: {e}", exc_info=True)
            return {"error": str(e)}

    def claude_chat(self, query: str, include_context: bool = True) -> Dict[str, Any]:
        """
        Direct Claude chat with rich portfolio context.
        Calls Claude API directly for portfolio-aware analysis.
        """
        if not isinstance(query, str) or not query.strip():
            return {"error": "Missing query."}

        try:
            import anthropic
        except ImportError:
            return {"error": "anthropic package not installed. Run: pip install anthropic"}

        api_key = (self.config or {}).get('claude', {}).get('api_key', '')
        if not api_key:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"error": "No Claude API key found. Set claude.api_key in config.yaml or ANTHROPIC_API_KEY env var."}

        # Build rich portfolio context
        context_parts = []
        if include_context:
            try:
                # Account info
                acct = self.get_account_info()
                if not acct.get("error"):
                    pdt_val = acct.get('pattern_day_trader')
                    pdt_line = "  Day Trading: Unlimited (post-PDT framework)" if not pdt_val else f"  PDT Status: {pdt_val}"
                    context_parts.append(
                        f"ACCOUNT:\n"
                        f"  Equity: ${acct.get('equity', 0):,.2f}\n"
                        f"  Cash: ${acct.get('cash', 0):,.2f}\n"
                        f"  Buying Power: ${acct.get('buying_power', 0):,.2f}\n"
                        f"  Portfolio Value: ${acct.get('portfolio_value', 0):,.2f}\n"
                        f"{pdt_line}"
                    )
            except Exception:
                pass

            try:
                # Current positions
                pos = self.get_positions()
                positions = pos.get("positions", [])
                if positions:
                    pos_lines = ["CURRENT POSITIONS:"]
                    for p in positions:
                        ticker = p.get('ticker', '?')
                        qty = p.get('qty', 0)
                        avg = p.get('avg_entry_price', 0)
                        cur = p.get('current_price', 0)
                        pl = p.get('unrealized_pl', 0)
                        plpc = p.get('unrealized_plpc', 0)
                        mv = p.get('market_value', 0)
                        pos_lines.append(
                            f"  {ticker}: {qty} shares @ ${avg:.2f} avg | "
                            f"Current: ${cur:.2f} | Value: ${mv:,.2f} | "
                            f"P&L: ${pl:+,.2f} ({plpc:+.1f}%)"
                        )
                    context_parts.append("\n".join(pos_lines))
                else:
                    context_parts.append("CURRENT POSITIONS: None")
            except Exception:
                pass

            try:
                # Config tickers (watchlist)
                tickers = (self.config or {}).get('tickers', [])
                crypto_tickers = (self.config or {}).get('crypto', {}).get('tickers', [])
                context_parts.append(f"WATCHLIST (config tickers): {', '.join(tickers)}")
                if crypto_tickers:
                    context_parts.append(f"CRYPTO WATCHLIST: {', '.join(crypto_tickers)}")
            except Exception:
                pass

            try:
                # Current signals
                sig = self.get_signals()
                signals = sig.get("signals", [])
                if signals:
                    sig_lines = ["CURRENT TRADING SIGNALS:"]
                    for s in signals[:15]:
                        if isinstance(s, dict):
                            sig_lines.append(f"  {s}")
                        else:
                            sig_lines.append(f"  {s}")
                    context_parts.append("\n".join(sig_lines))
            except Exception:
                pass

            try:
                # Risk settings from config
                cfg = self.config or {}
                context_parts.append(
                    f"RISK SETTINGS:\n"
                    f"  Stop Loss: {cfg.get('stop_loss_pct', 0.04) * 100:.0f}%\n"
                    f"  Take Profit: {cfg.get('take_profit_pct', 0.08) * 100:.0f}%\n"
                    f"  Max Position: {cfg.get('max_position_pct', 5)}%\n"
                    f"  VIX Threshold: {cfg.get('vix_threshold', 25)}\n"
                    f"  Buying Power Usage: {cfg.get('buying_power_pct', 50)}%"
                )
            except Exception:
                pass

        context_block = "\n\n".join(context_parts) if context_parts else "No portfolio context available."

        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")

        system_prompt = f"""You are a trading portfolio advisor integrated into an algorithmic trading bot. Today is {today}.

You have access to the user's live portfolio data, watchlist, and trading signals. Use this data to provide specific, actionable analysis.

Your role:
- Analyze portfolio composition, sector exposure, and concentration risk
- Assess current positions with specific P&L data from the context
- Recommend ticker additions, removals, or rebalancing with clear reasoning
- Consider current market conditions, sector rotation, and diversification
- Note when the watchlist is too concentrated in one sector
- Suggest defensive, value, or cyclical additions when appropriate to balance risk
- Provide specific price levels and entry points when relevant

Format guidelines:
- Lead with a bold portfolio assessment summary
- Show current position snapshots with prices and P&L when relevant
- Provide specific, actionable recommendations with reasoning
- Use clear structure with headers and bullet points
- Include brief market context
- End with risk disclaimers

Always include the disclaimer that this is not personalized financial advice and the user should do their own research."""

        user_message = f"""PORTFOLIO CONTEXT:
{context_block}

USER QUESTION:
{query}"""

        try:
            logger.info("Claude chat request received.")
            client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            response_text = message.content[0].text
            return {"success": True, "answer": response_text, "_source": "claude"}
        except Exception as e:
            logger.error(f"Claude chat failed: {e}", exc_info=True)
            return {"error": f"Claude API error: {e}"}

    def start_trading(self) -> Dict[str, Any]:
        """Start the trading bot"""
        with self.state_lock:
            if self.running:
                return {"success": False, "message": "Bot is already running"}

            if self.status == "error":
                return {"success": False, "message": f"Cannot start: {self.error_message}"}

            self.running = True
            self._set_status("running")
            self.bot_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.bot_thread.start()

            logger.info("Trading bot started")
            return {"success": True, "message": "Bot started successfully"}

    def stop_trading(self) -> Dict[str, Any]:
        """Stop the trading bot"""
        with self.state_lock:
            if not self.running:
                return {"success": False, "message": "Bot is not running"}

            self.running = False
            self._set_status("stopped")
            logger.info("Trading bot stopped")
            return {"success": True, "message": "Bot stopped successfully"}

    def run_now(self) -> Dict[str, Any]:
        """Trigger an immediate trading cycle (useful for testing)."""
        with self.state_lock:
            if not self.running:
                return {"success": False, "message": "Bot is not running. Click Start Bot first."}
        threading.Thread(target=self._execute_cycle, kwargs={"force": True}, daemon=True).start()
        return {"success": True, "message": "Triggered run-now"}

    def _start_heartbeat(self):
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self):
        while True:
            self.last_heartbeat = datetime.now().isoformat(timespec="seconds")
            time.sleep(1)

    def _refresh_api_session(self) -> bool:
        """Refresh Alpaca API session and reload config in case keys changed."""
        previous_api = self.api
        try:
            # Reload config before refresh so rotated credentials take effect without restart.
            refreshed_config = load_configuration(self.config_path)
            refreshed_config = maybe_override_tickers_from_json(refreshed_config)
            refreshed_config = maybe_add_crypto_tickers(refreshed_config)
            self.config = refreshed_config

            key = self.config['alpaca']['api_key']
            logger.info(f"Refreshing Alpaca session with key {key[:4]}...{key[-4:]}")

            refreshed_api = initialize_alpaca_api(
                key,
                self.config['alpaca']['api_secret'],
                self.config['alpaca']['base_url']
            )
            # Validate the new session with a quick account check.
            acct = refreshed_api.get_account()
            self.api = refreshed_api
            logger.info(f"Alpaca API session refreshed. Account status: {acct.status}, equity: ${acct.equity}")
            return True
        except Exception as e:
            # Keep previous API object if refresh fails.
            self.api = previous_api
            if is_alpaca_auth_error(e):
                logger.error(
                    "Failed to refresh API session: unauthorized (check Alpaca key/secret/base_url): %s",
                    e,
                )
            elif is_transient_alpaca_error(e):
                logger.error(f"Failed to refresh API session: {e}")
            else:
                logger.error(f"Failed to refresh API session: {e}", exc_info=True)
            return False

    def _execute_cycle(self, force: bool = False, session: str = 'afternoon'):
        """Execute a single trading cycle, optionally bypassing the time window.

        Args:
            force: If True, bypass time window checks
            session: 'morning' for gap trading, 'afternoon' for EOD signals
        """
        with self.job_lock:
            current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
            logger.info(f"Cycle running at {current_time.strftime('%H:%M:%S')} EDT (force={force}, session={session})")

            now_time = current_time.time()

            # DEV NOTE (fix for automatic trading issue discussed 2026-06):
            # Previously, config.yaml used morning_run: '09:45' / afternoon_run: '14:30'
            # but execution windows were hardcoded starting at 9:55/14:55.
            # This caused EVERY scheduled job (from schedule.every().day.at(...)) to log
            # "Outside ... window" and skip, even on weekdays. "Run Now" (force=True) worked.
            # Root cause identified from bot_service.log + journals (multiple "0 trades",
            # explicit "timing windows" complaints in 2026-06.md and 2026-05.md).
            # Fix: compute windows dynamically from the user's configured schedule times
            # so the nominal run time (e.g. 09:45) is the start of the allowed window,
            # giving ~30min buffer for scheduler drift / late firing while ensuring the
            # configured times actually execute. Updated both here and the startup check.
            # This makes automatic 2x daily trading actually work without manual "Run Now".
            schedule_config = (self.config or {}).get('schedule', {})
            morning_time_str = self._normalize_schedule_time(
                schedule_config.get('morning_run'), '09:45'
            )
            afternoon_time_str = self._normalize_schedule_time(
                schedule_config.get('afternoon_run'), '14:30'
            )

            def _parse_to_dt(tstr: str) -> dt_time:
                try:
                    hh, mm = map(int, tstr.split(':'))
                    return dt_time(hh, mm)
                except Exception:
                    return dt_time(9, 45)

            morning_start = _parse_to_dt(morning_time_str)
            # 30 min window starting at the scheduled time (allows drift after nominal run)
            m_h, m_m = map(int, morning_time_str.split(':'))
            m_end_m = (m_m + 30) % 60
            m_end_h = (m_h + (m_m + 30) // 60) % 24
            morning_end = dt_time(m_end_h, m_end_m)

            afternoon_start = _parse_to_dt(afternoon_time_str)
            a_h, a_m = map(int, afternoon_time_str.split(':'))
            a_end_m = (a_m + 30) % 60
            a_end_h = (a_h + (a_m + 30) // 60) % 24
            afternoon_end = dt_time(a_end_h, a_end_m)

            if not force:
                # Check if we're in a valid execution window
                in_morning_window = morning_start <= now_time <= morning_end
                in_afternoon_window = afternoon_start <= now_time <= afternoon_end
                is_weekday = current_time.weekday() < 5

                if not is_weekday:
                    logger.info("Weekend - skipping cycle.")
                    self._set_status("running")
                    return

                if session == 'intraday':
                    intraday_cfg = (self.config or {}).get('intraday', {}) or {}
                    i_start = _parse_to_dt(self._normalize_schedule_time(intraday_cfg.get('start'), '09:45'))
                    i_end = _parse_to_dt(self._normalize_schedule_time(intraday_cfg.get('end'), '15:15'))
                    if not (i_start <= now_time <= i_end):
                        logger.info(f"Outside intraday window ({i_start.strftime('%H:%M')}-{i_end.strftime('%H:%M')} ET). Current: {now_time.strftime('%H:%M')}. Skipping cycle.")
                        self._set_status("running")
                        return
                    interval_s = max(5, int(intraday_cfg.get('interval_minutes', 30))) * 60
                    last_run = self._last_cycle_time.get('intraday')
                    if last_run and (current_time - last_run).total_seconds() < interval_s - 120:
                        logger.info(f"Skipping intraday cycle — ran {int((current_time - last_run).total_seconds())}s ago.")
                        self._set_status("running")
                        return
                elif session == 'morning' and not in_morning_window:
                    morning_end_str = f"{morning_end.hour:02d}:{morning_end.minute:02d}"
                    logger.info(f"Outside morning execution window ({morning_time_str}-{morning_end_str} ET). Current: {now_time.strftime('%H:%M')}. Skipping cycle.")
                    self._set_status("running")
                    return
                elif session == 'afternoon' and not in_afternoon_window:
                    afternoon_end_str = f"{afternoon_end.hour:02d}:{afternoon_end.minute:02d}"
                    logger.info(f"Outside afternoon execution window ({afternoon_time_str}-{afternoon_end_str} ET). Current: {now_time.strftime('%H:%M')}. Skipping cycle.")
                    self._set_status("running")
                    return

                # Skip if this session already ran within the last 20 minutes
                # (intraday has its own interval-based guard above)
                last_run = None if session == 'intraday' else self._last_cycle_time.get(session)
                if last_run and (current_time - last_run).total_seconds() < 1200:
                    logger.info(f"Skipping {session} cycle — already ran {int((current_time - last_run).total_seconds())}s ago at {last_run.strftime('%H:%M:%S')}.")
                    self._set_status("running")
                    return

            # Refresh API session before each cycle to avoid stale connections.
            if not self._refresh_api_session():
                logger.error("Skipping cycle because Alpaca API session refresh failed.")
                self._set_status("error", error="Alpaca API refresh failed")
                return

            self._set_status("trading")
            try:
                cycle_timeout = int(self.config.get('cycle_timeout_seconds', 600))  # default 10 min
                cycle_error = [None]  # mutable container for thread result

                def _cycle_inner():
                    try:
                        self._run_cycle_body(session, current_time)
                    except Exception as exc:
                        cycle_error[0] = exc

                cycle_thread = threading.Thread(target=_cycle_inner, daemon=True)
                cycle_thread.start()
                cycle_thread.join(timeout=cycle_timeout)

                if cycle_thread.is_alive():
                    logger.error(f"Trading cycle TIMED OUT after {cycle_timeout}s — schedule loop will continue. "
                                 "The hung thread will be abandoned (daemon).")
                    self._set_status("error", error=f"Cycle timed out after {cycle_timeout}s")
                    return

                if cycle_error[0] is not None:
                    raise cycle_error[0]

                with self.state_lock:
                    self.last_execution_time = current_time
                self._last_cycle_time[session] = current_time
                self._set_status("running")
            except Exception as e:
                logger.error(f"Error in cycle: {e}", exc_info=True)
                self._set_status("error", error=str(e))

    def _run_cycle_body(self, session, current_time):
        """Inner cycle body — runs inside a daemon thread with timeout protection."""
        try:
            # FIRST: Check existing positions for stop loss enforcement
            # This catches positions that should have been stopped out
            logger.info("Checking existing positions for stop loss enforcement...")
            stop_actions = check_and_enforce_stops(self.api, self.config)
            if stop_actions:
                logger.warning(f"Position monitor closed {len(stop_actions)} positions")

            # Portfolio-level drawdown kill switch. Updates peak equity and trips
            # a flag when drawdown exceeds the configured threshold. Tripped state
            # blocks new entries below; existing positions remain managed by stops.
            dd_state = check_drawdown_kill_switch(self.api, self.config)
            kill_tripped = is_kill_switch_tripped()

            # Fetch historical data if needed
            if self.raw_historical is None:
                self.raw_historical = fetch_current_market_data_with_crypto(
                    self.config['tickers'],
                    self.config['polygon']['api_key'],
                    self.api,
                    config=self.config,
                )
                if self.raw_historical.empty:
                    raise RuntimeError("Failed to fetch historical data")

            # Fetch latest data
            latest_data = fetch_latest_data(self.config['tickers'], self.api, config=self.config)
            if latest_data.empty:
                logger.warning("No new data fetched. Skipping this cycle.")
                self._set_status("running")
                return

            import pandas as pd
            self.raw_historical = pd.concat([self.raw_historical, latest_data]).drop_duplicates(
                subset=['ticker', 'date'], keep='last'
            )
            self.raw_historical = self.raw_historical.groupby('ticker').tail(200).reset_index(drop=True)

            engineered_data = engineer_features(self.raw_historical, config=self.config)
            if engineered_data.empty:
                logger.warning("Feature engineering resulted in empty DataFrame. Skipping this cycle.")
                self._set_status("running")
                return
            try:
                engineered_data = add_sentiment_features(engineered_data, self.config)
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}. Using zero sentiment as fallback.")
                engineered_data['Sentiment_Score'] = 0.0

            positions_snapshot = {p['ticker']: p for p in fetch_current_positions(self.api)}

            with self.signal_lock:
                signals_df = generate_signals(
                    self.model,
                    engineered_data,
                    self.selected_features + ['Sentiment_Score'],
                    self.config,
                    self.q_table,
                    positions_snapshot=positions_snapshot,
                )

            if signals_df.empty:
                with self.state_lock:
                    self.current_signals = []
                self._set_status("running")
                return

            latest_signals = signals_df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)
            with self.state_lock:
                self.current_signals = self._format_signals_for_gui(latest_signals)

            actionable = latest_signals[latest_signals['Signal'].isin([1, -1])]
            # Only execute trades for tickers with fresh bars fetched this cycle.
            # This avoids trading stale stock signals when Alpaca does not return a new daily bar.
            fresh_tickers = set(latest_data['ticker'].astype(str).tolist()) if 'ticker' in latest_data.columns else set()
            if fresh_tickers:
                stale_actionable = actionable[~actionable['ticker'].isin(fresh_tickers)]
                if not stale_actionable.empty:
                    stale_names = ", ".join(sorted(stale_actionable['ticker'].astype(str).unique().tolist()))
                    logger.info(f"Skipping stale actionable signals without fresh bars: {stale_names}")
                actionable = actionable[actionable['ticker'].isin(fresh_tickers)]
            if not actionable.empty:
                if kill_tripped:
                    logger.warning(
                        f"Drawdown kill switch tripped — blocking {len(actionable)} new entries this cycle. "
                        f"Reset via dashboard to resume trading."
                    )
                else:
                    execute_trading_logic_live(
                        self.api,
                        actionable,
                        self.config,
                        self.q_table,
                        self.guard,
                        dexter_gate=self.dexter_gate,
                        buying_power_pct=float((self.config or {}).get("buying_power_pct", 100)),
                    )
                    with self.state_lock:
                        self.trade_count += int(len(actionable))
        except Exception as e:
            logger.error(f"Error in cycle body: {e}", exc_info=True)
            raise

    def _set_status(self, status: str, error: str | None = None):
        with self.state_lock:
            self.status = status
            if error is not None:
                self.error_message = error

    @staticmethod
    def _normalize_schedule_time(value: str, default: str) -> str:
        """Force HH:MM (zero-padded) format that the `schedule` library accepts."""
        if not value:
            return default
        try:
            parts = str(value).strip().split(':')
            if len(parts) < 2:
                return default
            hh = int(parts[0])
            mm = int(parts[1])
            if not (0 <= hh < 24 and 0 <= mm < 60):
                return default
            return f"{hh:02d}:{mm:02d}"
        except (ValueError, AttributeError):
            return default

    def _trading_loop(self):
        """Main trading loop - runs in background thread"""
        # Get schedule config (default to 2x daily: morning gaps + afternoon close)
        schedule_config = (self.config or {}).get('schedule', {})
        morning_time = self._normalize_schedule_time(
            schedule_config.get('morning_run'), '10:00'
        )  # 30 min after open for gap confirmation
        afternoon_time = self._normalize_schedule_time(
            schedule_config.get('afternoon_run'), '15:15'
        )  # 45 min before close
        enable_morning = schedule_config.get('enable_morning_run', True)

        # ---- Intraday mode: cycle every N minutes during market hours ----
        intraday_cfg = (self.config or {}).get('intraday', {}) or {}
        if intraday_cfg.get('enabled', False):
            interval = max(5, int(intraday_cfg.get('interval_minutes', 30)))
            i_start_str = self._normalize_schedule_time(intraday_cfg.get('start'), '09:45')
            i_end_str = self._normalize_schedule_time(intraday_cfg.get('end'), '15:15')
            logger.info(f"Starting INTRADAY trading mode: every {interval} min, {i_start_str}-{i_end_str} ET (weekdays)")

            def intraday_job():
                self._execute_cycle(force=False, session='intraday')

            # Immediate run at startup if inside the window
            _now = datetime.now(tz=pytz.timezone('US/Eastern'))
            _i_start = dt_time(*map(int, i_start_str.split(':')))
            _i_end = dt_time(*map(int, i_end_str.split(':')))
            if _now.weekday() < 5 and _i_start <= _now.time() <= _i_end:
                logger.info("Within intraday window at startup. Running cycle...")
                self._execute_cycle(force=False, session='intraday')
            else:
                logger.info(f"Outside intraday window at startup ({i_start_str}-{i_end_str} ET).")

            schedule.every(interval).minutes.do(intraday_job)
            self._register_rotation_job()
            self._register_analyzer_job()
            self._update_next_execution_time()

            while self.running:
                schedule.run_pending()
                self._update_next_execution_time()
                time.sleep(60)

            logger.info("Trading loop stopped")
            return

        if enable_morning:
            logger.info(f"Starting 2x daily trading mode (Morning: {morning_time} ET, Afternoon: {afternoon_time} ET)...")
        else:
            logger.info(f"Starting daily close trading mode (executes at {afternoon_time} ET)...")

        def morning_job():
            """Morning job - catch overnight gaps"""
            last_run = self._last_cycle_time.get('morning')
            if last_run:
                now = datetime.now(tz=pytz.timezone('US/Eastern'))
                elapsed = (now - last_run).total_seconds()
                if elapsed < 1200:
                    logger.info(f"Skipping scheduled morning run — startup already ran {int(elapsed)}s ago at {last_run.strftime('%H:%M:%S')}.")
                    return
            logger.info("=== MORNING RUN: Checking for gap opportunities ===")
            self._execute_cycle(force=False, session='morning')

        def afternoon_job():
            """Afternoon job - end of day signals"""
            # Check duplicate guard BEFORE acquiring job_lock to give clear logs
            last_run = self._last_cycle_time.get('afternoon')
            if last_run:
                now = datetime.now(tz=pytz.timezone('US/Eastern'))
                elapsed = (now - last_run).total_seconds()
                if elapsed < 1200:
                    logger.info(f"Skipping scheduled afternoon run — startup already ran {int(elapsed)}s ago at {last_run.strftime('%H:%M:%S')}.")
                    return
            logger.info("=== AFTERNOON RUN: End of day trading signals ===")
            self._execute_cycle(force=False, session='afternoon')

        # Check if we should run immediately based on current time
        current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
        now_time = current_time.time()

        # DEV NOTE (fix for automatic trading issue discussed 2026-06):
        # Same window alignment problem as in _execute_cycle: hardcoded 9:55/14:55
        # meant that if the process started near the configured time (09:45/14:30),
        # the startup immediate-run check would also skip or misbehave.
        # Now dynamically derived from the (already normalized) morning_time/afternoon_time
        # strings so behavior is consistent with scheduled jobs and user config.
        # Windows start at the nominal scheduled time and extend ~30min for drift.
        morning_start = dt_time(*map(int, morning_time.split(':')))
        m_h, m_m = map(int, morning_time.split(':'))
        m_end_m = (m_m + 30) % 60
        m_end_h = (m_h + (m_m + 30) // 60) % 24
        morning_end = dt_time(m_end_h, m_end_m)

        afternoon_start = dt_time(*map(int, afternoon_time.split(':')))
        a_h, a_m = map(int, afternoon_time.split(':'))
        a_end_m = (a_m + 30) % 60
        a_end_h = (a_h + (a_m + 30) // 60) % 24
        afternoon_end = dt_time(a_end_h, a_end_m)

        if current_time.weekday() < 5:  # Weekday
            if enable_morning and morning_start <= now_time <= morning_end:
                logger.info("Within morning execution window at startup. Running morning job...")
                self._execute_cycle(force=False, session='morning')
            elif afternoon_start <= now_time <= afternoon_end:
                logger.info("Within afternoon execution window at startup. Running afternoon job...")
                self._execute_cycle(force=False, session='afternoon')
            else:
                logger.info(f"Outside execution windows at startup. Next runs: {morning_time} AM, {afternoon_time} PM ET")

        # Record startup time so the duplicate-run guard in _execute_cycle can
        # prevent the scheduled job from re-running if the startup already ran.
        if self._last_cycle_time:
            logger.info(f"Startup cycle completed. Duplicate guard active for: {list(self._last_cycle_time.keys())}")

        # Schedule morning run (gap trading)
        if enable_morning:
            schedule.every().day.at(morning_time).do(morning_job)
            logger.info(f"Scheduled morning execution at {morning_time} ET (gap trading)")

        # Schedule afternoon run (end of day)
        schedule.every().day.at(afternoon_time).do(afternoon_job)
        logger.info(f"Scheduled afternoon execution at {afternoon_time} ET (EOD signals)")

        # Daily pre-market AI watchlist rotation (optional)
        self._register_rotation_job()
        self._register_analyzer_job()

        # Calculate next execution time
        self._update_next_execution_time()

        # Main loop
        while self.running:
            schedule.run_pending()
            self._update_next_execution_time()
            time.sleep(60)  # Check every minute

        logger.info("Trading loop stopped")

    def _register_rotation_job(self):
        """Register the optional daily pre-market AI watchlist rotation.

        Runs auto_ticker_rotator.py as a subprocess before market open so the
        bot picks its own stocks each day (safety rails live in the rotator:
        held positions never rotated out, whitelist, churn cap, audit log).
        The next trading cycle picks up tickers_auto.json automatically.
        """
        rot_cfg = (self.config or {}).get('watchlist_rotation', {}) or {}
        if not rot_cfg.get('daily_premarket', False):
            return
        rot_time = self._normalize_schedule_time(rot_cfg.get('time'), '09:00')

        def rotation_job():
            now = datetime.now(tz=pytz.timezone('US/Eastern'))
            if now.weekday() >= 5:
                return
            logger.info("=== PRE-MARKET WATCHLIST ROTATION ===")
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / 'auto_ticker_rotator.py')],
                    capture_output=True, text=True, timeout=900,
                    cwd=str(Path(__file__).parent),
                )
                if result.returncode == 0:
                    logger.info("Watchlist rotation completed — new tickers apply on the next cycle.")
                else:
                    logger.warning(f"Watchlist rotation exited {result.returncode}: {(result.stderr or '')[-500:]}")
            except Exception as e:
                logger.error(f"Watchlist rotation failed: {e}")

        schedule.every().day.at(rot_time).do(rotation_job)
        logger.info(f"Scheduled daily pre-market watchlist rotation at {rot_time} ET")

    def _register_analyzer_job(self):
        """Nightly FIFO trade analysis: refreshes logs/trades.db + trade_report.html
        so the dashboard's Matched Trades table stays current (was manual-only)."""
        def analyzer_job():
            now = datetime.now(tz=pytz.timezone('US/Eastern'))
            if now.weekday() >= 5:
                return
            logger.info("=== NIGHTLY TRADE ANALYSIS (FIFO refresh) ===")
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / 'trade_analyzer.py')],
                    capture_output=True, text=True, timeout=600,
                    cwd=str(Path(__file__).parent),
                )
                if result.returncode == 0:
                    logger.info("Trade analysis refreshed (logs/trades.db + trade_report.html).")
                else:
                    logger.warning(f"Trade analyzer exited {result.returncode}: {(result.stderr or '')[-400:]}")
            except Exception as e:
                logger.error(f"Trade analyzer failed: {e}")

        schedule.every().day.at('16:15').do(analyzer_job)
        logger.info("Scheduled nightly trade analysis at 16:15 ET")

    def _update_next_execution_time(self):
        """Update the next scheduled execution time"""
        current_time = datetime.now(tz=pytz.timezone('US/Eastern'))
        next_run = schedule.next_run()

        if next_run:
            self.next_execution_time = next_run.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Calculate tomorrow's first execution time (morning run)
            tomorrow = current_time + timedelta(days=1)
            schedule_config = (self.config or {}).get('schedule', {})
            morning_time = schedule_config.get('morning_run', '10:00')
            hour, minute = map(int, morning_time.split(':'))
            self.next_execution_time = tomorrow.replace(hour=hour, minute=minute, second=0).strftime("%Y-%m-%d %H:%M:%S")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        with self.state_lock:
            return {
                "status": self.status,
                "running": self.running,
                "next_execution": self.next_execution_time,
                "last_execution": self.last_execution_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_execution_time else None,
                "trade_count": self.trade_count,
                "last_heartbeat": self.last_heartbeat,
                "error": self.error_message
            }

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            raw_positions = fetch_current_positions(self.api)
            positions = []
            for p in raw_positions:
                qty = float(p.get('quantity', 0))
                avg_entry = float(p.get('average_price', 0))
                current = float(p.get('current_price', avg_entry))
                market_value = qty * current
                unrealized_pl = (current - avg_entry) * qty if avg_entry else 0.0
                unrealized_plpc = ((current - avg_entry) / avg_entry) * 100 if avg_entry else 0.0
                positions.append({
                    "ticker": p.get('ticker'),
                    "qty": qty,
                    "avg_entry_price": avg_entry,
                    "current_price": current,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                })
            return {"positions": positions}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {"error": str(e)}

    def get_signals(self) -> Dict[str, Any]:
        """Get current trading signals"""
        with self.state_lock:
            return {"signals": self.current_signals}

    def refresh_signals(self) -> Dict[str, Any]:
        """Kick off a background signals refresh (non-blocking for GUI)."""
        with self.state_lock:
            if self.signals_refresh_running:
                return {"success": False, "message": "Signals refresh already running."}
            self.signals_refresh_running = True
            self.signals_refresh_error = None
        threading.Thread(target=self._signals_refresh_job, daemon=True).start()
        return {"success": True, "message": "Signals refresh started."}

    def get_signals_status(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                "running": self.signals_refresh_running,
                "last_refresh": self.last_signals_refresh.isoformat(timespec="seconds") if self.last_signals_refresh else None,
                "error": self.signals_refresh_error,
            }

    def _signals_refresh_job(self):
        """Compute latest signals without executing trades."""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Train first.")

            if not self._refresh_api_session():
                raise RuntimeError("Failed to refresh Alpaca session before signals refresh.")

            # Refresh config/tickers in case they changed
            self.config = load_configuration(self.config_path)
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            raw = fetch_current_market_data_with_crypto(
                self.config['tickers'],
                self.config['polygon']['api_key'],
                self.api,
                config=self.config,
            )
            if raw.empty:
                raise RuntimeError("No market data returned.")

            latest = fetch_latest_data(self.config['tickers'], self.api, config=self.config)
            if not latest.empty:
                import pandas as pd
                raw = pd.concat([raw, latest]).drop_duplicates(subset=['ticker', 'date'], keep='last')

            engineered = engineer_features(raw, config=self.config)
            if engineered.empty:
                raise RuntimeError("Feature engineering produced empty data.")
            engineered = add_sentiment_features(engineered, self.config)

            positions_snapshot = {p['ticker']: p for p in fetch_current_positions(self.api)}
            with self.signal_lock:
                signals_df = generate_signals(
                    self.model,
                    engineered,
                    self.selected_features + ['Sentiment_Score'],
                    self.config,
                    self.q_table,
                    positions_snapshot=positions_snapshot,
                )
            if signals_df.empty:
                with self.state_lock:
                    self.current_signals = []
                    self.last_signals_refresh = datetime.now()
                return

            latest_signals = signals_df.sort_values(['ticker', 'date']).groupby('ticker').tail(1)
            formatted = self._format_signals_for_gui(latest_signals)
            with self.state_lock:
                self.current_signals = formatted
                self.last_signals_refresh = datetime.now()
        except Exception as e:
            logger.error(f"Error refreshing signals: {e}", exc_info=True)
            with self.state_lock:
                self.signals_refresh_error = str(e)
        finally:
            with self.state_lock:
                self.signals_refresh_running = False

    def run_backtest(self) -> Dict[str, Any]:
        """Start a background job to download/train/backtest."""
        with self.state_lock:
            if self.backtest_running:
                return {"success": False, "message": "Backtest already running."}
            self.backtest_running = True
            self.backtest_last_error = None
            self.backtest_started_at = datetime.now().isoformat(timespec="seconds")
            self.backtest_last_update = self.backtest_started_at
            self.backtest_phase = "starting"
            self.backtest_progress = "Queued"
        self._backtest_thread = threading.Thread(target=self._backtest_job, daemon=True)
        self._backtest_thread.start()
        return {"success": True, "message": "Backtest started."}

    def get_backtest_status(self) -> Dict[str, Any]:
        with self.state_lock:
            thread_alive = bool(self._backtest_thread and self._backtest_thread.is_alive())
            return {
                "running": self.backtest_running,
                "last_run": self.backtest_last_run,
                "error": self.backtest_last_error,
                "phase": self.backtest_phase,
                "started_at": self.backtest_started_at,
                "last_update": self.backtest_last_update,
                "progress": self.backtest_progress,
                "thread_alive": thread_alive,
            }

    def _backtest_job(self):
        try:
            logger.info("Starting GUI-triggered backtest job...")
            self._set_backtest_progress("loading_config", "Loading configuration and tickers")
            # Reload config/tickers
            self.config = load_configuration(self.config_path)
            self.config = maybe_override_tickers_from_json(self.config)
            self.config = maybe_add_crypto_tickers(self.config)

            tickers = self.config['tickers']
            os.makedirs("./data", exist_ok=True)
            self._set_backtest_progress("downloading", f"Downloading historical data ({len(tickers)} tickers)")
            download_historical_data_with_crypto(tickers, self.config)
            self._set_backtest_progress("loading_data", "Loading historical CSVs")
            data = load_historical_data("./data", self.config)
            if data.empty:
                raise RuntimeError("No data loaded from ./data")
            self._set_backtest_progress("feature_engineering", "Engineering features")
            data = engineer_features(data, config=self.config, is_backtest=True)
            self._set_backtest_progress("sentiment", "Adding sentiment features (can be slow)")
            data = add_sentiment_features(data, self.config, is_backtest=True)
            if data.empty:
                raise RuntimeError("No data after feature engineering/sentiment.")

            selected = list(self.selected_features) + ['Sentiment_Score']
            self._set_backtest_progress("train_split", "Preparing train/test data")
            X_train, X_test, y_train, y_test = prepare_train_test_data(data, selected)
            self._set_backtest_progress("training", "Training model (RandomizedSearchCV)")
            model = tune_and_train_model(X_train, y_train)
            if model is None:
                raise RuntimeError("Model training returned None.")
            self._set_backtest_progress("evaluation", "Evaluating model and saving artifacts")
            evaluate_model(model, X_test, y_test)
            with self.state_lock:
                self.model = model
            self._set_backtest_progress("backtesting", "Running backtest simulation")
            backtest_strategy(model, data, selected, self.config, self.q_table)

            with self.state_lock:
                self.backtest_last_run = datetime.now().isoformat(timespec="seconds")
                self.backtest_phase = "completed"
                self.backtest_progress = "Done"
                self.backtest_last_update = datetime.now().isoformat(timespec="seconds")
            logger.info("Backtest job completed successfully.")
        except Exception as e:
            logger.error(f"Backtest job failed: {e}", exc_info=True)
            with self.state_lock:
                self.backtest_last_error = str(e)
                self.backtest_phase = "failed"
                self.backtest_progress = "Failed"
                self.backtest_last_update = datetime.now().isoformat(timespec="seconds")
        finally:
            with self.state_lock:
                self.backtest_running = False

    def _set_backtest_progress(self, phase: str, message: str):
        now_s = datetime.now().isoformat(timespec="seconds")
        with self.state_lock:
            self.backtest_phase = phase
            self.backtest_progress = message
            self.backtest_last_update = now_s
        logger.info(f"[backtest] {phase}: {message}")

    def _format_signals_for_gui(self, df) -> List[Dict[str, Any]]:
        """Format latest per-ticker signals for the GUI table."""
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            sig = int(row.get('Signal', 0))
            if sig == 1:
                action = "buy"
            elif sig == -1:
                action = "sell"
            else:
                action = "hold"
            out.append({
                "ticker": row.get('ticker', ''),
                "action": action,
                "confidence": float(row.get('Prediction', 0)),
                "timestamp": str(row.get('date', '')),
            })
        return out

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information with retry on connection errors"""
        if self.api is None:
            return {"error": "Alpaca API is not initialized"}

        max_attempts = 3
        refreshed_after_auth = False
        for attempt in range(max_attempts):
            try:
                account = self.api.get_account()
                return {
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "portfolio_value": float(account.portfolio_value),
                    "equity": float(account.equity),
                    "pattern_day_trader": bool(getattr(account, "pattern_day_trader", False)),
                }
            except Exception as e:
                if is_alpaca_auth_error(e):
                    # One forced refresh attempt for expired/rotated credentials.
                    if not refreshed_after_auth:
                        refreshed_after_auth = True
                        logger.warning(
                            "Alpaca auth error fetching account; refreshing session and retrying once: %s",
                            e,
                        )
                        if self._refresh_api_session():
                            continue
                    logger.error(f"Error fetching account info: {e}")
                    return {"error": str(e)}

                if is_transient_alpaca_error(e):
                    if attempt < max_attempts - 1:
                        wait_seconds = 2 * (attempt + 1)
                        logger.warning(
                            f"Connection error fetching account (attempt {attempt + 1}/{max_attempts}), "
                            f"retrying in {wait_seconds}s: {e}"
                        )
                        time.sleep(wait_seconds)
                        self._refresh_api_session()
                        continue
                    logger.error(f"Error fetching account info after {max_attempts} attempts: {e}")
                    return {"error": str(e)}

                logger.error(f"Error fetching account info: {e}")
                return {"error": str(e)}
        return {"error": "Unknown error"}

    def execute_manual_trade(self, ticker: str, action: str, quantity: int) -> Dict[str, Any]:
        """Execute a manual trade"""
        try:
            logger.info(f"Manual trade request: {action} {quantity} shares of {ticker}")

            # Basic validation
            if action not in ['buy', 'sell']:
                return {"error": "Action must be 'buy' or 'sell'"}

            if quantity <= 0:
                return {"error": "Quantity must be positive"}

            # Execute trade via Alpaca
            if action == 'buy':
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
            else:
                order = self.api.submit_order(
                    symbol=ticker,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

            logger.info(f"Manual trade executed: {order.id}")
            return {
                "success": True,
                "order_id": order.id,
                "message": f"{action.upper()} {quantity} {ticker}"
            }

        except Exception as e:
            logger.error(f"Error executing manual trade: {e}", exc_info=True)
            return {"error": str(e)}

    def reload_model(self) -> Dict[str, Any]:
        """Reload the XGBoost model from disk. Used by auto_retrain.py after a successful retrain.

        Thread-safe enough for practical use — generate_signals reads self.model once per cycle.
        A swap mid-cycle is benign (worst case: one cycle uses old features list).
        """
        model_path = Path("artifacts/final_model.pkl")
        if not model_path.exists():
            return {"success": False, "error": "model file not found"}
        try:
            new_model = joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to reload model: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        with self.state_lock:
            self.model = new_model
            try:
                self.selected_features = list(new_model.get_booster().feature_names) or self.selected_features
            except Exception:
                pass  # keep existing feature list
        logger.warning(f"Model hot-reloaded from {model_path}. Features: {self.selected_features}")
        return {"success": True, "features": self.selected_features}

    def flatten_all(self, halt_after: bool = True) -> Dict[str, Any]:
        """KILL SWITCH: Cancel every open order and market-sell every position.

        Optionally halts the trading loop afterwards so the bot doesn't immediately
        re-enter positions on the next cycle. Designed to be safe to call from a
        phone via the dashboard's red button.

        Returns a summary including counts and any per-symbol errors so the UI can
        show exactly what happened. Does NOT raise on partial failure -- best-effort,
        report everything.
        """
        summary = {
            "success": False,
            "orders_canceled": 0,
            "positions_closed": 0,
            "errors": [],
            "trading_halted": False,
        }
        logger.warning("=" * 60)
        logger.warning("FLATTEN ALL invoked. Cancelling orders + closing positions.")
        logger.warning("=" * 60)

        # 1) Cancel ALL open orders first (so close_all doesn't fight live brackets)
        try:
            canceled = self.api.cancel_all_orders()
            summary["orders_canceled"] = len(canceled) if canceled is not None else 0
            logger.warning(f"Canceled {summary['orders_canceled']} open orders.")
        except Exception as e:
            err = f"cancel_all_orders failed: {e}"
            logger.error(err, exc_info=True)
            summary["errors"].append(err)

        # Brief settle for Alpaca to register the cancellations
        time.sleep(1)

        # 2) Close all positions (Alpaca will issue market sells for longs, market buys for shorts)
        try:
            closed = self.api.close_all_positions(cancel_orders=True)
            # close_all_positions returns a list of order objects (one per closed position).
            n = 0
            if closed is not None:
                for c in closed:
                    n += 1
                    sym = getattr(c, 'symbol', None) or (
                        getattr(c, 'body', {}).get('symbol') if hasattr(c, 'body') else None
                    )
                    status = getattr(c, 'status', None) or (
                        getattr(c, 'body', {}).get('status') if hasattr(c, 'body') else None
                    )
                    logger.warning(f"Close order: symbol={sym} status={status}")
            summary["positions_closed"] = n
        except Exception as e:
            err = f"close_all_positions failed: {e}"
            logger.error(err, exc_info=True)
            summary["errors"].append(err)
            # Fallback: try positions one by one in case close_all bombed
            try:
                positions = self.api.list_positions()
                for pos in positions:
                    try:
                        self.api.close_position(pos.symbol)
                        summary["positions_closed"] += 1
                        logger.warning(f"Fallback closed: {pos.symbol}")
                    except Exception as pe:
                        summary["errors"].append(f"close_position({pos.symbol}) failed: {pe}")
            except Exception as le:
                summary["errors"].append(f"list_positions fallback failed: {le}")

        # 3) Halt trading loop so we don't re-enter
        if halt_after:
            try:
                self.stop_trading()
                summary["trading_halted"] = True
                logger.warning("Trading loop halted post-flatten.")
            except Exception as e:
                summary["errors"].append(f"stop_trading failed: {e}")

        summary["success"] = len(summary["errors"]) == 0
        logger.warning(f"FLATTEN ALL complete: {summary}")
        return summary

    def shutdown(self):
        """Shutdown the bot service"""
        logger.info("Shutting down bot service...")
        self.stop_trading()
        if self.ipc_server:
            self.ipc_server.stop()
        logger.info("Bot service shutdown complete")


PID_FILE = Path("/tmp/trader_bot.pid")


def _kill_stale_instances():
    """Kill any previously running bot_service.py processes and clean up."""
    my_pid = os.getpid()

    # 1) Check PID file for a recorded previous instance
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if old_pid != my_pid:
                try:
                    os.kill(old_pid, 0)  # Check if alive
                    logger.warning(f"Killing previous bot_service (PID {old_pid}) from PID file")
                    os.kill(old_pid, 9)
                    time.sleep(0.5)
                except ProcessLookupError:
                    pass  # Already dead
        except (ValueError, OSError):
            pass

    # 2) Scan for any OTHER bot_service.py processes we missed
    try:
        result = subprocess.run(
            ["pgrep", "-f", "bot_service\\.py"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
                if pid != my_pid:
                    logger.warning(f"Killing stale bot_service.py process (PID {pid})")
                    os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError):
                pass
        if result.stdout.strip():
            time.sleep(1)  # Give them a moment to exit
    except Exception as e:
        logger.warning(f"Could not scan for stale processes: {e}")

    # 3) Write our PID
    PID_FILE.write_text(str(my_pid))
    logger.info(f"PID file written: {PID_FILE} (PID {my_pid})")

    # 4) Clear any stale schedule jobs from a previous import
    schedule.clear()


def main():
    """Run the bot service as a standalone process"""
    _kill_stale_instances()

    service = TradingBotService()

    if not service.initialize():
        logger.error("Failed to initialize bot service")
        sys.exit(1)

    # Auto-start trading
    service.start_trading()

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        service.shutdown()
        # Clean up PID file
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
