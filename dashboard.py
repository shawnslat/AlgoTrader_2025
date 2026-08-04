"""
AlgoTrader Streamlit Dashboard
Run: streamlit run dashboard.py --server.port 8502
"""
import os
import json
import socket
import sqlite3
import subprocess
import signal
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yaml
import requests
from openpyxl import load_workbook

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
TRADES_DB = BASE_DIR / "logs" / "trades.db"
EXIT_STATE_FILE = BASE_DIR / "logs" / "exit_tiers.json"
BOT_LOG = BASE_DIR / "logs" / "bot_service.log"
PID_FILE = Path("/tmp/trader_bot.pid")
IPC_SOCKET = "/tmp/trader_bot.sock"
BOT_SCRIPT = BASE_DIR / "bot_service.py"
INCOME_TRACKER = BASE_DIR / "income_portfolio" / "Income_Portfolio_Tracker.xlsx"
PENDING_TICKER_CHANGE = BASE_DIR / "pending_ticker_change.json"
KILL_SWITCH_FILE = BASE_DIR / "logs" / "kill_switch_tripped.json"

st.set_page_config(page_title="AlgoTrader Dashboard", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# TradingView Widget Helpers
# ---------------------------------------------------------------------------

def tv_ticker_tape(symbols: list) -> str:
    """Scrolling ticker tape widget for the top of the dashboard."""
    items = []
    for s in symbols:
        # Convert crypto format: BTC/USD -> BINANCE:BTCUSD
        if "/" in s:
            tv_sym = f"BINANCE:{s.replace('/', '')}"
        else:
            tv_sym = f"NASDAQ:{s}" if s in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA") else f"NYSE:{s}"
        items.append(f'{{"proName":"{tv_sym}","title":"{s}"}}')
    symbols_json = "[" + ",".join(items) + "]"
    return f'''
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
        {{
          "symbols": {symbols_json},
          "showSymbolLogo": true,
          "isTransparent": true,
          "displayMode": "adaptive",
          "colorTheme": "dark",
          "locale": "en"
        }}
      </script>
    </div>'''


def tv_advanced_chart(symbol: str, height: int = 500) -> str:
    """Full interactive TradingView chart using iframe embed for reliable sizing."""
    if "/" in symbol:
        tv_sym = f"BINANCE:{symbol.replace('/', '')}"
    elif symbol in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"):
        tv_sym = f"NASDAQ:{symbol}"
    else:
        tv_sym = f"NYSE:{symbol}"
    return f'''
    <html>
    <body style="margin:0;padding:0;background:#131722;">
    <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
      <div id="tv_chart_container" style="width:100%;height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "container_id": "tv_chart_container",
          "autosize": true,
          "symbol": "{tv_sym}",
          "interval": "D",
          "timezone": "America/New_York",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#131722",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
          "hide_side_toolbar": false,
          "withdateranges": true
        }});
      </script>
    </div>
    </body>
    </html>'''


def tv_technical_analysis(symbol: str, height: int = 425) -> str:
    """Technical analysis gauge widget (buy/sell/neutral)."""
    if "/" in symbol:
        tv_sym = f"BINANCE:{symbol.replace('/', '')}"
    elif symbol in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"):
        tv_sym = f"NASDAQ:{symbol}"
    else:
        tv_sym = f"NYSE:{symbol}"
    return f'''
    <html>
    <body style="margin:0;padding:0;background:transparent;">
    <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
        {{
          "interval": "1D",
          "width": "100%",
          "isTransparent": true,
          "height": "{height}",
          "symbol": "{tv_sym}",
          "showIntervalTabs": true,
          "displayMode": "single",
          "locale": "en",
          "colorTheme": "dark"
        }}
      </script>
    </div>
    </body>
    </html>'''


def tv_market_heatmap(height: int = 500) -> str:
    """S&P 500 market heatmap widget."""
    return f'''
    <html>
    <body style="margin:0;padding:0;background:transparent;">
    <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
        {{
          "exchanges": [],
          "dataSource": "SPX500",
          "grouping": "sector",
          "blockSize": "market_cap_basic",
          "blockColor": "change",
          "locale": "en",
          "symbolUrl": "",
          "colorTheme": "dark",
          "hasTopBar": true,
          "isDataSetEnabled": true,
          "isZoomEnabled": true,
          "hasSymbolTooltip": true,
          "isMonoSize": false,
          "width": "100%",
          "height": "{height}"
        }}
      </script>
    </div>
    </body>
    </html>'''


def tv_market_overview(height: int = 500) -> str:
    """Market overview widget with indices, futures, bonds, forex."""
    return f'''
    <html>
    <body style="margin:0;padding:0;background:transparent;">
    <div class="tradingview-widget-container" style="width:100%;height:{height}px;">
      <div class="tradingview-widget-container__widget" style="width:100%;height:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
        {{
          "colorTheme": "dark",
          "dateRange": "1D",
          "showChart": true,
          "locale": "en",
          "isTransparent": true,
          "showSymbolLogo": true,
          "showFloatingTooltip": true,
          "width": "100%",
          "height": "{height}",
          "tabs": [
            {{
              "title": "Indices",
              "symbols": [
                {{"s": "FOREXCOM:SPXUSD", "d": "S&P 500"}},
                {{"s": "FOREXCOM:NSXUSD", "d": "Nasdaq 100"}},
                {{"s": "FOREXCOM:DJI", "d": "Dow Jones"}},
                {{"s": "INDEX:VIX", "d": "VIX"}}
              ],
              "originalTitle": "Indices"
            }},
            {{
              "title": "Crypto",
              "symbols": [
                {{"s": "BINANCE:BTCUSD", "d": "Bitcoin"}},
                {{"s": "BINANCE:ETHUSD", "d": "Ethereum"}},
                {{"s": "BINANCE:SOLUSD", "d": "Solana"}}
              ],
              "originalTitle": "Crypto"
            }}
          ]
        }}
      </script>
    </div>
    </body>
    </html>'''


def tv_symbol_info(symbol: str) -> str:
    """Compact symbol info widget (price, change, volume)."""
    if "/" in symbol:
        tv_sym = f"BINANCE:{symbol.replace('/', '')}"
    elif symbol in ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"):
        tv_sym = f"NASDAQ:{symbol}"
    else:
        tv_sym = f"NYSE:{symbol}"
    return f'''
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js" async>
        {{
          "symbol": "{tv_sym}",
          "width": "100%",
          "locale": "en",
          "colorTheme": "dark",
          "isTransparent": true
        }}
      </script>
    </div>'''


# ---------------------------------------------------------------------------
# IPC Client (talk to bot_service)
# ---------------------------------------------------------------------------

def ipc_command(cmd: dict, timeout: float = 10.0) -> dict:
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(IPC_SOCKET)
        sock.sendall(json.dumps(cmd).encode("utf-8") + b"\n")
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break
        sock.close()
        return json.loads(data.decode("utf-8").strip()) if data else {"error": "No response"}
    except FileNotFoundError:
        return {"error": "Bot not running (no socket)"}
    except ConnectionRefusedError:
        return {"error": "Bot not running (connection refused)"}
    except socket.timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=15)
def fetch_account(_ak, _sk, _bu):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(_ak, _sk, _bu)
    a = api.get_account()
    return {
        "equity": float(a.equity), "cash": float(a.cash),
        "buying_power": float(a.buying_power), "portfolio_value": float(a.portfolio_value),
        "last_equity": float(a.last_equity), "status": a.status,
        "day_trade_count": int(getattr(a, "daytrade_count", 0) or 0),
        "pdt_flag": bool(getattr(a, "pattern_day_trader", False)),
    }


@st.cache_data(ttl=15)
def fetch_positions(_ak, _sk, _bu):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(_ak, _sk, _bu)
    rows = []
    for p in api.list_positions():
        rows.append({
            "Symbol": p.symbol, "Qty": float(p.qty),
            "Entry": float(p.avg_entry_price), "Current": float(p.current_price),
            "Market Value": float(p.market_value),
            "P&L $": float(p.unrealized_pl), "P&L %": float(p.unrealized_plpc) * 100,
            "Change Today %": float(p.change_today) * 100,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=15)
def fetch_orders(_ak, _sk, _bu, limit=50):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(_ak, _sk, _bu)
    rows = []
    for o in api.list_orders(status="all", limit=limit):
        rows.append({
            "Symbol": o.symbol, "Side": o.side,
            "Qty": float(o.qty) if o.qty else 0,
            "Filled Qty": float(o.filled_qty) if o.filled_qty else 0,
            "Avg Fill": float(o.filled_avg_price) if o.filled_avg_price else 0,
            "Status": o.status, "Type": o.type,
            "Submitted": str(o.submitted_at)[:19] if o.submitted_at else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Income Portfolio Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_income_tracker():
    """Load income portfolio tracker data from Excel."""
    data = {}
    try:
        if not INCOME_TRACKER.exists():
            return None
        wb = load_workbook(INCOME_TRACKER, data_only=True)

        # Holdings
        ws = wb["Holdings"]
        holdings = []
        for row in range(5, 25):
            ticker = ws.cell(row=row, column=1).value
            if ticker:
                holdings.append({
                    "Ticker": ticker,
                    "Name": ws.cell(row=row, column=2).value or "",
                    "Bucket": ws.cell(row=row, column=3).value or "",
                    "Shares": ws.cell(row=row, column=4).value or 0,
                    "Avg Cost": ws.cell(row=row, column=5).value or 0,
                    "Price": ws.cell(row=row, column=6).value or 0,
                    "Market Value": ws.cell(row=row, column=7).value or 0,
                    "G/L": ws.cell(row=row, column=9).value or 0,
                    "G/L %": ws.cell(row=row, column=10).value or 0,
                    "Yield %": ws.cell(row=row, column=11).value or 0,
                    "Annual Income": ws.cell(row=row, column=12).value or 0,
                    "Monthly Income": ws.cell(row=row, column=13).value or 0,
                    "% Portfolio": ws.cell(row=row, column=14).value or 0,
                    "Income Type": ws.cell(row=row, column=15).value or "",
                })
        data["holdings"] = pd.DataFrame(holdings) if holdings else pd.DataFrame()

        # Cash (SWVXX row 25)
        data["cash"] = ws.cell(row=25, column=7).value or 0
        data["cash_yield"] = ws.cell(row=25, column=11).value or 0

        # Dashboard metrics
        ws = wb["Dashboard"]
        data["portfolio_invested"] = ws["B6"].value or 0
        data["portfolio_total"] = ws["B7"].value or 0
        data["monthly_income"] = ws["B9"].value or 0
        data["annual_income"] = ws["B10"].value or 0
        data["portfolio_yield"] = ws["B11"].value or 0
        data["monthly_goal"] = ws["B12"].value or 0
        data["monthly_gap"] = ws["B13"].value or 0
        data["income_achievement"] = ws["B14"].value or 0
        data["deployable_cash"] = ws["B15"].value or 0

        # Allocation drift
        alloc = []
        for row in range(19, 23):
            alloc.append({
                "Bucket": ws.cell(row=row, column=1).value or "",
                "Actual": ws.cell(row=row, column=2).value or 0,
                "Target": ws.cell(row=row, column=3).value or 0,
                "Drift": ws.cell(row=row, column=4).value or 0,
                "Status": ws.cell(row=row, column=5).value or "",
            })
        data["allocation"] = pd.DataFrame(alloc)

        # Concentration
        data["top1_income"] = ws["B35"].value or 0
        data["top3_income"] = ws["B36"].value or 0
        data["num_positions"] = ws["B37"].value or 0

        # Macro signals
        data["vix"] = ws["B44"].value or 0
        data["treasury_2y"] = ws["B42"].value or 0
        data["treasury_10y"] = ws["B43"].value or 0
        data["hy_oas"] = ws["B45"].value or 0

        # Time-to-goal scenarios
        ttg = []
        for row in range(27, 31):
            ttg.append({
                "Scenario": ws.cell(row=row, column=1).value or "",
                "Monthly Savings": ws.cell(row=row, column=2).value or 0,
                "Annual Deploy": ws.cell(row=row, column=3).value or 0,
                "Yrs (Current)": ws.cell(row=row, column=4).value or "N/A",
                "Yrs (8% Yield)": ws.cell(row=row, column=5).value or 0,
            })
        data["ttg"] = pd.DataFrame(ttg)

        # Weekly log
        ws = wb["Weekly_Log"]
        log_entries = []
        for row in range(5, 57):
            date_val = ws.cell(row=row, column=1).value
            if date_val:
                log_entries.append({
                    "Date": date_val,
                    "Portfolio Value": ws.cell(row=row, column=2).value or 0,
                    "Cash": ws.cell(row=row, column=3).value or 0,
                    "Monthly Income": ws.cell(row=row, column=4).value or 0,
                    "Yield %": ws.cell(row=row, column=5).value or 0,
                })
        data["weekly_log"] = pd.DataFrame(log_entries) if log_entries else pd.DataFrame()

        wb.close()
        return data
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def fetch_portfolio_history(_ak, _sk, _bu, days=30):
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(_ak, _sk, _bu)
    try:
        h = api.get_portfolio_history(period=f"{days}D", timeframe="1D")
        return pd.DataFrame({"Date": pd.to_datetime(h.timestamp, unit="s"),
                             "Equity": h.equity, "P&L": h.profit_loss, "P&L %": h.profit_loss_pct})
    except Exception:
        return pd.DataFrame()


def load_exit_state():
    try:
        if EXIT_STATE_FILE.exists():
            with open(EXIT_STATE_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_bot_status():
    pid, alive = None, False
    # Method 1: PID file
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            alive = True
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    # Method 2: IPC socket (fallback if PID file is stale/missing)
    if not alive and os.path.exists(IPC_SOCKET):
        try:
            resp = ipc_command({"command": "get_status"}, timeout=3)
            if resp and not resp.get("error"):
                alive = True
                pid = pid or resp.get("pid")
        except Exception:
            pass
    last_log_time = None
    try:
        if BOT_LOG.exists():
            with open(BOT_LOG, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", errors="ignore")
            ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", tail)
            if ts:
                last_log_time = ts[-1]
    except Exception:
        pass
    return {"pid": pid, "alive": alive, "last_log": last_log_time}


def get_recent_logs(n=50):
    try:
        if BOT_LOG.exists():
            with open(BOT_LOG, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode("utf-8", errors="ignore")
            return tail.strip().split("\n")[-n:]
    except Exception:
        pass
    return []


def load_trade_history():
    if not TRADES_DB.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(TRADES_DB))
        df = pd.read_sql_query("SELECT * FROM matched_trades ORDER BY sell_timestamp DESC LIMIT 100", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Bot control helpers
# ---------------------------------------------------------------------------

def _resolve_python() -> str:
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    import sys, shutil
    return shutil.which("python3") or shutil.which("python") or sys.executable


def start_bot():
    bot = get_bot_status()
    if bot["alive"]:
        return "Bot is already running."
    python_exe = _resolve_python()
    subprocess.Popen([python_exe, str(BOT_SCRIPT)], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True,
                     cwd=str(BASE_DIR))
    time.sleep(3)
    bot = get_bot_status()
    return f"Bot started (PID {bot['pid']})" if bot["alive"] else "Failed to start bot. Check logs."


def stop_bot():
    bot = get_bot_status()
    if not bot["alive"]:
        return "Bot is not running."
    try:
        os.kill(bot["pid"], signal.SIGTERM)
        time.sleep(2)
        return f"Bot stopped (was PID {bot['pid']})"
    except Exception as e:
        return f"Error stopping bot: {e}"


def restart_bot():
    stop_bot()
    time.sleep(1)
    return start_bot()


# ---------------------------------------------------------------------------
# AI Chat helpers
# ---------------------------------------------------------------------------

def ask_grok(query: str, cfg: dict) -> str:
    api_key = cfg.get("grok_api_key", "")
    if not api_key:
        return "Error: No Grok API key in config.yaml"
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "grok-4", "messages": [{"role": "user", "content": query}],
                  "temperature": 0.3, "max_tokens": 1000},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


def ask_claude(query: str, cfg: dict) -> str:
    resp = ipc_command({"command": "claude_chat", "query": query, "include_context": True}, timeout=120)
    if resp.get("error"):
        return f"Error: {resp['error']}"
    return resp.get("answer", "No response")


# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    ak, sk, bu = cfg["alpaca"]["api_key"], cfg["alpaca"]["api_secret"], cfg["alpaca"]["base_url"]

    # ---- Ticker Tape (top of page) ----
    all_symbols = cfg.get("tickers", []) + cfg.get("crypto", {}).get("tickers", [])
    if all_symbols:
        components.html(tv_ticker_tape(all_symbols), height=78, scrolling=False)

    # ---- Sidebar ----
    with st.sidebar:
        st.title("AlgoTrader")
        st.caption("Real-time trading dashboard")

        bot = get_bot_status()
        if bot["alive"]:
            st.success(f"Bot Running (PID {bot['pid']})")
        else:
            st.error("Bot Stopped")
        if bot["last_log"]:
            st.caption(f"Last log: {bot['last_log']}")

        # ---- KILL SWITCH (top of sidebar, two-click confirmation) ----
        st.divider()
        st.markdown("### 🚨 Kill Switch")
        if not st.session_state.get("_flatten_armed", False):
            if st.button("ARM Flatten All", use_container_width=True,
                         help="First click arms the kill switch. Second click flattens all positions."):
                st.session_state["_flatten_armed"] = True
                st.rerun()
        else:
            st.warning("Armed. Click below to FLATTEN.")
            kc1, kc2 = st.columns(2)
            with kc1:
                if st.button("FLATTEN", type="primary", use_container_width=True):
                    with st.spinner("Cancelling orders + closing all positions..."):
                        resp = ipc_command({"command": "flatten_all", "halt_after": True}, timeout=60)
                    st.session_state["_flatten_armed"] = False
                    if resp.get("success"):
                        st.success(
                            f"✓ Flattened. Canceled {resp.get('orders_canceled', 0)} orders, "
                            f"closed {resp.get('positions_closed', 0)} positions. "
                            f"Bot halted: {resp.get('trading_halted', False)}"
                        )
                    else:
                        errs = resp.get("errors") or [resp.get("error", "unknown")]
                        st.error(
                            f"Partial flatten. Canceled {resp.get('orders_canceled', 0)} orders, "
                            f"closed {resp.get('positions_closed', 0)} positions. Errors: {'; '.join(map(str, errs))}"
                        )
                    st.cache_data.clear()
            with kc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["_flatten_armed"] = False
                    st.rerun()

        # Drawdown kill switch status (set by position_monitor.check_drawdown_kill_switch)
        if KILL_SWITCH_FILE.exists():
            try:
                with open(KILL_SWITCH_FILE) as f:
                    ks = json.load(f)
                st.error(
                    f"**Drawdown kill switch tripped** — new entries blocked.\n\n"
                    f"Equity ${ks.get('equity', 0):,.2f} vs peak ${ks.get('peak', 0):,.2f} "
                    f"({ks.get('drawdown_pct', 0):.1%}) at {ks.get('tripped_at', '?')}"
                )
                if st.button("Reset Drawdown Kill Switch", use_container_width=True,
                             help="Clears the tripped flag and resets the equity peak to current equity. Bot resumes opening new positions on the next cycle."):
                    resp = ipc_command({"command": "reset_kill_switch"}, timeout=5)
                    if resp.get("success"):
                        st.success("Reset. Bot will resume new entries on the next cycle.")
                        st.rerun()
                    else:
                        st.error(f"Reset failed: {resp.get('error', 'unknown')}")
            except Exception as e:
                st.warning(f"Could not read kill switch status: {e}")

        st.divider()
        st.subheader("Bot Controls")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("Start", use_container_width=True):
                st.toast(start_bot())
                st.cache_data.clear()
                st.rerun()
        with bc2:
            if st.button("Stop", use_container_width=True):
                st.toast(stop_bot())
                st.cache_data.clear()
                st.rerun()
        with bc3:
            if st.button("Restart", use_container_width=True):
                st.toast(restart_bot())
                st.cache_data.clear()
                st.rerun()

        if bot["alive"]:
            if st.button("Force Run Now", use_container_width=True):
                resp = ipc_command({"command": "run_now"})
                st.toast(resp.get("message", resp.get("error", "Sent")))

        st.divider()
        st.subheader("Tickers")
        st.write(", ".join(cfg.get("tickers", [])))
        crypto = cfg.get("crypto", {}).get("tickers", [])
        if crypto:
            st.caption(f"Crypto: {', '.join(crypto)}")

        st.divider()
        st.subheader("Config")
        st.caption(f"Stop Loss: {cfg.get('stop_loss_pct', 0.04):.0%}")
        st.caption(f"Take Profit: {cfg.get('take_profit_pct', 0.08):.0%}")
        st.caption(f"Max Position: {cfg.get('max_position_pct', 15)}%")
        st.caption(f"Buying Power: {cfg.get('buying_power_pct', 90)}%")
        scaled = cfg.get("scaled_exits", {})
        if scaled.get("enabled"):
            tier_parts = []
            for t in scaled.get("tiers", []):
                if t.get("trailing"):
                    tier_parts.append(f"Trail {scaled.get('trail_pct', 0.03):.0%} ({t['sell_fraction']:.0%})")
                elif t.get("pct"):
                    tier_parts.append(f"+{t['pct']:.0%} ({t['sell_fraction']:.0%})")
            st.caption(f"Scaled Exits: {' > '.join(tier_parts)}")

        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ---- Header metrics ----
    try:
        acct = fetch_account(ak, sk, bu)
    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
        st.info("Check your API keys in config.yaml")
        return

    daily_pnl = acct["equity"] - acct["last_equity"]
    daily_pnl_pct = (daily_pnl / acct["last_equity"] * 100) if acct["last_equity"] else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity", f"${acct['equity']:,.2f}", f"{daily_pnl:+.2f}")
    col2.metric("Cash", f"${acct['cash']:,.2f}")
    col3.metric("Buying Power", f"${acct['buying_power']:,.2f}")
    col4.metric("Day P&L", f"${daily_pnl:+,.2f}", f"{daily_pnl_pct:+.1f}%")
    day_trading_enabled = cfg.get('day_trading', {}).get('enabled', False)
    if day_trading_enabled:
        col5.metric("Day Trades", f"{acct['day_trade_count']}", "Unlimited")
    else:
        col5.metric("Day Trades", f"{acct['day_trade_count']}/3", "PDT" if acct["pdt_flag"] else "OK")

    st.divider()

    # ---- Tabs ----
    (tab_portfolio, tab_positions, tab_orders, tab_charts, tab_analysis,
     tab_seer, tab_exits, tab_chat, tab_backtest, tab_logs, tab_income,
     tab_manual, tab_settings) = st.tabs(
        ["Portfolio", "Positions", "Orders", "Charts", "Market",
         "SEER", "Scaled Exits", "AI Chat", "Backtest", "Logs", "Income Portfolio",
         "Manual Trade", "Trading Settings"]
    )

    # ---- Manual Trade Tab (ported from the retired PyQt GUI) ----
    with tab_manual:
        st.markdown("### Manual Trade")
        st.warning("This executes a real market order through the bot service.")
        mt_col1, mt_col2, mt_col3 = st.columns(3)
        mt_ticker = mt_col1.text_input("Ticker", placeholder="e.g., AAPL", key="mt_ticker").strip().upper()
        mt_action = mt_col2.selectbox("Action", ["buy", "sell"], key="mt_action")
        mt_qty = mt_col3.number_input("Quantity", min_value=1, max_value=10000, value=10, step=1, key="mt_qty")
        mt_confirm = st.checkbox(
            f"Confirm: {mt_action.upper()} {int(mt_qty)} share(s) of {mt_ticker or '...'} at market",
            key="mt_confirm",
        )
        if st.button("Execute Trade", type="primary",
                     disabled=not (mt_ticker and mt_confirm), key="mt_execute"):
            resp = ipc_command({
                "command": "manual_trade",
                "ticker": mt_ticker,
                "action": mt_action,
                "quantity": int(mt_qty),
            }, timeout=30)
            if resp.get("error"):
                st.error(f"Trade failed: {resp['error']}")
            elif resp.get("success"):
                st.success(f"{resp.get('message', 'Trade executed')} — Order ID: {resp.get('order_id', 'N/A')}")
            else:
                st.warning("Trade status unclear — check the Positions/Orders tabs.")

    # ---- Portfolio Tab ----
    with tab_portfolio:
        hist = fetch_portfolio_history(ak, sk, bu, days=30)
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Equity"], mode="lines+markers", name="Equity",
                line=dict(color="#00d4aa", width=2), fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
            ))
            fig.update_layout(title="Portfolio Equity (30 Days)", xaxis_title="Date",
                              yaxis_title="Equity ($)", template="plotly_dark", height=400,
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            if "P&L" in hist.columns:
                colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in hist["P&L"]]
                fig2 = go.Figure(go.Bar(x=hist["Date"], y=hist["P&L"], marker_color=colors, name="Daily P&L"))
                fig2.update_layout(title="Daily P&L", template="plotly_dark", height=250,
                                   margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No portfolio history available yet.")

        trades = load_trade_history()
        if not trades.empty:
            st.subheader("Recent Matched Trades (FIFO)")
            display_cols = [c for c in ["ticker", "side", "buy_price", "sell_price", "pnl", "pnl_pct",
                                         "buy_timestamp", "sell_timestamp"] if c in trades.columns]
            if display_cols:
                st.dataframe(trades[display_cols].head(20), use_container_width=True, hide_index=True)

    # ---- Positions Tab ----
    with tab_positions:
        positions = fetch_positions(ak, sk, bu)
        if positions.empty:
            st.info("No open positions.")
        else:
            total_value = positions["Market Value"].sum()
            total_pnl = positions["P&L $"].sum()
            winners = (positions["P&L $"] > 0).sum()
            losers = (positions["P&L $"] < 0).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Value", f"${total_value:,.2f}")
            c2.metric("Unrealized P&L", f"${total_pnl:+,.2f}")
            c3.metric("Winners", f"{winners}")
            c4.metric("Losers", f"{losers}")

            st.dataframe(
                positions.style.applymap(
                    lambda v: "color: #00d4aa" if isinstance(v, (int, float)) and v > 0
                    else "color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else "",
                    subset=["P&L $", "P&L %"],
                ).format({
                    "Entry": "${:.2f}", "Current": "${:.2f}", "Market Value": "${:,.2f}",
                    "P&L $": "${:+,.2f}", "P&L %": "{:+.1f}%", "Change Today %": "{:+.1f}%", "Qty": "{:.4f}",
                }),
                use_container_width=True, hide_index=True,
            )

            # Allocation pie
            fig = px.pie(positions, values="Market Value", names="Symbol", title="Position Allocation",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)

            # TradingView chart for selected position
            selected = st.selectbox("View chart for:", positions["Symbol"].tolist(), key="pos_chart_sym")
            if selected:
                components.html(tv_advanced_chart(selected, height=550), height=570, scrolling=False)

    # ---- Orders Tab ----
    with tab_orders:
        orders = fetch_orders(ak, sk, bu, limit=50)
        if orders.empty:
            st.info("No recent orders.")
        else:
            cf1, cf2 = st.columns(2)
            with cf1:
                status_filter = st.multiselect("Status", options=orders["Status"].unique().tolist(),
                                               default=orders["Status"].unique().tolist())
            with cf2:
                side_filter = st.multiselect("Side", options=orders["Side"].unique().tolist(),
                                             default=orders["Side"].unique().tolist())
            filtered = orders[orders["Status"].isin(status_filter) & orders["Side"].isin(side_filter)]
            st.dataframe(filtered.style.format({"Qty": "{:.2f}", "Filled Qty": "{:.2f}", "Avg Fill": "${:.2f}"}),
                         use_container_width=True, hide_index=True)

    # ---- Charts Tab (TradingView) ----
    with tab_charts:
        chart_sym = st.selectbox("Symbol", all_symbols, key="tv_chart_sym")
        if chart_sym:
            components.html(tv_advanced_chart(chart_sym, height=700), height=720, scrolling=False)

            st.subheader(f"Technical Analysis: {chart_sym}")
            components.html(tv_technical_analysis(chart_sym, height=450), height=470, scrolling=False)

    # ---- Market Tab ----
    with tab_analysis:
        st.subheader("Market Overview")

        st.caption("Indices & Crypto")
        components.html(tv_market_overview(height=500), height=520, scrolling=False)

        st.caption("S&P 500 Heatmap")
        components.html(tv_market_heatmap(height=500), height=520, scrolling=False)

    # ---- SEER Tab (Prediction Market Arbitrage Scanner) ----
    # Updated 2026-04-29 for kalshi_v2 (Seer dashboard on :8501).
    # Schema unchanged; v2 added platform_api.is_live_mode() and a watchlist table.
    with tab_seer:
        SEER_PROJECT_DIR = Path("/Users/shawnslat/Documents/Programming/kalshi_v2")
        seer_db_path = SEER_PROJECT_DIR / "seer.db"

        link_col_a, link_col_b = st.columns([3, 1])
        with link_col_b:
            st.link_button("Open Seer v2 (8501)", "http://localhost:8501",
                           use_container_width=True)

        if not seer_db_path.exists():
            st.warning("SEER database not found. Make sure SEER has been run at least once.")
        else:
            try:
                seer_conn = sqlite3.connect(str(seer_db_path))

                # Helper: defensive table existence check (matches v2/db.py)
                def _seer_table_exists(table: str) -> bool:
                    try:
                        return seer_conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            (table,)
                        ).fetchone() is not None
                    except Exception:
                        return False

                # --- Detect trading mode ---
                # Prefer v2's platform_api.is_live_mode() if importable; fallback to config.py text scan.
                is_live = False
                live_daily_loss_limit = 30.0
                live_bankroll_cap = 200.0
                live_max_trades_hour = 10

                try:
                    _seer_python = "/opt/homebrew/Cellar/python@3.14/3.14.0_1/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python"
                    result = subprocess.run(
                        [_seer_python, "-c",
                         f"import sys; sys.path.insert(0,'{SEER_PROJECT_DIR}/dashboard');"
                         f"sys.path.insert(0,'{SEER_PROJECT_DIR}');"
                         "from platform_api import is_live_mode;"
                         "print('1' if is_live_mode() else '0')"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        is_live = result.stdout.strip().endswith('1')
                except Exception:
                    pass

                seer_config_path = SEER_PROJECT_DIR / "config.py"
                if seer_config_path.exists():
                    try:
                        config_text = seer_config_path.read_text()
                        # Fallback live detection if platform_api wasn't reachable
                        if not is_live and ("PAPER_TRADING_MODE = False" in config_text
                                            or "PAPER_TRADING_MODE=False" in config_text):
                            is_live = True
                        for line in config_text.splitlines():
                            if "LIVE_DAILY_LOSS_LIMIT" in line and "=" in line:
                                try: live_daily_loss_limit = float(line.split("=")[1].split("#")[0].strip())
                                except: pass
                            if "LIVE_BANKROLL_CAP" in line and "=" in line:
                                try: live_bankroll_cap = float(line.split("=")[1].split("#")[0].strip())
                                except: pass
                            if "LIVE_MAX_TRADES_PER_HOUR" in line and "=" in line:
                                try: live_max_trades_hour = int(line.split("=")[1].split("#")[0].strip())
                                except: pass
                    except Exception:
                        pass

                # --- Mode banner ---
                if is_live:
                    st.success("**LIVE TRADING** -- SEER is executing real trades on Kalshi/Polymarket")
                else:
                    st.info("**PAPER TRADING** -- SEER is running in simulation mode")

                # --- Fetch live balances via SEER's Python (needs cryptography + polymarket_us) ---
                kalshi_balance = None
                polymarket_balance = None
                if is_live:
                    try:
                        result = subprocess.run(
                            [_seer_python, "-c",
                             f"import sys; sys.path.insert(0,'{SEER_PROJECT_DIR}');"
                             "from portfolio_manager import get_account_balance;"
                             "from polymarket_executor import get_balance;"
                             "k=get_account_balance(); p=get_balance();"
                             "print(f'{k},{p}')"],
                            capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0:
                            parts = result.stdout.strip().split('\n')[-1].split(',')
                            if len(parts) == 2:
                                kalshi_balance = float(parts[0]) if parts[0] != 'None' else None
                                polymarket_balance = float(parts[1]) if parts[1] != 'None' else None
                    except Exception:
                        pass

                # --- Compute trade stats ---
                # Live mode cutoff: trades after going live (2026-04-04)
                live_cutoff = "2026-04-04"

                if is_live:
                    # Only count trades since going live
                    trade_filter = f"AND timestamp >= '{live_cutoff}'"
                    label_prefix = "Live"
                else:
                    trade_filter = ""
                    label_prefix = "Paper"

                if _seer_table_exists("paper_trades"):
                    pnl_row = seer_conn.execute(
                        f"SELECT SUM(pnl) FROM paper_trades WHERE status='closed' AND pnl IS NOT NULL {trade_filter}"
                    ).fetchone()
                    realized_pnl = pnl_row[0] if pnl_row and pnl_row[0] else 0.0

                    total_row = seer_conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE 1=1 {trade_filter}").fetchone()
                    open_row = seer_conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE status='open' {trade_filter}").fetchone()
                    win_row = seer_conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE win=1 {trade_filter}").fetchone()
                    loss_row = seer_conn.execute(f"SELECT COUNT(*) FROM paper_trades WHERE win=0 {trade_filter}").fetchone()
                else:
                    realized_pnl = 0.0
                    pnl_row = total_row = open_row = win_row = loss_row = None
                    st.info("paper_trades table not yet created in seer.db — run the v2 scanner first.")

                total_trades = total_row[0] if total_row else 0
                open_positions = open_row[0] if open_row else 0
                wins = win_row[0] if win_row else 0
                losses = loss_row[0] if loss_row else 0
                closed_count = wins + losses
                win_rate = (wins / closed_count * 100) if closed_count > 0 else 0

                # --- Daily P&L (trades closed today) ---
                daily_pnl = 0.0
                trades_this_hour = 0
                if _seer_table_exists("paper_trades"):
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    daily_pnl_row = seer_conn.execute(
                        "SELECT SUM(pnl) FROM paper_trades WHERE status='closed' AND resolved_at LIKE ?",
                        (f"{today_str}%",)
                    ).fetchone()
                    daily_pnl = daily_pnl_row[0] if daily_pnl_row and daily_pnl_row[0] else 0.0

                    # --- Trades this hour ---
                    from datetime import timedelta
                    hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                    trades_hour_row = seer_conn.execute(
                        "SELECT COUNT(*) FROM paper_trades WHERE timestamp >= ?", (hour_ago,)
                    ).fetchone()
                    trades_this_hour = trades_hour_row[0] if trades_hour_row else 0

                # --- Header metrics ---
                if is_live:
                    # Live balances (fetched from Kalshi/Polymarket APIs via SEER's Python)
                    bal1, bal2, bal3 = st.columns(3)
                    kalshi_display = f"${kalshi_balance:,.2f}" if kalshi_balance is not None else "N/A"
                    poly_display = f"${polymarket_balance:,.2f}" if polymarket_balance is not None else "N/A"
                    bal1.metric("Kalshi", kalshi_display)
                    bal2.metric("Polymarket", poly_display)
                    combined = 0.0
                    if kalshi_balance is not None:
                        combined += kalshi_balance
                    if polymarket_balance is not None:
                        combined += polymarket_balance
                    if kalshi_balance is not None or polymarket_balance is not None:
                        bal3.metric("Total Capital", f"${combined:,.2f}",
                                    delta=f"${realized_pnl:+,.2f} realized P&L" if realized_pnl != 0 else None)
                    else:
                        bal3.metric("Total Capital", "N/A")
                    st.divider()

                sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                sm1.metric("Daily P&L", f"${daily_pnl:+,.2f}",
                           delta=f"Limit: -${live_daily_loss_limit:.0f}" if is_live else None)
                sm2.metric(f"{label_prefix} Open", f"{open_positions}")
                sm3.metric("Win Rate", f"{win_rate:.0f}%",
                           delta=f"{wins}W / {losses}L" if closed_count > 0 else "No trades yet")
                sm4.metric(f"{label_prefix} Trades", f"{total_trades}")
                sm5.metric("Closed", f"{closed_count}")

                # --- Safety indicators (live mode) ---
                if is_live:
                    safety_col1, safety_col2, safety_col3, safety_col4 = st.columns(4)
                    with safety_col1:
                        daily_loss_pct = abs(daily_pnl) / live_daily_loss_limit * 100 if daily_pnl < 0 else 0
                        if daily_pnl < -live_daily_loss_limit * 0.8:
                            st.error(f"Daily Loss: ${daily_pnl:+,.2f} / -${live_daily_loss_limit:.0f} ({daily_loss_pct:.0f}%)")
                        elif daily_pnl < -live_daily_loss_limit * 0.5:
                            st.warning(f"Daily Loss: ${daily_pnl:+,.2f} / -${live_daily_loss_limit:.0f}")
                        else:
                            st.success(f"Daily P&L: ${daily_pnl:+,.2f} / -${live_daily_loss_limit:.0f} limit")
                    with safety_col2:
                        if trades_this_hour >= live_max_trades_hour * 0.8:
                            st.warning(f"Trades/Hour: {trades_this_hour}/{live_max_trades_hour}")
                        else:
                            st.success(f"Trades/Hour: {trades_this_hour}/{live_max_trades_hour}")
                    with safety_col3:
                        # Open position value (sum of sizes)
                        open_value = 0.0
                        if _seer_table_exists("paper_trades"):
                            open_value_row = seer_conn.execute(
                                "SELECT SUM(size) FROM paper_trades WHERE status='open'"
                            ).fetchone()
                            open_value = open_value_row[0] if open_value_row and open_value_row[0] else 0.0
                        if open_value > live_bankroll_cap * 0.9:
                            st.warning(f"Exposure: ${open_value:,.0f} / ${live_bankroll_cap:.0f} cap")
                        else:
                            st.success(f"Exposure: ${open_value:,.0f} / ${live_bankroll_cap:.0f} cap")
                    with safety_col4:
                        # Kill switch status
                        kill_count = 0
                        if _seer_table_exists("kill_switch_events"):
                            kill_row = seer_conn.execute(
                                "SELECT COUNT(*) FROM kill_switch_events"
                            ).fetchone()
                            kill_count = kill_row[0] if kill_row else 0
                        if kill_count > 0:
                            st.error(f"Kill Switch: {kill_count} triggered")
                        else:
                            st.success("Kill Switch: Clear")

                st.divider()

                # --- P&L chart over time (from closed trades, not paper bankroll) ---
                if _seer_table_exists("metrics"):
                    metrics_hist = pd.read_sql_query(
                        "SELECT timestamp, total_pnl, daily_pnl, open_positions FROM metrics ORDER BY timestamp",
                        seer_conn)
                else:
                    metrics_hist = pd.DataFrame()
                if not metrics_hist.empty:
                    metrics_hist["timestamp"] = pd.to_datetime(metrics_hist["timestamp"])

                    fig_bankroll = go.Figure()
                    fig_bankroll.add_trace(go.Scatter(
                        x=metrics_hist["timestamp"], y=metrics_hist["total_pnl"],
                        mode="lines", name="Realized P&L",
                        line=dict(color="#00d4aa", width=2),
                        fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                    ))
                    fig_bankroll.add_hline(y=0, line_dash="dash", line_color="gray",
                                          annotation_text="Break Even")
                    fig_bankroll.update_layout(
                        title="SEER Realized P&L Over Time", template="plotly_dark", height=350,
                        xaxis_title="", yaxis_title="P&L ($)",
                        margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_bankroll, use_container_width=True)

                # --- Two columns: Open Positions + Opportunity Pipeline ---
                seer_col1, seer_col2 = st.columns(2)

                with seer_col1:
                    if _seer_table_exists("paper_trades"):
                        open_query = ("SELECT market_title, category, size, entry_price, side, close_time, timestamp "
                                      "FROM paper_trades WHERE status='open'")
                        if is_live:
                            open_query += f" AND timestamp >= '{live_cutoff}'"
                        open_query += " ORDER BY timestamp DESC"
                        open_trades = pd.read_sql_query(open_query, seer_conn)
                    else:
                        open_trades = pd.DataFrame()
                    st.subheader(f"Open Positions ({len(open_trades)})")
                    if not open_trades.empty:
                        open_trades["timestamp"] = pd.to_datetime(open_trades["timestamp"]).dt.strftime("%m/%d %H:%M")
                        open_trades["close_time"] = pd.to_datetime(open_trades["close_time"], errors="coerce").dt.strftime("%m/%d")
                        open_trades.columns = ["Market", "Type", "Size $", "Entry", "Side", "Expires", "Opened"]
                        st.dataframe(open_trades.style.format({"Size $": "${:,.2f}", "Entry": "{:.3f}"}),
                                     use_container_width=True, hide_index=True, height=400)
                    else:
                        st.info("No open positions.")

                with seer_col2:
                    if _seer_table_exists("opportunities"):
                        recent_opps = pd.read_sql_query(
                            "SELECT market_title, category, quality_score, ev, market_price, true_prob, timestamp "
                            "FROM opportunities WHERE quality_score >= 6 ORDER BY timestamp DESC LIMIT 25", seer_conn)
                    else:
                        recent_opps = pd.DataFrame()
                    st.subheader("Recent High-Quality Opportunities")
                    if not recent_opps.empty:
                        recent_opps["timestamp"] = pd.to_datetime(recent_opps["timestamp"]).dt.strftime("%m/%d %H:%M")
                        recent_opps["ev"] = (recent_opps["ev"] * 100).round(1)
                        recent_opps.columns = ["Market", "Type", "Score", "EV %", "Price", "True Prob", "Time"]
                        st.dataframe(recent_opps.style.format({"Score": "{:.0f}", "EV %": "{:+.1f}%",
                                                                "Price": "{:.2f}", "True Prob": "{:.2f}"}),
                                     use_container_width=True, hide_index=True, height=400)
                    else:
                        st.info("No recent high-quality opportunities.")

                st.divider()

                # --- Bottom row: Category breakdown + P&L by trade ---
                seer_bot1, seer_bot2 = st.columns(2)

                with seer_bot1:
                    # Trade type breakdown
                    if _seer_table_exists("paper_trades"):
                        type_data = pd.read_sql_query(
                            "SELECT category, COUNT(*) as count, SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_count, "
                            "SUM(CASE WHEN status='closed' THEN pnl ELSE 0 END) as total_pnl "
                            "FROM paper_trades GROUP BY category ORDER BY count DESC", seer_conn)
                    else:
                        type_data = pd.DataFrame()
                    if not type_data.empty:
                        fig_cat = px.pie(type_data, values="count", names="category",
                                         title="Trades by Type",
                                         color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_cat.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(fig_cat, use_container_width=True)

                with seer_bot2:
                    # Cumulative P&L chart from closed trades
                    if _seer_table_exists("paper_trades"):
                        pnl_hist = pd.read_sql_query(
                            "SELECT resolved_at, pnl FROM paper_trades WHERE status='closed' AND pnl IS NOT NULL "
                            "ORDER BY resolved_at", seer_conn)
                    else:
                        pnl_hist = pd.DataFrame()
                    if not pnl_hist.empty:
                        pnl_hist["resolved_at"] = pd.to_datetime(pnl_hist["resolved_at"])
                        pnl_hist["cumulative_pnl"] = pnl_hist["pnl"].cumsum()
                        fig_pnl = go.Figure()
                        fig_pnl.add_trace(go.Scatter(
                            x=pnl_hist["resolved_at"], y=pnl_hist["cumulative_pnl"],
                            mode="lines+markers", name="Cumulative P&L",
                            line=dict(color="#00d4aa", width=2),
                            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                        ))
                        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_pnl.update_layout(
                            title="Cumulative Realized P&L", template="plotly_dark", height=300,
                            yaxis_title="P&L ($)", margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_pnl, use_container_width=True)
                    else:
                        # Fallback: EV distribution
                        if _seer_table_exists("opportunities"):
                            ev_data = pd.read_sql_query(
                                "SELECT ev FROM opportunities WHERE quality_score >= 6 AND ev > 0 "
                                "ORDER BY timestamp DESC LIMIT 200", seer_conn)
                        else:
                            ev_data = pd.DataFrame()
                        if not ev_data.empty:
                            ev_data["ev_pct"] = ev_data["ev"] * 100
                            fig_ev = px.histogram(ev_data, x="ev_pct", nbins=30,
                                                  title="EV Distribution (Recent Opportunities)",
                                                  labels={"ev_pct": "Expected Value %"},
                                                  color_discrete_sequence=["#00d4aa"])
                            fig_ev.update_layout(template="plotly_dark", height=300,
                                                 margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_ev, use_container_width=True)

                # --- Closed trades performance ---
                if _seer_table_exists("paper_trades"):
                    closed_query = ("SELECT market_title, category, size, entry_price, exit_price, pnl, win, side, resolved_at "
                                    "FROM paper_trades WHERE status='closed'")
                    if is_live:
                        closed_query += f" AND timestamp >= '{live_cutoff}'"
                    closed_query += " ORDER BY resolved_at DESC LIMIT 20"
                    closed_trades = pd.read_sql_query(closed_query, seer_conn)
                else:
                    closed_trades = pd.DataFrame()
                if not closed_trades.empty:
                    st.subheader("Recent Closed Trades")
                    closed_trades.columns = ["Market", "Type", "Size $", "Entry", "Exit", "P&L $", "Win", "Side", "Resolved"]
                    st.dataframe(
                        closed_trades.style.applymap(
                            lambda v: "color: #00d4aa" if isinstance(v, (int, float)) and v > 0
                            else "color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else "",
                            subset=["P&L $"],
                        ).format({"Size $": "${:,.2f}", "Entry": "{:.3f}", "Exit": "{:.3f}", "P&L $": "${:+,.2f}"}),
                        use_container_width=True, hide_index=True)

                seer_conn.close()

            except Exception as e:
                st.error(f"Error reading SEER data: {e}")

            st.divider()
            st.link_button("Open Full SEER Dashboard", "http://localhost:8501", use_container_width=True)

    # ---- Scaled Exits Tab ----
    with tab_exits:
        exit_state = load_exit_state()
        if not exit_state:
            st.info("No scaled exit state yet. Tiers appear after the bot enters positions with scaled exits enabled.")
        else:
            tiers_config = cfg.get("scaled_exits", {}).get("tiers", [])
            for symbol, state in exit_state.items():
                tiers_sold = state.get("tiers_sold", [])
                original_qty = state.get("original_qty", 0)
                high_water = state.get("high_water", 0)
                entry = state.get("entry_price", 0)
                with st.expander(f"**{symbol}** -- Entry: ${entry:.2f} | High: ${high_water:.2f} | Qty: {original_qty:.4f}", expanded=True):
                    cols = st.columns(len(tiers_config))
                    for i, (tier, col) in enumerate(zip(tiers_config, cols)):
                        sold = i in tiers_sold
                        pct = tier.get("pct", 0)
                        frac = tier.get("sell_fraction", 0)
                        is_trail = tier.get("trailing", False)
                        label = f"Trail {cfg.get('scaled_exits', {}).get('trail_pct', 0.03):.0%}" if is_trail else f"+{pct:.0%}"
                        with col:
                            st.metric(f"Tier {i+1}: {label}", f"{frac:.0%} of position",
                                      "SOLD" if sold else "Waiting")

    # ---- AI Chat Tab ----
    with tab_chat:
        st.subheader("Ask Claude or Grok")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar=msg.get("avatar")):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask about the market, your portfolio, or trading strategy...")
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            ai_choice = st.session_state.get("ai_provider", "Claude")
            if ai_choice == "Grok":
                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Asking Grok..."):
                        answer = ask_grok(prompt, cfg)
                    st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer, "avatar": "🧠"})
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Asking Claude (with portfolio context)..."):
                        answer = ask_claude(prompt, cfg)
                    st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer, "avatar": "🤖"})

        cc1, cc2, cc3 = st.columns([1, 1, 1])
        with cc1:
            st.radio("AI Provider", ["Claude", "Grok"], horizontal=True, key="ai_provider")
        with cc2:
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
        with cc3:
            st.caption("Claude includes portfolio context. Grok is a fast second opinion.")

        # ---- Ticker Manager ----
        st.markdown("---")
        st.subheader("Ticker Manager")

        current_stocks = cfg.get("tickers", [])
        current_crypto = cfg.get("crypto", {}).get("tickers", [])

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"**Stocks:** {', '.join(current_stocks)}")
        with tc2:
            st.markdown(f"**Crypto:** {', '.join(current_crypto)}")

        # Pending ticker change from auto_ticker_rotator (staged when churn > 50%)
        if PENDING_TICKER_CHANGE.exists():
            try:
                with open(PENDING_TICKER_CHANGE) as f:
                    pending = json.load(f)
            except Exception as e:
                pending = None
                st.error(f"Could not read pending ticker change: {e}")

            if pending:
                st.markdown("---")
                st.warning(
                    f"**Pending watchlist change staged for review** "
                    f"(churn {pending.get('churn_pct', 0):.0%}) — "
                    f"staged {pending.get('staged_at', '?')}"
                )
                reason = pending.get("reason_staged") or pending.get("reasoning")
                if reason:
                    st.caption(reason)

                changes = pending.get("changes", {}) or {}
                pc1, pc2 = st.columns(2)
                with pc1:
                    added_s = changes.get("added_stocks") or []
                    removed_s = changes.get("removed_stocks") or []
                    if added_s:
                        st.markdown(f"**Add stocks:** {', '.join(added_s)}")
                    if removed_s:
                        st.markdown(f"**Remove stocks:** {', '.join(removed_s)}")
                with pc2:
                    added_c = changes.get("added_crypto") or []
                    removed_c = changes.get("removed_crypto") or []
                    if added_c:
                        st.markdown(f"**Add crypto:** {', '.join(added_c)}")
                    if removed_c:
                        st.markdown(f"**Remove crypto:** {', '.join(removed_c)}")

                st.markdown(
                    f"**Proposed stocks:** {', '.join(pending.get('stocks', []))}  \n"
                    f"**Proposed crypto:** {', '.join(pending.get('crypto', []))}"
                )

                ap1, ap2, _ = st.columns([1, 1, 2])
                with ap1:
                    if st.button("Approve & Apply", type="primary", use_container_width=True, key="pending_apply"):
                        try:
                            new_stocks = pending.get("stocks") or current_stocks
                            new_crypto = pending.get("crypto") or current_crypto
                            with open(CONFIG_PATH, "r") as f:
                                config_data = yaml.safe_load(f) or {}
                            config_data["tickers"] = new_stocks
                            config_data.setdefault("crypto", {})["tickers"] = new_crypto
                            tmp_cfg = CONFIG_PATH.with_suffix(".yaml.tmp")
                            with open(tmp_cfg, "w") as f:
                                yaml.dump(config_data, f, default_flow_style=False)
                            os.replace(tmp_cfg, CONFIG_PATH)
                            auto_path = BASE_DIR / "tickers_auto.json"
                            tmp_auto = auto_path.with_suffix(".json.tmp")
                            with open(tmp_auto, "w") as f:
                                json.dump({"tickers": new_stocks}, f)
                            os.replace(tmp_auto, auto_path)
                            PENDING_TICKER_CHANGE.unlink()
                            load_config.clear()
                            st.success("Applied. Bot will pick up new tickers on the next cycle.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to apply: {e}")
                with ap2:
                    if st.button("Reject", use_container_width=True, key="pending_reject"):
                        try:
                            PENDING_TICKER_CHANGE.unlink()
                            st.info("Pending change rejected.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to reject: {e}")
                st.markdown("---")

        # AI Ticker Recommendations
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if st.button("Get Claude Recommendations", use_container_width=True):
                with st.spinner("Asking Claude for ticker recommendations..."):
                    today = datetime.now().strftime("%B %d, %Y")
                    prompt = (
                        f"Today is {today}. You are a trading analyst for an algo bot with a $5000 account.\n"
                        f"Current watchlist: Stocks={current_stocks}, Crypto={current_crypto}\n\n"
                        f"Recommend 6-10 stock tickers and 2-4 crypto pairs for swing trading. "
                        f"Focus on liquid assets with good volatility. Consider sector diversification.\n\n"
                        f"Reply ONLY as JSON: {{\"stocks\": [\"AAPL\", ...], \"crypto\": [\"BTC/USD\", ...], "
                        f"\"reasoning\": \"brief explanation\"}}"
                    )
                    answer = ask_claude(prompt, cfg)
                st.session_state["_ticker_rec"] = answer
                st.session_state["_ticker_rec_source"] = "Claude"
                st.rerun()
        with rec_col2:
            if st.button("Get Grok Recommendations", use_container_width=True):
                with st.spinner("Asking Grok for ticker recommendations..."):
                    today = datetime.now().strftime("%B %d, %Y")
                    prompt = (
                        f"Today is {today}. You are a trading analyst for an algo bot with a $5000 account.\n"
                        f"Current watchlist: Stocks={current_stocks}, Crypto={current_crypto}\n\n"
                        f"Recommend 6-10 stock tickers and 2-4 crypto pairs for swing trading. "
                        f"Focus on liquid assets with good volatility. Consider sector diversification.\n\n"
                        f"Reply ONLY as JSON: {{\"stocks\": [\"AAPL\", ...], \"crypto\": [\"BTC/USD\", ...], "
                        f"\"reasoning\": \"brief explanation\"}}"
                    )
                    answer = ask_grok(prompt, cfg)
                st.session_state["_ticker_rec"] = answer
                st.session_state["_ticker_rec_source"] = "Grok"
                st.rerun()

        # Show recommendations and apply button
        if "_ticker_rec" in st.session_state:
            raw = st.session_state["_ticker_rec"]
            source = st.session_state.get("_ticker_rec_source", "AI")
            st.info(f"**{source} Recommendation:**")

            # Try to parse JSON from response
            new_stocks, new_crypto, reasoning = None, None, None
            try:
                # Try code-fenced JSON first
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                if not json_match:
                    # Find the JSON object containing "stocks" key (most specific match)
                    json_match = re.search(r'(\{[^{}]*"stocks"\s*:\s*\[.*?\].*?\})', raw, re.DOTALL)
                if not json_match:
                    # Last resort: greedy match from first { to last }
                    json_match = re.search(r'(\{.*\})', raw, re.DOTALL)
                if json_match:
                    raw_json = json_match.group(1)
                    # Clean trailing commas
                    cleaned = re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', raw_json))
                    rec = json.loads(cleaned)
                    new_stocks = [t.upper().strip() for t in rec.get("stocks", []) if t.strip()]
                    new_crypto = [t.strip() for t in rec.get("crypto", []) if t.strip()]
                    reasoning = rec.get("reasoning", "")
            except Exception as e:
                st.warning(f"Couldn't parse ticker JSON: {e}")

            if new_stocks or new_crypto:
                if reasoning:
                    st.markdown(f"*{reasoning}*")
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    st.markdown(f"**Stocks:** {', '.join(new_stocks or [])}")
                with rcol2:
                    st.markdown(f"**Crypto:** {', '.join(new_crypto or [])}")

                if st.button("Apply These Tickers", type="primary", use_container_width=True):
                    try:
                        with open(CONFIG_PATH, "r") as f:
                            config_data = yaml.safe_load(f)
                        if new_stocks:
                            config_data["tickers"] = new_stocks
                        if new_crypto:
                            config_data.setdefault("crypto", {})["tickers"] = new_crypto
                        with open(CONFIG_PATH, "w") as f:
                            yaml.dump(config_data, f, default_flow_style=False)
                        # Also update tickers_auto.json
                        auto_path = BASE_DIR / "tickers_auto.json"
                        with open(auto_path, "w") as f:
                            json.dump({"tickers": new_stocks or current_stocks}, f)
                        del st.session_state["_ticker_rec"]
                        # Clear config cache so dashboard shows new tickers immediately
                        load_config.clear()
                        st.success("Tickers updated! Bot will use new tickers on the next trading cycle.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update: {e}")
            else:
                st.markdown(raw)

        # Manual ticker editing
        st.markdown("---")
        with st.expander("Manual Ticker Edit"):
            new_stock_str = st.text_input("Stocks (comma-separated)", value=", ".join(current_stocks))
            new_crypto_str = st.text_input("Crypto (comma-separated)", value=", ".join(current_crypto))
            if st.button("Save Tickers"):
                try:
                    with open(CONFIG_PATH, "r") as f:
                        config_data = yaml.safe_load(f)
                    stocks = [t.strip().upper() for t in new_stock_str.split(",") if t.strip()]
                    cryptos = [t.strip() for t in new_crypto_str.split(",") if t.strip()]
                    config_data["tickers"] = stocks
                    config_data.setdefault("crypto", {})["tickers"] = cryptos
                    with open(CONFIG_PATH, "w") as f:
                        yaml.dump(config_data, f, default_flow_style=False)
                    auto_path = BASE_DIR / "tickers_auto.json"
                    with open(auto_path, "w") as f:
                        json.dump({"tickers": stocks}, f)
                    load_config.clear()
                    st.success("Tickers saved! Bot will use new tickers on the next trading cycle.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    # ---- Backtest Tab ----
    with tab_backtest:
        st.subheader("Train Model & Backtest")
        bot_status = get_bot_status()
        if not bot_status["alive"]:
            st.warning("Bot must be running to trigger backtest (it uses the bot_service IPC).")
        else:
            st.info("Triggers the full pipeline: download data, engineer features, train model, and backtest. Takes 2-5 minutes.")
            if st.button("Run Train & Backtest", type="primary", use_container_width=True):
                resp = ipc_command({"command": "run_backtest"}, timeout=10)
                if resp.get("error"):
                    st.error(resp["error"])
                else:
                    st.success(resp.get("message", "Backtest started"))

            bt_status = ipc_command({"command": "get_backtest_status"}, timeout=5)
            if bt_status.get("running"):
                phase = bt_status.get("phase", "unknown")
                progress = bt_status.get("progress")
                st.info(f"Backtest running: **{phase}**")
                if progress is not None:
                    try:
                        st.progress(min(1.0, max(0.0, float(progress))))
                    except (TypeError, ValueError):
                        st.caption(str(progress))
            elif bt_status.get("last_run"):
                st.success(f"Last backtest: {bt_status['last_run']}")
                if bt_status.get("last_error"):
                    st.error(f"Error: {bt_status['last_error']}")

            results_path = BASE_DIR / "artifacts" / "backtest_results.json"
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        results = json.load(f)
                    st.subheader("Last Backtest Results")
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Total Return", f"{results.get('total_return_pct', 0):.1f}%")
                    rc2.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
                    rc3.metric("Win Rate", f"{results.get('win_rate', 0):.1f}%")
                    rc4.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.1f}%")
                except Exception:
                    pass

    # ---- Logs Tab ----
    with tab_logs:
        logs = get_recent_logs(80)
        if logs:
            log_filter = st.text_input("Filter logs", placeholder="e.g. ERROR, SELL, confidence, Grok...")
            if log_filter:
                logs = [l for l in logs if log_filter.lower() in l.lower()]
            st.code("\n".join(logs), language="log")
        else:
            st.info("No bot logs found.")


    # ---- Income Portfolio Tab ----
    with tab_income:
        ip = load_income_tracker()
        if ip is None:
            st.warning("Income Portfolio Tracker not found. Place Income_Portfolio_Tracker.xlsx in income_portfolio/")
        elif "error" in ip:
            st.error(f"Error loading tracker: {ip['error']}")
        else:
            st.markdown("### Income Portfolio Dashboard")
            st.caption(f"Target: $2,000/month passive income | Tracker: income_portfolio/Income_Portfolio_Tracker.xlsx")

            # Top metrics row
            ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)
            ic1.metric("Portfolio Value", f"${ip['portfolio_total']:,.0f}")
            ic2.metric("Invested", f"${ip['portfolio_invested']:,.0f}")
            ic3.metric("Cash (SWVXX)", f"${ip['cash']:,.0f}")
            ic4.metric("Monthly Income", f"${ip['monthly_income']:,.2f}")
            ic5.metric("Portfolio Yield", f"{ip['portfolio_yield']:.1%}" if isinstance(ip['portfolio_yield'], (int, float)) and ip['portfolio_yield'] > 0 else "0.0%")
            pct = ip['income_achievement']
            ic6.metric("Goal Progress", f"{pct:.1%}" if isinstance(pct, (int, float)) and pct > 0 else "0.0%")

            # Income gap bar
            goal = ip['monthly_goal'] or 2000
            current = ip['monthly_income'] or 0
            progress = min(current / goal, 1.0) if goal > 0 else 0
            st.progress(progress, text=f"${current:,.2f} / ${goal:,.0f} per month ({progress:.1%})")

            st.divider()

            col_left, col_right = st.columns(2)

            with col_left:
                # Allocation drift
                st.markdown("#### Allocation & Drift")
                alloc_df = ip["allocation"]
                if not alloc_df.empty:
                    # Color-coded drift chart
                    fig_alloc = go.Figure()
                    colors = []
                    for _, row in alloc_df.iterrows():
                        d = row["Drift"]
                        if isinstance(d, (int, float)):
                            if abs(d) <= 0.03:
                                colors.append("#00d4aa")
                            elif abs(d) <= 0.07:
                                colors.append("#ffa726")
                            else:
                                colors.append("#ef5350")
                        else:
                            colors.append("#888")

                    fig_alloc.add_trace(go.Bar(
                        x=alloc_df["Bucket"], y=[a if isinstance(a, (int,float)) else 0 for a in alloc_df["Actual"]],
                        name="Actual", marker_color=colors, text=[f"{a:.1%}" if isinstance(a,(int,float)) else "0%" for a in alloc_df["Actual"]],
                        textposition="auto",
                    ))
                    fig_alloc.add_trace(go.Scatter(
                        x=alloc_df["Bucket"], y=[t if isinstance(t,(int,float)) else 0 for t in alloc_df["Target"]],
                        name="Target", mode="markers+lines", marker=dict(size=12, color="white", symbol="diamond"),
                        line=dict(color="white", dash="dash"),
                    ))
                    fig_alloc.update_layout(
                        height=300, template="plotly_dark",
                        yaxis=dict(title="Allocation %", tickformat=".0%"),
                        legend=dict(orientation="h", y=1.15),
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True)

                    for _, row in alloc_df.iterrows():
                        status = row["Status"]
                        drift = row["Drift"]
                        drift_str = f"{drift:+.1%}" if isinstance(drift, (int, float)) else "0%"
                        if "SEVERELY" in str(status):
                            st.error(f"**{row['Bucket']}**: {drift_str} — {status}")
                        elif "OVER" in str(status) or "UNDER" in str(status):
                            st.warning(f"**{row['Bucket']}**: {drift_str} — {status}")
                        else:
                            st.success(f"**{row['Bucket']}**: {drift_str} — {status}")

            with col_right:
                # Income concentration & risk
                st.markdown("#### Concentration & Risk")
                risk_items = {
                    "Top 1 Income %": ip["top1_income"],
                    "Top 3 Income %": ip["top3_income"],
                    "Positions": ip["num_positions"],
                    "Deployable Cash": ip["deployable_cash"],
                }
                for label, val in risk_items.items():
                    if "%" in label and isinstance(val, (int, float)):
                        display = f"{val:.1%}"
                        if label == "Top 3 Income %" and val > 0.4:
                            st.error(f"**{label}**: {display} (limit: 40%)")
                        elif label == "Top 1 Income %" and val > 0.15:
                            st.warning(f"**{label}**: {display} (limit: 15%)")
                        else:
                            st.info(f"**{label}**: {display}")
                    elif label == "Deployable Cash":
                        st.info(f"**{label}**: ${val:,.0f}" if isinstance(val, (int, float)) else f"**{label}**: $0")
                    else:
                        st.info(f"**{label}**: {val}")

                # Macro signals
                st.markdown("#### Macro Signals")
                macro_cols = st.columns(4)
                macro_cols[0].metric("VIX", f"{ip['vix']:.0f}" if isinstance(ip['vix'], (int, float)) else "—")
                macro_cols[1].metric("2yr", f"{ip['treasury_2y']:.2%}" if isinstance(ip['treasury_2y'], (int, float)) else "—")
                macro_cols[2].metric("10yr", f"{ip['treasury_10y']:.2%}" if isinstance(ip['treasury_10y'], (int, float)) else "—")
                macro_cols[3].metric("HY OAS", f"{ip['hy_oas']:.2%}" if isinstance(ip['hy_oas'], (int, float)) else "—")

            st.divider()

            # Holdings table
            st.markdown("#### Holdings")
            hdf = ip["holdings"]
            if not hdf.empty:
                display_cols = ["Ticker", "Name", "Bucket", "Shares", "Price", "Market Value",
                                "Yield %", "Monthly Income", "% Portfolio", "Income Type"]
                show_df = hdf[display_cols].copy()
                show_df["Price"] = show_df["Price"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) else "$0")
                show_df["Market Value"] = show_df["Market Value"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) else "$0")
                show_df["Yield %"] = show_df["Yield %"].apply(lambda x: f"{x:.1%}" if isinstance(x, (int,float)) else "0%")
                show_df["Monthly Income"] = show_df["Monthly Income"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) else "$0")
                show_df["% Portfolio"] = show_df["% Portfolio"].apply(lambda x: f"{x:.1%}" if isinstance(x, (int,float)) else "0%")
                st.dataframe(show_df, use_container_width=True, hide_index=True)
            else:
                st.info("No holdings entered yet. Add positions in Income_Portfolio_Tracker.xlsx → Holdings sheet.")

            # Time to goal
            st.markdown("#### Time-to-Goal Scenarios")
            ttg_df = ip["ttg"]
            if not ttg_df.empty:
                display_ttg = ttg_df.copy()
                display_ttg["Monthly Savings"] = display_ttg["Monthly Savings"].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int,float)) else x)
                display_ttg["Annual Deploy"] = display_ttg["Annual Deploy"].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int,float)) else x)
                display_ttg["Yrs (8% Yield)"] = display_ttg["Yrs (8% Yield)"].apply(lambda x: f"{x:.1f}" if isinstance(x, (int,float)) else x)
                st.dataframe(display_ttg, use_container_width=True, hide_index=True)

            # Weekly log trend chart
            wlog = ip["weekly_log"]
            if not wlog.empty:
                st.markdown("#### Weekly Trend")
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=wlog["Date"], y=wlog["Portfolio Value"], mode="lines+markers",
                    name="Portfolio Value", line=dict(color="#00d4aa", width=2),
                ))
                fig_trend.update_layout(height=250, template="plotly_dark",
                    margin=dict(l=40, r=20, t=30, b=30))
                st.plotly_chart(fig_trend, use_container_width=True)

            if st.button("Refresh Income Data", key="refresh_income"):
                st.cache_data.clear()
                st.rerun()


    # ---- Trading Settings Tab ----
    with tab_settings:
        st.markdown("### Trading Settings")
        st.caption("Adjust trading frequency, risk, and confidence thresholds. Changes take effect on the next trading cycle.")

        with open(CONFIG_PATH) as f:
            current_cfg = yaml.safe_load(f)

        st.markdown("#### Schedule")
        sched = current_cfg.get("schedule", {})
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            enable_morning = st.checkbox(
                "Enable Morning Run",
                value=sched.get("enable_morning_run", True),
                help="Morning run catches overnight gap opportunities.",
            )
        with sc2:
            morning_run = st.text_input(
                "Morning Run Time (ET)",
                value=str(sched.get("morning_run", "10:15")),
                help="Format: HH:MM (24-hour). Window: 9:55-10:30 AM ET.",
            )
        with sc3:
            afternoon_run = st.text_input(
                "Afternoon Run Time (ET)",
                value=str(sched.get("afternoon_run", "15:00")),
                help="Format: HH:MM (24-hour). Window: 2:55-3:30 PM ET.",
            )

        st.markdown("#### Day Trading (PDT Rule)")
        dt_cfg = current_cfg.get("day_trading", {})
        dc1, dc2 = st.columns(2)
        with dc1:
            day_trading_on = st.checkbox(
                "Enable Unlimited Day Trading",
                value=dt_cfg.get("enabled", False),
                help="SEC eliminated the PDT rule on Apr 14, 2026. When enabled, the bot can execute unlimited same-day round trips.",
            )
        with dc2:
            max_day_trades = st.number_input(
                "Max Day Trades / Day (when disabled)",
                min_value=1, max_value=999,
                value=int(dt_cfg.get("max_day_trades", 999)),
                help="Only used when day trading is disabled. Default 3 = legacy PDT limit.",
            )

        st.markdown("#### Confidence Thresholds")
        conf_cfg = current_cfg.get("confidence", {})
        cc1, cc2 = st.columns(2)
        with cc1:
            min_conf = st.slider(
                "Minimum Confidence to Trade",
                min_value=0.30, max_value=0.95,
                value=float(conf_cfg.get("min_confidence_to_trade", 0.55)),
                step=0.05,
                help="Lower = more trades but lower-quality signals. Higher = fewer, higher-conviction trades.",
            )
        with cc2:
            full_conf = st.slider(
                "Full Position Confidence Threshold",
                min_value=0.50, max_value=0.95,
                value=float(conf_cfg.get("full_confidence_threshold", 0.70)),
                step=0.05,
                help="Above this confidence, full position size is used. Below, position is scaled down.",
            )

        st.markdown("#### Position Sizing & Risk")
        rs1, rs2, rs3 = st.columns(3)
        with rs1:
            risk_per_trade = st.slider(
                "Risk Per Trade (%)",
                min_value=0.25, max_value=5.0,
                value=float(current_cfg.get("risk_per_trade_pct", 1.0)),
                step=0.25,
                help="% of available cash risked per trade (used to size positions vs. ATR).",
            )
        with rs2:
            max_position = st.slider(
                "Max Position (% of equity)",
                min_value=1.0, max_value=50.0,
                value=float(current_cfg.get("max_position_pct", 15.0)),
                step=1.0,
                help="Hard cap on any single position as % of total equity.",
            )
        with rs3:
            buying_power_pct = st.slider(
                "Buying Power Cap (%)",
                min_value=10, max_value=100,
                value=int(current_cfg.get("buying_power_pct", 90)),
                step=5,
                help="% of Alpaca buying power the bot is allowed to use per cycle.",
            )

        st.markdown("#### Kelly Criterion Sizing")
        kelly_cfg_current = current_cfg.get("kelly_sizing", {}) or {}
        kc1, kc2 = st.columns(2)
        with kc1:
            kelly_enabled_input = st.checkbox(
                "Enable Kelly Sizing",
                value=bool(kelly_cfg_current.get("enabled", False)),
                help="Sizes positions based on Kelly criterion: f = p/SL - (1-p)/TP. Replaces the coarse 0/50%/100% confidence multiplier with a smooth edge-based size.",
            )
            kelly_fraction_input = st.slider(
                "Kelly Fraction",
                min_value=0.05, max_value=1.0,
                value=float(kelly_cfg_current.get("fraction", 0.25)),
                step=0.05,
                help="Fractional Kelly. 0.25 = quarter-Kelly (recommended). 1.0 = full Kelly (theoretically optimal but very volatile in practice).",
            )
        with kc2:
            kelly_min_edge_input = st.slider(
                "Minimum Edge (EV per $1)",
                min_value=0.0, max_value=0.20,
                value=float(kelly_cfg_current.get("min_edge", 0.05)),
                step=0.01, format="%.2f",
                help="Minimum expected return per dollar to take a trade: p*TP - (1-p)*SL >= min_edge. 0.05 = require 5%+ EV per dollar bet. Higher = more selective.",
            )
            kelly_max_fraction_input = st.slider(
                "Max % Equity at Risk Per Trade",
                min_value=0.01, max_value=0.50,
                value=float(kelly_cfg_current.get("max_fraction", 0.10)),
                step=0.01, format="%.2f",
                help="Hard cap on Kelly bet size as a fraction of total equity. Prevents extreme bets even if Kelly says to go bigger.",
            )

        st.markdown("#### Concurrent Trades")
        max_concurrent = st.slider(
            "Max Concurrent Tickers Traded Per Cycle",
            min_value=1, max_value=15,
            value=int(current_cfg.get("max_concurrent_trades", 5)),
            step=1,
            help="Bot concentrates into the top N ranked tickers each cycle.",
        )

        st.markdown("#### Crypto Allocation Cap")
        crypto_alloc_cap = st.slider(
            "Max % of Equity in Crypto (combined BTC/ETH/SOL/...)",
            min_value=0.0, max_value=100.0,
            value=float(current_cfg.get("crypto", {}).get("allocation_cap_pct", 50.0)),
            step=5.0, format="%.0f%%",
            help="Bot will refuse new crypto buys that would push total crypto position value above this % of equity. Set to 100% to disable. Sells are unaffected.",
        )

        st.markdown("#### Stop Loss & Take Profit")
        sl1, sl2 = st.columns(2)
        with sl1:
            stop_loss_pct_display = st.slider(
                "Default Stop Loss (%)",
                min_value=1.0, max_value=20.0,
                value=float(current_cfg.get("stop_loss_pct", 0.06)) * 100,
                step=0.5, format="%.1f%%",
                help="Base stop loss as % of entry price. ATR adjustment may widen this in volatile conditions.",
            )
            stop_loss = stop_loss_pct_display / 100.0
        with sl2:
            take_profit_pct_display = st.slider(
                "Default Take Profit (%)",
                min_value=1.0, max_value=40.0,
                value=max(float(current_cfg.get("take_profit_pct", 0.12)) * 100, 1.0),
                step=0.5, format="%.1f%%",
                help="Base take profit as % of entry price. Should be at least 2x stop loss for a 1:2 R:R ratio. Bot enforces 2:1 minimum dynamically.",
            )
            take_profit = take_profit_pct_display / 100.0

        rr_ratio = take_profit_pct_display / stop_loss_pct_display if stop_loss_pct_display > 0 else 0
        if rr_ratio < 1.5:
            st.warning(f"Risk:Reward = 1:{rr_ratio:.2f} — below recommended 1:2. Consider raising take profit or lowering stop loss.")
        elif rr_ratio < 2.0:
            st.info(f"Risk:Reward = 1:{rr_ratio:.2f} — acceptable but below 1:2 best practice.")
        else:
            st.success(f"Risk:Reward = 1:{rr_ratio:.2f} — meets 1:2 minimum.")

        st.divider()

        def _normalize_hhmm(value: str, default: str) -> str:
            """Force HH:MM format that the `schedule` library accepts (it rejects '9:45')."""
            try:
                parts = str(value).strip().split(":")
                hh, mm = int(parts[0]), int(parts[1])
                if 0 <= hh < 24 and 0 <= mm < 60:
                    return f"{hh:02d}:{mm:02d}"
            except (ValueError, AttributeError, IndexError):
                pass
            return default

        save_col, restart_col = st.columns([1, 1])
        with save_col:
            if st.button("Save Settings", type="primary", use_container_width=True):
                try:
                    current_cfg.setdefault("schedule", {})
                    current_cfg["schedule"]["enable_morning_run"] = enable_morning
                    current_cfg["schedule"]["morning_run"] = _normalize_hhmm(morning_run, "10:15")
                    current_cfg["schedule"]["afternoon_run"] = _normalize_hhmm(afternoon_run, "15:00")

                    current_cfg.setdefault("day_trading", {})
                    current_cfg["day_trading"]["enabled"] = day_trading_on
                    current_cfg["day_trading"]["max_day_trades"] = int(max_day_trades)

                    current_cfg.setdefault("confidence", {})
                    current_cfg["confidence"]["min_confidence_to_trade"] = float(min_conf)
                    current_cfg["confidence"]["full_confidence_threshold"] = float(full_conf)

                    current_cfg["risk_per_trade_pct"] = float(risk_per_trade)
                    current_cfg["max_position_pct"] = float(max_position)
                    current_cfg["buying_power_pct"] = int(buying_power_pct)
                    current_cfg["max_concurrent_trades"] = int(max_concurrent)
                    current_cfg["stop_loss_pct"] = float(stop_loss)
                    current_cfg["take_profit_pct"] = float(take_profit)

                    current_cfg.setdefault("crypto", {})
                    current_cfg["crypto"]["allocation_cap_pct"] = float(crypto_alloc_cap)

                    current_cfg.setdefault("kelly_sizing", {})
                    current_cfg["kelly_sizing"]["enabled"] = bool(kelly_enabled_input)
                    current_cfg["kelly_sizing"]["fraction"] = float(kelly_fraction_input)
                    current_cfg["kelly_sizing"]["min_edge"] = float(kelly_min_edge_input)
                    current_cfg["kelly_sizing"]["max_fraction"] = float(kelly_max_fraction_input)

                    with open(CONFIG_PATH, "w") as f:
                        yaml.dump(current_cfg, f, default_flow_style=False)
                    load_config.clear()
                    st.success("Settings saved! Restart the bot for schedule changes to take effect.")
                except Exception as e:
                    st.error(f"Failed to save settings: {e}")
        with restart_col:
            if st.button("Restart Bot to Apply", use_container_width=True):
                st.toast(restart_bot())
                st.cache_data.clear()
                st.rerun()


if __name__ == "__main__":
    main()
