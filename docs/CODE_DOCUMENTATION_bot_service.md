# Code Documentation: bot_service.py

**Purpose:** Background service wrapper for the trading bot with GUI integration
**File:** `bot_service.py`
**Lines:** 966
**Dependencies:** Multiple (see imports section)

---

## Overview

`bot_service.py` is the bridge between the GUI and the core trading logic. It wraps `Trader_main_Grok4_20250731.py` in a non-blocking service architecture that can:
- Run trading bot in background 24/7
- Respond to GUI commands via IPC
- Schedule daily trading at 3:55 PM ET
- Provide real-time status updates
- Handle backtest/training requests
- Integrate with Dexter AI for ticker research and bias generation

**Architecture Pattern:** Service Wrapper
- **Core bot:** Trader_main_Grok4_20250731.py (trading logic)
- **Service layer:** bot_service.py (scheduling, GUI integration)
- **GUI layer:** launch_gui_proper.py (user interface)

---

## Imports & Dependencies

### Lines 1-52: Import Section

**Core Python:**
- `os`, `sys`, `threading`, `time`: System operations
- `datetime`, `timedelta`, `pytz`: Timezone-aware timestamps
- `subprocess`, `shlex`: External command execution (Dexter)
- `logging`: Log management

**Trading Bot Core:**
```python
from Trader_main_Grok4_20250731 import (
    load_configuration,              # YAML config loader
    initialize_alpaca_api,           # Alpaca REST API client
    fetch_current_market_data,       # Historical data from Polygon
    fetch_latest_data,               # Real-time Alpaca data
    engineer_features,               # Technical indicators
    generate_signals,                # ML + Q-Learning signals
    execute_trading_logic_live,      # Place orders via Alpaca
    fetch_current_positions,         # Get open positions
    load_q_table,                    # Reinforcement learning table
    download_historical_data,        # Polygon batch download
    load_historical_data,            # CSV loader
    add_sentiment_features,          # NewsAPI + Grok sentiment
    prepare_train_test_data,         # ML dataset prep
    tune_and_train_model,            # XGBoost training
    evaluate_model,                  # Accuracy metrics
    backtest_strategy,               # Historical backtest
    maybe_override_tickers_from_json,# Dexter ticker override
    maybe_fetch_tickers_via_dexter,  # Dexter ticker research
)
```

**Supporting Modules:**
- `pdt_guard.DayTradeGuard`: PDT protection
- `dexter_gate.DexterGate`: AI sentiment gate
- `ipc_protocol.IPCServer, LogStreamer`: GUI communication
- `joblib`: Model serialization
- `schedule`: Job scheduling library

**Programming Notes:**
- All imports from `Trader_main_*` are functions (no classes)
- Heavy reliance on trading bot core (service is thin wrapper)
- IPC for inter-process communication
- Logging to `logs/bot_service.log`

---

## Class: `TradingBotService`

**Purpose:** Main service class that orchestrates bot operations

### Lines 62-111: Constructor `__init__(self, config_path: str = "config.yaml")`

**What it initializes:**

**Core Components:**
```python
self.config = None           # Loaded from config.yaml
self.api = None              # Alpaca REST API client
self.model = None            # XGBoost model (loaded from artifacts/)
self.q_table = None          # Q-Learning state-action table
self.selected_features = []  # Feature names for ML model
self.guard = None            # PDT protection (DayTradeGuard)
self.dexter_gate = None      # AI sentiment filter (DexterGate)
```

**Runtime State:**
```python
self.running = False         # Bot scheduled?
self.bot_thread = None       # Background trading loop thread
self.raw_historical = None   # Cached historical data
self.last_execution_time = None     # When trading last ran
self.next_execution_time = None     # Next scheduled execution
self.trade_count = 0         # Trades executed this session
```

**Signals State:**
```python
self.current_signals = []              # Latest ML/RL signals
self.last_signals_refresh = None       # When signals computed
self.signals_refresh_running = False   # Background refresh active?
self.signals_refresh_error = None      # Refresh error message
```

**Backtest State:**
```python
self.backtest_running = False          # Backtest in progress?
self.backtest_last_run = None          # When last backtest completed
self.backtest_last_error = None        # Backtest error message
self.backtest_phase = "idle"           # Current backtest step
self.backtest_started_at = None        # Backtest start time
self.backtest_last_update = None       # Last progress update
self.backtest_progress = None          # Progress description
self._backtest_thread = None           # Background backtest thread
```

**Dexter State:**
```python
self.dexter_last_answer = None         # Last Dexter chat response
self.dexter_last_query = None          # Last Dexter chat query
self.dexter_update_running = False     # Ticker research active?
self.dexter_update_error = None        # Ticker research error
self.dexter_update_last = None         # Last ticker research time
self.dexter_bias_running = False       # Bias generation active?
self.dexter_bias_error = None          # Bias generation error
self.dexter_bias_last = None           # Last bias generation time
```

**Service State:**
```python
self.status = "stopped"      # stopped, idle, running, trading, error
self.error_message = None    # Error description
self.last_heartbeat = None   # Heartbeat timestamp (1/sec)
```

**IPC & Threading:**
```python
self.ipc_server = None       # Unix socket server for GUI
self.log_streamer = LogStreamer()   # Log publish/subscribe
self.state_lock = threading.RLock() # Protects state variables
self.job_lock = threading.Lock()    # Prevents concurrent trading
self.heartbeat_thread = None        # 1-second heartbeat thread
```

**Programming Notes:**
- Many state variables for GUI observability
- `RLock` (reentrant lock) allows same thread to acquire multiple times
- Separate locks for state vs job execution (reduces contention)
- All state changes go through locks (thread-safe)

---

### Lines 112-180: Method `initialize(self) -> bool`

**What it does:**
Initializes all bot components (config, API, guards, model, IPC server)

**Process:**
1. Load `config.yaml`
2. Override tickers from `tickers_auto.json` if exists (Dexter)
3. Initialize Alpaca API client
4. Initialize PDT guard (max 2 day trades/day)
5. Initialize Dexter gate (load `dexter_bias.json`)
6. Load model from `artifacts/final_model.pkl`
7. Load Q-table from `artifacts/q_table.csv`
8. Set feature list (14-16 features depending on TA-Lib availability)
9. Start IPC server on `/tmp/trader_bot.sock`
10. Start heartbeat thread (updates `last_heartbeat` every second)
11. Start Dexter autofetch (optional background ticker research)
12. Set status to "idle"

**Returns:**
- `True`: Initialization successful
- `False`: Error occurred (sets `status = "error"`)

**Error Handling:**
```python
if not model_path.exists():
    logger.warning("No model found - bot needs training")
    self.status = "error"
    self.error_message = "Model not found. Please train the model first."
    return False
```

**Programming Notes:**
- Checks for model file existence (critical dependency)
- TA-Lib detection adds optional features (Momentum, SMA_20)
- Dexter ticker research runs in background (non-blocking)
- IPC server must start successfully (GUI depends on it)
- Heartbeat provides liveness signal for GUI

---

### Lines 182-231: Method `_handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]`

**What it does:**
Routes incoming GUI commands to appropriate handler methods

**Supported Commands:**

| Command | Handler | Purpose |
|---------|---------|---------|
| `start` | `start_trading()` | Start scheduled bot |
| `stop` | `stop_trading()` | Stop scheduled bot |
| `run_now` | `run_now()` | Trigger immediate execution |
| `get_status` | `get_status()` | Get bot state |
| `get_positions` | `get_positions()` | Get open positions |
| `get_signals` | `get_signals()` | Get latest signals |
| `refresh_signals` | `refresh_signals()` | Recompute signals |
| `get_signals_status` | `get_signals_status()` | Get refresh progress |
| `get_account` | `get_account_info()` | Get account balance |
| `run_backtest` | `run_backtest()` | Start backtest job |
| `get_backtest_status` | `get_backtest_status()` | Get backtest progress |
| `dexter_chat` | `dexter_chat()` | AI chat query |
| `dexter_update_tickers` | `dexter_update_tickers()` | Research tickers |
| `dexter_generate_bias` | `dexter_generate_bias()` | Generate bias file |
| `get_dexter_status` | `get_dexter_status()` | Get Dexter state |
| `manual_trade` | `execute_manual_trade()` | Place manual order |
| `refresh_dexter` | `dexter_gate.refresh()` | Reload bias file |

**Programming Notes:**
- Central dispatcher pattern (single entry point)
- Try-except catches handler errors (returns `{"error": "..."}`)
- Unknown commands return error (no silent failures)
- Thread-safe (handlers use locks internally)

---

### Lines 233-308: Method `dexter_chat(self, query: str, include_context: bool = False) -> Dict[str, Any]`

**What it does:**
Executes external Dexter AI chat command and returns answer

**Configuration:**
```yaml
# config.yaml or environment variable
dexter_chat_command: "python scripts/dexter_chat.py"
dexter_chat_timeout_seconds: 120
```

**Process:**
1. Check query is valid (non-empty string)
2. Get command from env `DEXTER_CHAT_CMD` or config
3. Optionally inject context (account, positions, signals)
4. Set environment variable `DEXTER_QUERY` to full query
5. Append query as shell argument
6. Execute command with timeout
7. Parse JSON response: `{"answer": "..."}`
8. Store query and answer in state
9. Return answer to GUI

**Context Injection (if enabled):**
```json
{
  "account": {"equity": 100000, "cash": 50000, ...},
  "positions": [{"ticker": "AAPL", "qty": 10, ...}],
  "signals": [{"ticker": "NVDA", "signal": "buy", ...}]
}
```

**Response Format:**
```json
{
  "success": true,
  "answer": "Based on your portfolio..."
}
```

**Error Handling:**
- Missing query → `{"error": "Missing query."}`
- Not configured → `{"error": "Dexter chat not configured."}`
- Command failed → `{"error": "Dexter command failed (exit 1): ..."}`
- Empty output → `{"error": "Dexter returned empty output."}`
- Timeout → `subprocess.TimeoutExpired` caught

**Programming Notes:**
- `shlex.quote()` prevents command injection
- Command runs in separate process (subprocess.run)
- Timeout prevents hanging (configurable, default 120s)
- JSON parsing flexible (falls back to plain text if not JSON)
- Context injection lightweight (uses existing data)

---

### Lines 310-319: Method `get_dexter_status(self) -> Dict[str, Any]`

**What it does:**
Returns status of Dexter background jobs

**Returns:**
```python
{
  "tickers_update_running": False,     # Ticker research active?
  "tickers_update_last": "2025-12-17T14:30:00",  # Last research time
  "tickers_update_error": None,        # Research error
  "bias_running": False,               # Bias generation active?
  "bias_last": "2025-12-17T15:00:00",  # Last bias time
  "bias_error": None                   # Bias error
}
```

**Programming Notes:**
- Thread-safe read (state_lock)
- Used by GUI to show Dexter status

---

### Lines 321-458: Dexter Ticker Research & Bias Generation

**Methods:**
- `dexter_update_tickers()` → Starts ticker research (background)
- `_dexter_update_tickers_job()` → Actually runs research
- `dexter_generate_bias()` → Starts bias generation (background)
- `_dexter_bias_job()` → Actually generates bias
- `_start_dexter_autofetch()` → Optional startup ticker research
- `_dexter_autofetch_job()` → Runs ticker command

**Ticker Research Flow:**
```
GUI clicks "Update Tickers"
    ↓
dexter_update_tickers() called
    ↓
Sets dexter_update_running = True
    ↓
Spawns background thread → _dexter_update_tickers_job()
    ↓
Runs command (e.g., python dexter_research.py)
    ↓
Command returns JSON: {"tickers": ["AAPL", "MSFT", "NVDA"]}
    ↓
Cleans/deduplicates tickers
    ↓
Writes tickers_auto.json
    ↓
Reloads config to apply new tickers
    ↓
Sets dexter_update_last timestamp
    ↓
Sets dexter_update_running = False
```

**Bias Generation Flow:**
```
GUI clicks "Generate Bias"
    ↓
dexter_generate_bias() called
    ↓
Sets dexter_bias_running = True
    ↓
Spawns background thread → _dexter_bias_job()
    ↓
Runs command (e.g., python dexter_bias.py)
    ↓
Command analyzes news, RSI, MACD for each ticker
    ↓
Returns JSON: {"AAPL": "allow", "TSLA": {"decision": "avoid", ...}}
    ↓
Writes dexter_bias.json
    ↓
Calls dexter_gate.refresh() to reload file
    ↓
Sets dexter_bias_last timestamp
    ↓
Sets dexter_bias_running = False
```

**Programming Notes:**
- Both operations run in daemon threads (don't block GUI)
- Timeout protection (45s for tickers, 90s for bias)
- Error handling stores error message for GUI display
- State flags prevent concurrent execution
- Thread-safe state updates with locks
- Best-effort operations (failures don't crash service)

---

### Lines 460-486: Trading Control Methods

**Method: `start_trading() -> Dict[str, Any]`**

**What it does:**
Starts the scheduled trading bot (3:55 PM daily execution)

**Process:**
1. Check if already running → return error
2. Check if status is "error" → return error
3. Set `running = True`
4. Set status to "running"
5. Spawn background thread → `_trading_loop()`
6. Return success

**Returns:**
```python
{"success": True, "message": "Bot started successfully"}
```

**Programming Notes:**
- Thread-safe (state_lock)
- `daemon=True` thread (dies when main process exits)
- Trading loop runs independently

---

**Method: `stop_trading() -> Dict[str, Any]`**

**What it does:**
Stops the scheduled trading bot

**Process:**
1. Check if not running → return error
2. Set `running = False`
3. Set status to "stopped"
4. Return success

**Programming Notes:**
- Trading loop checks `self.running` every minute
- Loop exits when flag becomes False
- Doesn't kill thread (graceful shutdown)

---

**Method: `run_now() -> Dict[str, Any]`**

**What it does:**
Triggers immediate trading execution (bypasses schedule)

**Process:**
1. Check if bot running → return error if not
2. Spawn thread → `_execute_cycle(force=True)`
3. Return success immediately (non-blocking)

**Returns:**
```python
{"success": True, "message": "Triggered run-now"}
```

**Programming Notes:**
- `force=True` bypasses time window check
- Runs in separate thread (non-blocking)
- Can run outside 3:50-4:00 PM window
- Useful for testing

---

### Lines 496-505: Heartbeat Methods

**Method: `_start_heartbeat(self)`**

**What it does:**
Starts 1-second heartbeat thread

**Method: `_heartbeat_loop(self)`**

**What it does:**
Updates `last_heartbeat` every second

**Programming Notes:**
- Infinite loop (daemon thread)
- GUI uses this to detect if service crashed
- Updates timestamp as ISO string

---

### Lines 507-593: Core Trading Cycle

**Method: `_execute_cycle(self, force: bool = False)`**

**What it does:**
Executes a single trading cycle (fetch data → engineer features → generate signals → place orders)

**Process:**

**1. Time Window Check (unless force=True):**
```python
execution_start = 15:50  # 3:50 PM ET
execution_end = 16:00    # 4:00 PM ET
if not force and not (start <= now <= end and weekday < 5):
    skip cycle
```

**2. Fetch Historical Data (first run only):**
```python
if self.raw_historical is None:
    self.raw_historical = fetch_current_market_data(tickers, polygon_key)
```

**3. Fetch Latest Data:**
```python
latest_data = fetch_latest_data(tickers, alpaca_api)
```

**4. Merge Historical + Latest:**
```python
self.raw_historical = concat([historical, latest])
self.raw_historical = groupby('ticker').tail(200)  # Keep last 200 bars
```

**5. Engineer Features:**
```python
engineered_data = engineer_features(self.raw_historical)
# Adds: MA10, MA50, RSI, MACD, Bollinger Bands, ATR, etc.
```

**6. Fetch Current Positions:**
```python
positions_snapshot = {p['ticker']: p for p in fetch_current_positions(api)}
```

**7. Generate Signals:**
```python
signals_df = generate_signals(
    model,                    # XGBoost classifier
    engineered_data,
    features + ['Sentiment_Score'],
    config,
    q_table,                 # Q-Learning table
    positions_snapshot=positions_snapshot
)
```

**8. Store Signals for GUI:**
```python
latest_signals = signals_df.groupby('ticker').tail(1)
self.current_signals = self._format_signals_for_gui(latest_signals)
```

**9. Execute Trades:**
```python
actionable = latest_signals[signals_df['Signal'].isin([1, -1])]  # Buy or Sell
if not actionable.empty:
    execute_trading_logic_live(
        api,
        actionable,
        config,
        q_table,
        guard,           # PDT protection
        dexter_gate,     # AI sentiment filter
        buying_power_pct=100
    )
```

**10. Update Execution Time:**
```python
self.last_execution_time = current_time
```

**Programming Notes:**
- `job_lock` prevents concurrent cycles (one at a time)
- Data caching (historical data fetched once, updated incrementally)
- Tail(200) keeps memory bounded (only last 200 bars per ticker)
- Status changes: idle → trading → idle (or error)
- Errors caught and logged (doesn't crash service)

---

### Lines 595-599: Helper Method `_set_status(self, status: str, error: str | None = None)`

**What it does:**
Thread-safe status update

**Programming Notes:**
- Acquires state_lock before modification
- Optionally sets error message

---

### Lines 601-648: Trading Loop & Scheduling

**Method: `_trading_loop(self)`**

**What it does:**
Main background loop that schedules and executes daily trading

**Process:**

**1. Initial Execution Check:**
```python
if execution_start <= now <= execution_end and weekday < 5:
    # Within window at startup → run immediately
    self._execute_cycle(force=False)
```

**2. Schedule Daily Execution:**
```python
schedule.every().day.at("15:55").do(job)
```

**3. Main Loop:**
```python
while self.running:
    schedule.run_pending()  # Run jobs if scheduled time reached
    self._update_next_execution_time()
    time.sleep(60)  # Check every minute
```

**Scheduling Library:**
- Uses `schedule` Python library
- Cron-like syntax: `.every().day.at("15:55")`
- `run_pending()` executes jobs if time reached
- Checks every 60 seconds (1 minute granularity)

**Programming Notes:**
- Runs in daemon thread (background)
- `while self.running` allows graceful shutdown
- `sleep(60)` reduces CPU usage (don't poll every second)
- Next execution time updated every minute (for GUI display)

---

**Method: `_update_next_execution_time(self)`**

**What it does:**
Calculates and stores next scheduled execution time

**Logic:**
```python
next_run = schedule.next_run()
if next_run:
    self.next_execution_time = next_run.strftime("%Y-%m-%d %H:%M:%S")
else:
    # No pending jobs → calculate tomorrow 3:55 PM
    tomorrow = now + timedelta(days=1)
    self.next_execution_time = tomorrow.replace(hour=15, minute=55)
```

**Programming Notes:**
- Used by GUI to show "Next execution: 2025-12-17 15:55:00"
- Handles edge cases (job just ran, no jobs scheduled)

---

### Lines 649-660: Method `get_status() -> Dict[str, Any]`

**What it does:**
Returns comprehensive bot status for GUI

**Returns:**
```python
{
  "status": "running",           # stopped, idle, running, trading, error
  "running": True,               # Bot scheduled?
  "next_execution": "2025-12-17 15:55:00",
  "last_execution": "2025-12-17 15:55:05",
  "trade_count": 3,              # Trades this session
  "last_heartbeat": "2025-12-17 15:56:30",
  "error": None                  # Error message if status=error
}
```

**Programming Notes:**
- Thread-safe read (state_lock)
- Called every 5 seconds by GUI
- Drives GUI icon color and menu states

---

### Lines 662-686: Method `get_positions() -> Dict[str, Any]`

**What it does:**
Fetches current open positions from Alpaca with P&L calculations

**Returns:**
```python
{
  "positions": [
    {
      "ticker": "AAPL",
      "qty": 10.0,
      "avg_entry_price": 250.50,
      "current_price": 255.75,
      "market_value": 2557.50,
      "unrealized_pl": 52.50,         # Dollar P&L
      "unrealized_plpc": 2.09         # Percentage P&L
    }
  ]
}
```

**Calculations:**
```python
market_value = qty * current_price
unrealized_pl = (current_price - avg_entry_price) * qty
unrealized_plpc = ((current_price - avg_entry_price) / avg_entry_price) * 100
```

**Programming Notes:**
- Calls `fetch_current_positions(api)` from Trader_main
- Enriches data with P&L calculations
- Returns `{"error": "..."}` on failure

---

### Lines 688-762: Signal Refresh Methods

**Method: `get_signals() -> Dict[str, Any]`**

**What it does:**
Returns cached current signals

**Returns:**
```python
{"signals": self.current_signals}
```

---

**Method: `refresh_signals() -> Dict[str, Any]`**

**What it does:**
Starts background job to recompute signals (non-blocking)

**Process:**
1. Check if refresh already running → return error
2. Set `signals_refresh_running = True`
3. Spawn thread → `_signals_refresh_job()`
4. Return success immediately

---

**Method: `get_signals_status() -> Dict[str, Any]`**

**What it does:**
Returns refresh job status

**Returns:**
```python
{
  "running": False,
  "last_refresh": "2025-12-17T15:55:30",
  "error": None
}
```

---

**Method: `_signals_refresh_job(self)`**

**What it does:**
Background job that recomputes signals without executing trades

**Process:**
1. Check model loaded → error if not
2. Reload config/tickers (may have changed)
3. Fetch market data
4. Engineer features
5. Add sentiment features
6. Fetch current positions
7. Generate signals (ML + Q-Learning)
8. Format signals for GUI
9. Store in `self.current_signals`
10. Update `last_signals_refresh` timestamp

**Programming Notes:**
- Full signal computation (same as trading cycle)
- Does NOT execute trades (read-only)
- Error stored in `signals_refresh_error`
- Finally block clears `signals_refresh_running` flag

---

### Lines 764-860: Backtest Methods

**Method: `run_backtest() -> Dict[str, Any]`**

**What it does:**
Starts background backtest job (download data → train model → evaluate)

**Process:**
1. Check if backtest already running → return error
2. Set backtest state flags
3. Spawn thread → `_backtest_job()`
4. Return success immediately

---

**Method: `get_backtest_status() -> Dict[str, Any]`**

**What it does:**
Returns backtest progress

**Returns:**
```python
{
  "running": True,
  "last_run": "2025-12-17T14:00:00",
  "error": None,
  "phase": "training_model",         # Current step
  "started_at": "2025-12-17T14:00:00",
  "last_update": "2025-12-17T14:05:30",
  "progress": "Training XGBoost (epoch 50/150)",
  "thread_alive": True
}
```

---

**Method: `_backtest_job(self)`**

**What it does:**
Full backtest pipeline (runs in background, 5-10 minutes)

**Steps:**

**1. Load Config & Tickers:**
```python
self._set_backtest_progress("loading_config", "Loading configuration")
config = load_configuration(config_path)
maybe_fetch_tickers_via_dexter(config)
config = maybe_override_tickers_from_json(config)
```

**2. Download Historical Data:**
```python
self._set_backtest_progress("downloading", "Downloading historical data")
download_historical_data(tickers, polygon_key, days_back=730)
```

**3. Load Data:**
```python
self._set_backtest_progress("loading_data", "Loading historical data")
data = load_historical_data('data', config)
```

**4. Engineer Features:**
```python
self._set_backtest_progress("engineering_features", "Engineering features")
data = engineer_features(data)
```

**5. Add Sentiment:**
```python
self._set_backtest_progress("sentiment", "Adding sentiment features")
data = add_sentiment_features(data, config)
```

**6. Prepare Train/Test Split:**
```python
self._set_backtest_progress("preparing", "Preparing train/test data")
X_train, X_test, y_train, y_test = prepare_train_test_data(data, features)
```

**7. Train Model:**
```python
self._set_backtest_progress("training", "Training model")
model = tune_and_train_model(X_train, y_train, X_test, y_test)
```

**8. Evaluate:**
```python
self._set_backtest_progress("evaluating", "Evaluating model")
evaluate_model(model, X_test, y_test, features)
```

**9. Backtest Strategy:**
```python
self._set_backtest_progress("backtesting", "Running backtest")
backtest_strategy(model, data, config, q_table, features)
```

**10. Save Model:**
```python
self._set_backtest_progress("saving", "Saving model")
joblib.dump(model, 'artifacts/final_model.pkl')
```

**11. Complete:**
```python
self._set_backtest_progress("completed", "Backtest completed")
self.backtest_last_run = datetime.now().isoformat()
```

**Programming Notes:**
- Long-running (5-10 minutes typical)
- Progress updates every step (GUI shows progress bar)
- Errors stored in `backtest_last_error`
- Finally block clears `backtest_running` flag
- Model reloaded into service after training

---

### Lines 861-922: Manual Trading & Account Info

**Method: `execute_manual_trade(self, ticker, action, quantity) -> Dict[str, Any]`**

**What it does:**
Places manual order (bypass ML/RL signals)

**Parameters:**
- `ticker`: Stock symbol (e.g., "AAPL")
- `action`: "buy" or "sell"
- `quantity`: Number of shares

**Process:**
1. Validate inputs (non-empty, positive quantity)
2. Get current price from Alpaca
3. Calculate order value
4. Check available cash (for buys) or position (for sells)
5. Place market order via Alpaca
6. Return order details

**Programming Notes:**
- Bypasses PDT guard (user override)
- Bypasses Dexter gate (user override)
- Market orders (immediate execution at current price)
- Returns `{"success": true, "order_id": "..."}` or error

---

**Method: `get_account_info() -> Dict[str, Any]`**

**What it does:**
Fetches Alpaca account details

**Returns:**
```python
{
  "equity": 105000.00,      # Total account value
  "cash": 50000.00,         # Available cash
  "buying_power": 50000.00, # Max buying power
  "portfolio_value": 105000.00,
  "last_equity": 104500.00, # Previous day close
  "long_market_value": 55000.00,
  "short_market_value": 0.00,
  "pattern_day_trader": False,
  "daytrade_count": 0,      # Rolling 5-day count
  "daytrading_buying_power": 50000.00
}
```

**Programming Notes:**
- Calls `api.get_account()` from Alpaca REST API
- Returns all account fields
- Used by dashboard window

---

### Lines 923-966: Helper Methods

**Method: `_format_signals_for_gui(self, signals_df) -> List[Dict[str, Any]]`**

**What it does:**
Converts signals DataFrame to GUI-friendly format

**Input (DataFrame):**
```
ticker  Signal  Confidence  RSI   date
AAPL    1       0.75       45    2025-12-17
MSFT    0       0.60       52    2025-12-17
NVDA    -1      0.80       68    2025-12-17
```

**Output (List of Dicts):**
```python
[
  {
    "ticker": "AAPL",
    "signal": "buy",      # 1 → "buy", -1 → "sell", 0 → "hold"
    "confidence": 0.75,
    "rsi": 45.0,
    "date": "2025-12-17"
  },
  ...
]
```

**Programming Notes:**
- Converts numeric signals to human-readable strings
- Extracts key fields for GUI display
- Handles NaN values gracefully

---

**Method: `_set_backtest_progress(self, phase, description)`**

**What it does:**
Updates backtest progress for GUI

**Programming Notes:**
- Thread-safe (state_lock)
- Updates `backtest_phase`, `backtest_progress`, `backtest_last_update`
- Logs progress to console

---

## Threading Architecture

```
Main Thread (bot_service.py)
├── IPCServer accept thread (handles GUI connections)
│   └── Client handler threads (one per GUI request)
├── Heartbeat thread (updates every second)
├── Trading loop thread (checks schedule every minute)
│   └── _execute_cycle() (runs trading logic)
├── Signals refresh thread (background signal computation)
├── Backtest thread (long-running training job)
├── Dexter ticker thread (background ticker research)
└── Dexter bias thread (background bias generation)
```

**Thread Safety:**
- `state_lock` (RLock): Protects all state variables
- `job_lock` (Lock): Prevents concurrent trading cycles
- Daemon threads: All background threads are daemons (die with main process)

---

## State Machine Diagram

```
[Initialize]
    ↓
[idle] ←─────────────┐
    ↓                │
[start_trading()]    │
    ↓                │
[running] ←──────────┤
    ↓                │
[3:55 PM reached]    │
    ↓                │
[trading] ───────────┤
    ↓                │
[cycle complete] ────┘
    ↓
[error] (if exception)
```

**Status Values:**
- `stopped`: Bot not scheduled
- `idle`: Service running, bot not scheduled
- `running`: Bot scheduled, waiting for 3:55 PM
- `trading`: Actively executing trades
- `error`: Fatal error, requires restart

---

## Configuration File Integration

**config.yaml Fields Used:**

```yaml
alpaca:
  api_key: "..."
  api_secret: "..."
  base_url: "https://paper-api.alpaca.markets"

polygon:
  api_key: "..."

tickers: [AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL]
max_day_trades: 2
buying_power_pct: 100

dexter_autofetch: true
dexter_ticker_command: "python scripts/dexter_research.py"
dexter_ticker_timeout_seconds: 45
dexter_bias_command: "python scripts/dexter_bias.py"
dexter_bias_timeout_seconds: 90
dexter_chat_command: "python scripts/dexter_chat.py"
dexter_chat_timeout_seconds: 120
```

---

## File Dependencies

**Reads:**
- `config.yaml`: Main configuration
- `artifacts/final_model.pkl`: Trained XGBoost model
- `artifacts/q_table.csv`: Q-Learning table
- `tickers_auto.json`: Dexter-generated tickers
- `dexter_bias.json`: Dexter-generated bias

**Writes:**
- `logs/bot_service.log`: Service logs
- `logs/trade_logs/*.csv`: Trade execution logs
- `artifacts/last_decisions.csv`: Latest signals
- `tickers_auto.json`: Updated tickers (if Dexter enabled)
- `dexter_bias.json`: Updated bias (if generated)

---

## Summary

`bot_service.py` is the orchestration layer that:
- ✅ Wraps core trading bot in service architecture
- ✅ Schedules daily execution at 3:55 PM ET
- ✅ Provides non-blocking IPC for GUI control
- ✅ Manages background jobs (backtest, signals, Dexter)
- ✅ Thread-safe state management
- ✅ Graceful error handling (doesn't crash on exceptions)
- ✅ Real-time status updates for GUI
- ✅ Integrates PDT guard and Dexter gate
- ✅ Supports manual overrides (run-now, manual trades)

**Key Design Patterns:**
- **Service Wrapper:** Thin layer over core bot
- **Command Pattern:** IPC command dispatcher
- **Observer Pattern:** Log streamer pub/sub
- **State Machine:** Explicit status transitions
- **Background Jobs:** Daemon threads for long operations

**Used by:** GUI (via IPC), command-line scripts
**Requires:** Trader_main_Grok4_20250731.py, config.yaml, trained model
