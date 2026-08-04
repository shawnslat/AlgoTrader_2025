# Code Documentation: dexter_gate.py

**Purpose:** Sentiment-based trading gate that blocks trades based on AI analysis
**File:** `dexter_gate.py`
**Lines:** 42
**Dependencies:** `json`, `pathlib`, `typing`

---

## Overview

DexterGate is a decision filter that blocks trades on tickers flagged as "avoid" by an external AI system (Dexter). It reads a JSON bias file that contains AI-generated sentiment/analysis and prevents the bot from trading stocks/crypto that Dexter determines are risky.

**Key Concept:** This is a **veto system** - Dexter can say "no" to trades, but doesn't generate buy signals (those come from ML model + Q-Learning).

---

## Class: `DexterGate`

**Purpose:** Load and enforce AI-generated trading biases from external analysis

**Key Attributes:**
- `bias_path` (Path): Path to JSON file containing Dexter's decisions
- `bias` (Dict): Loaded bias data mapping ticker → decision

---

### Constructor: `__init__(self, bias_path: str = "dexter_bias.json")`

**Lines:** 12-14

**What it does:**
1. Sets path to bias file (defaults to "dexter_bias.json" in current directory)
2. Immediately loads bias data on initialization

**Parameters:**
- `bias_path`: Path to JSON file with Dexter decisions (default: "dexter_bias.json")

**Programming Notes:**
- Converts string path to `Path` object for cross-platform compatibility
- Calls `_load_bias()` immediately so bias is available right away
- Path is stored as instance variable for refresh operations

**Expected File Structure:**
```json
{
  "AAPL": {"decision": "allow", "confidence": 0.8},
  "TSLA": "avoid",
  "NVDA": {"decision": "avoid", "reasoning": "Overbought RSI"}
}
```

---

### Method: `_load_bias(self) -> Dict[str, Any]`

**Lines:** 16-22

**What it does:**
Safely loads bias data from JSON file, returns empty dict if file missing or invalid

**Returns:**
- Dictionary mapping ticker → bias data
- Empty dict `{}` if file doesn't exist or has parse errors

**Process:**
1. Check if bias file exists
2. If exists: read file as text
3. Parse JSON
4. If any error: return empty dict (fail-safe)
5. If no file: return empty dict

**Programming Notes:**
- Uses `Path.exists()` to check file before reading
- `Path.read_text()` reads entire file as string
- Bare `except Exception:` catches ALL errors (file read errors, JSON syntax errors, etc.)
- **Fail-safe design:** If anything goes wrong, returns `{}` and allows all trades
- Private method (leading underscore) - internal use only

**Error Handling:**
```python
# Missing file → returns {}
# Invalid JSON → returns {}
# File read permission error → returns {}
# All trades proceed normally (no bias applied)
```

---

### Method: `refresh(self)`

**Lines:** 24-25

**What it does:**
Reloads bias data from file (useful when Dexter updates the file)

**Process:**
- Calls `_load_bias()` and overwrites `self.bias`

**Programming Notes:**
- No return value (modifies state in-place)
- Can be called after Dexter generates new analysis
- Allows dynamic bias updates without restarting bot

**Usage:**
```python
# Dexter analyzes market and updates dexter_bias.json
run_dexter_analysis()

# Bot refreshes gate to use new decisions
gate.refresh()

# Now has latest "avoid" list
```

---

### Method: `should_allow(self, ticker: str, trades_remaining: int, context: Dict[str, Any]) -> bool`

**Lines:** 27-41

**What it does:**
Main decision function - determines if a trade should be allowed based on:
1. Remaining day trade capacity
2. Dexter's bias for the ticker

**Parameters:**
- `ticker`: Stock/crypto symbol (e.g., "AAPL", "BTCUSD")
- `trades_remaining`: How many day trades are left today (from PDT guard)
- `context`: Additional data (currently unused, reserved for future expansion)

**Returns:**
- `True`: Trade allowed (proceed)
- `False`: Trade blocked (skip)

**Decision Logic Flow:**

```
1. Check trades_remaining
   If <= 0 → return False (PDT limit hit)

2. Get Dexter bias for ticker
   If no bias data → return True (allow by default)

3. Check bias format (string or dict):

   String format:
   If bias == "avoid" → return False

   Dict format:
   If bias["decision"] == "avoid" → return False
   If bias["allow"] == False → return False

4. Default → return True (allow trade)
```

**Programming Notes:**
- **Fail-open design:** If no bias data, defaults to allowing trades
- Handles two bias formats (string or dict) for flexibility
- Case-insensitive "avoid" check (`.lower()`)
- `.strip()` removes whitespace from string bias
- `bias.get("decision", "")` safely accesses dict, returns "" if missing

---

## Bias File Formats

### Format 1: Simple String
```json
{
  "AAPL": "avoid",
  "MSFT": "allow",
  "NVDA": "avoid"
}
```
**How it works:**
- `should_allow("AAPL", ...)` checks if `bias["AAPL"] == "avoid"`
- Blocks AAPL and NVDA, allows MSFT

---

### Format 2: Detailed Dict
```json
{
  "AAPL": {
    "decision": "allow",
    "confidence": 0.75,
    "reasoning": "Strong earnings beat, positive sentiment"
  },
  "TSLA": {
    "decision": "avoid",
    "confidence": 0.92,
    "reasoning": "Overbought RSI, negative news cycle"
  },
  "NVDA": {
    "allow": false,
    "reason": "High volatility expected"
  }
}
```
**How it works:**
- Checks `bias["TSLA"]["decision"] == "avoid"` → blocks
- Also checks `bias["NVDA"]["allow"] == False` → blocks
- Allows AAPL (decision = "allow")

---

### Format 3: Mixed (Supported)
```json
{
  "AAPL": "allow",
  "TSLA": {"decision": "avoid", "confidence": 0.9},
  "NVDA": {"allow": false}
}
```
**How it works:**
- AAPL: String "allow" → passes
- TSLA: Dict with decision="avoid" → blocked
- NVDA: Dict with allow=false → blocked

---

## Integration with Trading Bot

### In `Trader_main_Grok4_20250731.py`:

```python
from dexter_gate import DexterGate
from pdt_guard import DayTradeGuard

# Initialize gates
guard = DayTradeGuard(max_day_trades=2)
dexter_gate = DexterGate("dexter_bias.json")

# Before placing trade
ticker = "AAPL"
trades_remaining = guard.remaining_today(datetime.now())

if dexter_gate.should_allow(ticker, trades_remaining, {}):
    # Passed both PDT and Dexter checks
    place_buy_order(ticker)
else:
    logger.warning(f"Trade blocked by Dexter gate: {ticker}")
```

---

### In `bot_service.py`:

```python
def execute_trading_logic_live():
    for signal in signals:
        ticker = signal['ticker']

        # Check Dexter gate
        dexter_gate.refresh()  # Get latest bias
        if not dexter_gate.should_allow(ticker, trades_remaining, context):
            logger.info(f"Dexter gate blocked trade on {ticker}")
            continue

        # Proceed with trade
        place_order(ticker)
```

---

## How Dexter Generates Bias

**External Process (not in this file):**

1. **News Analysis:** Dexter fetches news articles for each ticker
2. **Sentiment Scoring:** Uses AI (xAI Grok or similar) to analyze sentiment
3. **Technical Check:** Reviews RSI, MACD, trend strength
4. **Decision:** Outputs "allow" or "avoid" with reasoning
5. **Write JSON:** Saves to `dexter_bias.json`

**Bot reads this file via DexterGate before trading.**

---

## Why "Gate" Instead of "Signal"?

**Design Philosophy:**

❌ **Dexter does NOT generate buy signals**
- ML model + Q-Learning generate signals
- Dexter only vetoes bad ones

✅ **Dexter is a risk filter**
- Blocks trades on stocks with bad news
- Blocks trades on overbought/oversold extremes
- Allows model to trade normally otherwise

**Analogy:**
- **ML Model:** Accelerator pedal (says when to trade)
- **Dexter Gate:** Brake pedal (says when NOT to trade)

---

## Fail-Safe Design

**What happens if:**

| Scenario | Result | Why Safe? |
|----------|--------|-----------|
| `dexter_bias.json` missing | Allow all trades | File not required, bot works standalone |
| JSON parse error | Allow all trades | Bad data ignored, doesn't crash bot |
| Ticker not in bias file | Allow trade | Unknown tickers pass through |
| `trades_remaining = 0` | Block trade | PDT protection overrides everything |
| Empty file `{}` | Allow all trades | No bias = no restrictions |

**Key principle:** **Fail-open** (errors allow trades) NOT fail-closed (errors block trades)

**Reason:** Better to miss Dexter's advice than to stop trading entirely due to a file error.

---

## Advanced Features (Not Implemented)

**Confidence Thresholds:**
```python
def should_allow(self, ticker, trades_remaining, context):
    # ... existing checks ...

    if isinstance(bias, dict):
        confidence = bias.get("confidence", 0)
        # Only block if Dexter is highly confident
        if bias.get("decision") == "avoid" and confidence > 0.75:
            return False

    return True
```

**Time-Based Expiry:**
```json
{
  "AAPL": {
    "decision": "avoid",
    "expires": "2025-12-17T16:00:00",
    "reasoning": "Avoid until after earnings call"
  }
}
```

**Weighted Decisions:**
```python
# Dexter says "avoid" but model is very confident
if dexter_says_avoid and model_confidence > 0.9:
    # Override Dexter's caution
    return True
```

---

## Thread Safety

**Current implementation:** Thread-safe for reads, NOT for writes

**Safe operations:**
```python
# Multiple threads can call this simultaneously
result = gate.should_allow("AAPL", 2, {})
```

**Unsafe operations:**
```python
# Race condition if two threads refresh simultaneously
gate.refresh()  # Thread 1
gate.refresh()  # Thread 2
# Both read file at same time, might corrupt bias dict
```

**If using multiple threads:**
```python
import threading

class DexterGate:
    def __init__(self, bias_path: str = "dexter_bias.json"):
        self.lock = threading.RLock()
        # ... rest of init

    def refresh(self):
        with self.lock:
            self.bias = self._load_bias()

    def should_allow(self, ticker, trades_remaining, context):
        with self.lock:
            # ... existing logic
```

---

## Testing Examples

```python
from dexter_gate import DexterGate

# Create test bias file
import json
bias_data = {
    "AAPL": "allow",
    "TSLA": "avoid",
    "NVDA": {"decision": "avoid", "confidence": 0.9}
}
with open("test_bias.json", "w") as f:
    json.dump(bias_data, f)

# Initialize gate
gate = DexterGate("test_bias.json")

# Test decisions
print(gate.should_allow("AAPL", 2, {}))  # True (allow)
print(gate.should_allow("TSLA", 2, {}))  # False (avoid - string)
print(gate.should_allow("NVDA", 2, {}))  # False (avoid - dict)
print(gate.should_allow("MSFT", 2, {}))  # True (not in bias)
print(gate.should_allow("AAPL", 0, {}))  # False (no trades remaining)

# Update bias and refresh
bias_data["AAPL"] = "avoid"
with open("test_bias.json", "w") as f:
    json.dump(bias_data, f)
gate.refresh()
print(gate.should_allow("AAPL", 2, {}))  # False (now avoided)
```

---

## Configuration in `config.yaml`

```yaml
# Dexter settings
dexter:
  enabled: true
  bias_file: "dexter_bias.json"
  auto_refresh: true  # Refresh before each trading cycle

  # Command to regenerate bias (called by bot_service.py)
  generate_command: "python scripts/dexter_generate_bias.py"
```

---

## Real-World Example

**Scenario:** Market crash on Dec 17, 2025

**Without DexterGate:**
```
ML model sees TSLA oversold (RSI = 25)
→ Generates BUY signal
→ Bot buys TSLA
→ TSLA drops another 10% (catching falling knife)
```

**With DexterGate:**
```
Dexter analyzes news: "Tesla factory fire, SEC investigation, analyst downgrades"
→ Writes: {"TSLA": {"decision": "avoid", "reasoning": "Multiple negative catalysts"}}
→ ML model sees TSLA oversold (RSI = 25)
→ Generates BUY signal
→ DexterGate blocks: should_allow("TSLA", ...) returns False
→ Bot skips trade
→ TSLA drops another 10% (avoided loss)
```

---

## Summary

`dexter_gate.py` is a simple but powerful risk filter that:
- ✅ Blocks trades on AI-flagged "avoid" tickers
- ✅ Integrates external sentiment analysis into trading decisions
- ✅ Fail-safe design (errors don't crash bot)
- ✅ Supports flexible bias formats (string or dict)
- ✅ Refreshable (updates without restart)
- ✅ Combines with PDT guard for dual protection

**Used by:** `Trader_main_Grok4_20250731.py`, `bot_service.py`
**Key method:** `should_allow()` - check before every trade entry
**Data source:** `dexter_bias.json` (generated by external Dexter AI system)
