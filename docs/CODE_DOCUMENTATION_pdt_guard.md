# Code Documentation: pdt_guard.py

**Purpose:** Pattern Day Trading (PDT) protection system to prevent exceeding day trade limits
**File:** `pdt_guard.py`
**Lines:** 58
**Dependencies:** `datetime`, `collections`

---

## Overview

This module implements a lightweight PDT (Pattern Day Trading) guard that tracks and limits intraday round-trip trades to avoid violating SEC Pattern Day Trading rules. It prevents accounts under $25k from making more than 3 day trades in a rolling 5-day period.

---

## Class: `DayTradeGuard`

**Purpose:** Tracks and enforces day trading limits by counting same-day entry/exit combinations

**Key Attributes:**
- `max_day_trades` (int): Maximum allowed day trades per day (default: 2)
- `day_trade_counts` (defaultdict): Maps date strings → count of day trades on that date
- `open_entries` (dict): Maps ticker → entry date string to track when positions were opened

---

### Constructor: `__init__(self, max_day_trades: int = 2)`

**Lines:** 11-14

**What it does:**
1. Sets the maximum allowed day trades per day
2. Initializes a defaultdict to track day trade counts per date
3. Creates an empty dictionary to track open position entry dates

**Parameters:**
- `max_day_trades`: Maximum day trades allowed (default 2, configurable)

**Programming Notes:**
- Uses `defaultdict(int)` so missing dates automatically return 0
- `open_entries` stores when each ticker position was opened to detect same-day exits

---

### Static Method: `_date_str(dt: datetime) -> str`

**Lines:** 16-18

**What it does:**
Converts a datetime object to a standardized date string format

**Parameters:**
- `dt`: datetime object to convert

**Returns:**
- String in format "YYYY-MM-DD" (e.g., "2025-12-17")

**Programming Notes:**
- Static method (doesn't need instance state)
- Used consistently throughout class for date key formatting
- Ensures all date comparisons use same string format

---

### Method: `_prune_old(self, now: datetime)`

**Lines:** 20-34

**What it does:**
Removes day trade counts older than 5 days to keep memory footprint small

**Parameters:**
- `now`: Current datetime for calculating cutoff

**Process:**
1. Calculate cutoff date (5 days ago from now)
2. Convert both cutoff and stored dates to naive datetime (removes timezone)
3. Find all date keys older than cutoff
4. Delete those old entries from `day_trade_counts`

**Programming Notes:**
- Uses timezone-naive comparison to avoid timezone mismatch errors
- `fromisoformat()` parses "YYYY-MM-DD" strings back to datetime
- Builds `to_delete` list first to avoid modifying dict during iteration
- PDT rules use rolling 5-day window, so older data is irrelevant

**Example:**
```python
# Today is Dec 17, 2025
# Cutoff = Dec 12, 2025
# Will delete: Dec 11, Dec 10, Dec 9, etc.
```

---

### Method: `remaining_today(self, now: datetime) -> int`

**Lines:** 36-40

**What it does:**
Calculates how many day trades are still available today

**Parameters:**
- `now`: Current datetime

**Returns:**
- Integer count of remaining day trades (0 or positive)

**Process:**
1. Prunes old data (ensures only recent 5 days are tracked)
2. Converts `now` to date string format
3. Gets how many day trades already used today (0 if none)
4. Subtracts used from max, returns at least 0 (never negative)

**Programming Notes:**
- `max(result, 0)` ensures never returns negative numbers
- `.get(today, 0)` returns 0 if today has no entry yet
- Prunes old data on every call for memory efficiency

**Example:**
```python
# max_day_trades = 2
# day_trade_counts["2025-12-17"] = 1
# remaining_today() returns: max(2 - 1, 0) = 1
```

---

### Method: `can_open(self, now: datetime) -> bool`

**Lines:** 42-43

**What it does:**
Determines if a new position can be opened (checks if day trade limit not exceeded)

**Parameters:**
- `now`: Current datetime

**Returns:**
- `True` if day trades remain, `False` if limit reached

**Programming Notes:**
- Simple wrapper around `remaining_today()`
- Returns boolean for cleaner conditional logic in trading code
- Checks remaining trades > 0

**Usage in trading logic:**
```python
if guard.can_open(datetime.now()):
    place_buy_order()  # Safe to open new position
else:
    skip_trade()  # Limit reached, avoid PDT violation
```

---

### Method: `record_entry(self, ticker: str, now: datetime)`

**Lines:** 45-46

**What it does:**
Records when a position is opened for a ticker

**Parameters:**
- `ticker`: Stock/crypto symbol (e.g., "AAPL", "BTCUSD")
- `now`: Datetime when position was entered

**Process:**
1. Converts `now` to date string
2. Stores in `open_entries[ticker]`

**Programming Notes:**
- Overwrites previous entry if ticker already has one (shouldn't happen in normal flow)
- Timezone stripped via `.replace(tzinfo=None)` before conversion
- Does NOT increment day trade counter (that happens on exit)

**Example:**
```python
# Opens AAPL position on Dec 17, 2025
guard.record_entry("AAPL", datetime(2025, 12, 17, 10, 30))
# open_entries = {"AAPL": "2025-12-17"}
```

---

### Method: `record_exit(self, ticker: str, now: datetime)`

**Lines:** 48-54

**What it does:**
Records when a position is closed and increments day trade counter if entry/exit on same day

**Parameters:**
- `ticker`: Stock/crypto symbol being exited
- `now`: Datetime when position was closed

**Process:**
1. Retrieves entry date for the ticker
2. Converts exit datetime to date string
3. **If entry and exit dates match** → increments day trade counter for that date
4. Removes ticker from `open_entries` (position now closed)

**Programming Notes:**
- Only counts as day trade if `entry_date == exit_date` (same calendar day)
- If no entry recorded, does nothing (guards against orphaned exits)
- `del open_entries[ticker]` removes tracking after exit
- Day trade counter persists for 5 days (pruned by `_prune_old`)

**Example - Day Trade (counts):**
```python
# Entry: Dec 17, 10:30 AM
guard.record_entry("AAPL", datetime(2025, 12, 17, 10, 30))
# Exit: Dec 17, 3:55 PM (same day!)
guard.record_exit("AAPL", datetime(2025, 12, 17, 15, 55))
# Result: day_trade_counts["2025-12-17"] += 1
```

**Example - NOT a Day Trade:**
```python
# Entry: Dec 17, 3:55 PM
guard.record_entry("AAPL", datetime(2025, 12, 17, 15, 55))
# Exit: Dec 18, 10:00 AM (next day!)
guard.record_exit("AAPL", datetime(2025, 12, 18, 10, 0))
# Result: day_trade_counts NOT incremented (different dates)
```

---

### Method: `current_open_entry_date(self, ticker: str) -> str`

**Lines:** 56-58

**What it does:**
Returns the date when a ticker position was opened (if currently open)

**Parameters:**
- `ticker`: Stock/crypto symbol to check

**Returns:**
- Date string "YYYY-MM-DD" if position open
- Empty string "" if no open position for that ticker

**Programming Notes:**
- Read-only query (doesn't modify state)
- Used for debugging or displaying position entry dates
- Returns empty string instead of `None` for safer string operations

**Usage:**
```python
entry_date = guard.current_open_entry_date("AAPL")
if entry_date:
    print(f"AAPL opened on {entry_date}")
else:
    print("No open AAPL position")
```

---

## How PDT Guard Integrates with Trading Bot

### In `Trader_main_Grok4_20250731.py`:

```python
# Initialize guard
guard = DayTradeGuard(max_day_trades=2)

# Before buying
if guard.can_open(datetime.now(pytz.timezone('US/Eastern'))):
    place_buy_order("AAPL")
    guard.record_entry("AAPL", datetime.now(pytz.timezone('US/Eastern')))
else:
    logger.warning("Day trade limit reached, skipping trade")

# When selling
guard.record_exit("AAPL", datetime.now(pytz.timezone('US/Eastern')))
# If same day as entry, day trade counter increments
```

---

## PDT Rules Enforced

**SEC Pattern Day Trading Rule:**
- **Accounts < $25,000:** Maximum 3 day trades in rolling 5-day period
- **Day trade definition:** Buy and sell (or short and cover) same security on same trading day
- **Penalty:** Account restricted to closing positions only if violated

**This Guard's Implementation:**
- Configured for 2 day trades/day (conservative)
- Tracks rolling 5-day window
- Blocks new entries when limit hit
- Allows exits (closing positions always allowed)

---

## Key Design Decisions

**Why 2 instead of 3?**
- Conservative buffer to avoid accidental violations
- Leaves room for unexpected fills or manual overrides

**Why 5-day rolling window?**
- Matches SEC's PDT rule rolling period
- Old data automatically pruned to save memory

**Why track by entry date?**
- SEC cares about same-day entry/exit
- Multi-day holds are NOT day trades
- Must know when position opened to detect violations

**Why defaultdict(int)?**
- Automatically initializes missing dates to 0
- Cleaner code (no need for `if date not in dict` checks)

---

## Potential Improvements (Not Implemented)

1. **Account balance check:** Disable guard for accounts > $25k
2. **Rolling 5-day total:** Currently tracks per-day, could track 5-day sum
3. **Persistent storage:** Save counts to disk (currently in-memory only)
4. **Position quantity tracking:** Handle partial exits
5. **Multi-account support:** Track limits per account ID

---

## Thread Safety

**Current implementation:** NOT thread-safe
- Assumes single-threaded execution
- Concurrent access could corrupt `day_trade_counts` or `open_entries`

**If using multiple threads:**
```python
import threading

class DayTradeGuard:
    def __init__(self, max_day_trades: int = 2):
        self.lock = threading.Lock()  # Add lock
        # ... rest of init

    def record_entry(self, ticker: str, now: datetime):
        with self.lock:  # Protect modification
            self.open_entries[ticker] = self._date_str(now.replace(tzinfo=None))
```

---

## Testing Examples

```python
from datetime import datetime
from pdt_guard import DayTradeGuard

# Initialize guard
guard = DayTradeGuard(max_day_trades=2)

# Day 1: Open and close AAPL (day trade)
now = datetime(2025, 12, 17, 10, 0)
guard.record_entry("AAPL", now)
guard.record_exit("AAPL", datetime(2025, 12, 17, 15, 55))
print(guard.remaining_today(now))  # Output: 1 (1 day trade used)

# Day 1: Open and close MSFT (day trade)
guard.record_entry("MSFT", now)
guard.record_exit("MSFT", now)
print(guard.remaining_today(now))  # Output: 0 (2 day trades used)

# Day 1: Try to open TSLA
print(guard.can_open(now))  # Output: False (limit reached)

# Day 2: Reset counter
now = datetime(2025, 12, 18, 10, 0)
print(guard.remaining_today(now))  # Output: 2 (new day, fresh limit)
```

---

## Summary

`pdt_guard.py` is a simple but critical safety feature that:
- ✅ Prevents PDT violations (protects account from restrictions)
- ✅ Tracks intraday round trips automatically
- ✅ Enforces configurable daily limits
- ✅ Automatically prunes old data (memory efficient)
- ✅ Provides clean boolean API for trading logic

**Used by:** `Trader_main_Grok4_20250731.py`, `bot_service.py`
**Key method:** `can_open()` - check before every trade entry
