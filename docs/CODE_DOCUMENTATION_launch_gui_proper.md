# Code Documentation: launch_gui_proper.py

**Purpose:** macOS system tray GUI launcher for trading bot control
**File:** `launch_gui_proper.py`
**Lines:** 262
**Dependencies:** `sys`, `os`, `PyQt6`, `ipc_protocol`

---

## Overview

This is the main entry point for the trading bot GUI. It creates a system tray icon (menu bar icon) that provides a control panel for the bot without requiring a full window. The GUI communicates with the background bot service via IPC (Inter-Process Communication).

**Key Features:**
- System tray icon with color-coded status (green/yellow/red/blue)
- Real-time status updates every 5 seconds
- Start/stop/run-now controls
- Opens dashboard, logs, settings, backtest windows
- Checks bot service is running before starting
- macOS-native PyQt6 application

---

## Initialization & Pre-PyQt6 Checks

### Lines 1-9: Script Header & macOS Environment Setup

```python
#!/usr/bin/env python3
os.environ['QT_MAC_WANTS_LAYER'] = '1'
```

**What it does:**
- Shebang line: Makes script executable directly (`./launch_gui_proper.py`)
- Sets Qt environment variable for macOS layer rendering

**Programming Notes:**
- `QT_MAC_WANTS_LAYER='1'` fixes PyQt6 rendering issues on macOS
- Must be set BEFORE importing PyQt6 (or crashes/glitches occur)
- macOS uses Core Animation layers for window rendering

---

### Lines 12-27: Bot Service Check

```python
from ipc_protocol import IPCClient
client = IPCClient()

if not client.is_running():
    print("❌ ERROR: Bot service is not running!")
    print("Start it first in another terminal:")
    print("  ./1_start_bot_service.sh")
    sys.exit(1)
```

**What it does:**
Verifies bot service is running BEFORE creating GUI

**Process:**
1. Create IPC client
2. Check if bot service responds
3. If not running → print error message and exit
4. If running → proceed to GUI creation

**Programming Notes:**
- Done BEFORE PyQt6 import (faster failure)
- Prevents "bot not found" errors after GUI loads
- User-friendly error message with instructions
- `sys.exit(1)` returns error code (shell knows it failed)

**Why check before GUI?**
- Creating PyQt6 app takes 1-2 seconds
- If bot not running, fail fast (don't waste time)
- Better UX (immediate error vs delayed failure)

---

## PyQt6 Application Setup

### Lines 29-36: Import PyQt6 & Create Application

```python
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QAction
from PyQt6.QtCore import QTimer, Qt

app = QApplication(sys.argv)
app.setApplicationName("Trading Bot")
app.setQuitOnLastWindowClosed(False)
```

**What it does:**
Creates PyQt6 application instance

**Key Settings:**
- `QApplication(sys.argv)`: Main Qt application (processes command-line args)
- `setApplicationName("Trading Bot")`: Shows in macOS dock/menu
- `setQuitOnLastWindowClosed(False)`: **Critical** - keeps app running when windows closed

**Programming Notes:**
- `sys.argv` passes command-line arguments to Qt
- `QuitOnLastWindowClosed(False)` is essential for system tray apps
  - Default behavior: app quits when last window closes
  - System tray apps have no main window
  - Must set False or app exits immediately
- App runs until `app.quit()` called (Quit menu action)

---

## Icon Creation

### Lines 38-57: Function `create_icon(color_name)`

**What it does:**
Dynamically creates colored circular icons for system tray

**Parameters:**
- `color_name` (str): "green", "yellow", "red", or "gray"

**Returns:**
- `QIcon` object with colored circle

**Process:**
1. Create 32x32 pixel transparent canvas
2. Initialize painter
3. Enable antialiasing (smooth circles)
4. Select color from dictionary
5. Draw filled circle (24x24, centered with 4px margin)
6. Return as QIcon

**Programming Notes:**
- `QPixmap(32, 32)`: Bitmap image (raster, not vector)
- `Qt.GlobalColor.transparent`: RGBA(0,0,0,0)
- `QPainter`: Qt drawing API
- `setRenderHint(Antialiasing)`: Smooth edges (no jagged pixels)
- `setBrush()`: Fill color for shapes
- `setPen(NoPen)`: No outline (solid circle)
- `drawEllipse(4, 4, 24, 24)`: Circle at (4,4) with 24px diameter
- `painter.end()`: Must call before using pixmap

**Color Meanings:**
- 🟢 **Green:** Bot running, trading scheduled
- 🟡 **Yellow:** Bot idle, not scheduled
- 🔴 **Red:** Bot error, not responding
- ⚪ **Gray:** Default/unknown state
- 🔵 **Blue:** Bot actively trading (bonus color, not in dict)

**Why dynamic icons?**
- No external image files needed (self-contained)
- Easy to add more colors
- Consistent size/style
- Works on high-DPI displays (scales cleanly)

---

## System Tray Initialization

### Lines 59-67: System Tray Availability Check

```python
if not QSystemTrayIcon.isSystemTrayAvailable():
    print("❌ ERROR: System tray not available on this system")
    sys.exit(1)

tray = QSystemTrayIcon()
tray.setIcon(create_icon("yellow"))
tray.setToolTip("Trading Bot - Loading...")
```

**What it does:**
Checks if system tray supported, creates tray icon

**Programming Notes:**
- `isSystemTrayAvailable()`: Checks if OS supports tray icons
  - macOS: Always True
  - Some Linux DEs: May be False (depends on desktop environment)
  - Windows: Always True
- Initial icon: Yellow (loading state)
- Tooltip: Shows on hover

---

## Menu Creation

### Lines 69-112: Create Context Menu

**Structure:**
```
🟡 Bot Status: Loading...        [Disabled, shows state]
⏱ Last heartbeat: —             [Disabled, shows timestamp]
─────────────────────────────────
▶️  Start Bot                     [Clickable]
⏸️  Stop Bot                     [Clickable, disabled initially]
⚡ Run Now                       [Clickable, disabled initially]
─────────────────────────────────
📊 Dashboard                     [Clickable]
📄 Live Logs                     [Clickable]
⚙️  Settings                     [Clickable]
🧪 Train/Backtest               [Clickable]
─────────────────────────────────
❌ Quit                          [Clickable]
```

**Programming Notes:**
- `QMenu()`: Context menu (right-click menu)
- `QAction(text)`: Menu item
- `.setEnabled(False)`: Grayed out, not clickable
- `.addSeparator()`: Horizontal line separator
- `.triggered.connect(func)`: Calls function when clicked

**Status Actions (Non-clickable):**
- `status_action`: Shows bot state (running/idle/error)
- `heartbeat_action`: Shows last update time

**Control Actions:**
- `start_action`: Starts scheduled bot
- `stop_action`: Stops scheduled bot
- `run_now_action`: Triggers immediate execution

**Window Actions:**
- `dashboard_action`: Opens dashboard window
- `logs_action`: Opens live logs window
- `settings_action`: Opens settings editor
- `backtest_action`: Opens train/backtest tool

**App Actions:**
- `quit_action`: Quits entire application

---

## Status Update Logic

### Lines 114-156: Function `update_status()`

**What it does:**
Polls bot service for current status and updates GUI accordingly

**Process:**
1. Send `get_status` command via IPC
2. Parse response
3. Update icon color
4. Update status text
5. Enable/disable menu items
6. Update tooltip

**Response Handling:**

**Error Response:**
```python
{"error": "Bot service not running"}
```
- Icon: 🔴 Red
- Status: "🔴 Bot Status: Error"
- All controls disabled

**Success Response:**
```python
{
  "status": "running",
  "running": true,
  "next_execution": "15:55:00",
  "last_heartbeat": "2025-12-17 15:57:30"
}
```

**Status States:**

| `status` value | Icon | Status Text | Start Enabled | Stop Enabled | Run Now Enabled |
|----------------|------|-------------|---------------|--------------|-----------------|
| `"running"` | 🟢 Green | "🟢 Running (Next: 15:55)" | No | Yes | Yes |
| `"trading"` | 🔵 Blue | "🔵 Trading" | No | Yes | Yes |
| `"idle"` or other | 🟡 Yellow | "🟡 Idle" | Yes | No | No |
| Error response | 🔴 Red | "🔴 Error: ..." | No | No | No |

**Programming Notes:**
- Try-except catches IPC errors (don't crash GUI)
- `response.get("error")`: Checks for error field first
- Icon colors match status (visual consistency)
- Tooltip = shortened version of status text
- Menu item enabling/disabling prevents invalid commands

---

## Command Handlers

### Lines 158-163: Function `start_bot()`

**What it does:**
Sends "start" command to bot service

**Process:**
1. Send IPC command: `{"command": "start"}`
2. Check if successful
3. Show notification toast if successful
4. Update status

**Programming Notes:**
- `send_command()`: Blocking call (waits for response)
- `tray.showMessage()`: macOS notification toast
  - Title: "Trading Bot"
  - Message: "Bot started successfully"
  - Icon: Information icon
  - Duration: 3000ms (3 seconds)
- `update_status()`: Immediately refreshes GUI

---

### Lines 165-169: Function `stop_bot()`

**What it does:**
Sends "stop" command to bot service

**Process:**
Same as `start_bot()` but with "stop" command

---

### Lines 171-178: Function `run_now()`

**What it does:**
Triggers immediate trading execution (doesn't wait for scheduled time)

**Process:**
1. Send IPC command: `{"command": "run_now"}`
2. Check response
3. If successful → show success toast
4. If failed → show error toast with details
5. Update status

**Programming Notes:**
- Can fail (e.g., if already trading, or market closed)
- Error message extracted from `response.get("message")` or `response.get("error")`
- Uses `MessageIcon.Critical` for errors (red icon)
- 5 second duration for errors (longer than success messages)

---

## Window Handlers

### Lines 180-193: Function `show_dashboard()`

**What it does:**
Opens the dashboard window (portfolio overview, positions, P&L)

**Process:**
1. Import `DashboardWindow` class (lazy import)
2. Check if window already exists
3. If not, create new instance
4. Show window
5. Raise to front (`.raise_()`)
6. Activate window (`.activateWindow()` - gives focus)

**Programming Notes:**
- `global dashboard_window`: Persists window object across calls
- Lazy import: Only import when needed (faster startup)
- Window reuse: If already open, just raise it (don't create duplicate)
- `globals()`: Dict of global variables
- Try-except catches import errors (module missing)
- `traceback.print_exc()`: Prints full error stack

**Why lazy import?**
- Dashboard might not be used every session
- Saves ~100ms startup time
- Reduces memory if never opened

---

### Lines 195-208: Function `show_logs()`

**What it does:**
Opens live logs window (real-time log streaming)

**Process:**
Same pattern as `show_dashboard()` but for `LogsWindow`

---

### Lines 210-223: Function `show_settings()`

**What it does:**
Opens settings editor window (edit config.yaml)

**Process:**
Same pattern as `show_dashboard()` but for `SettingsWindow`

---

### Lines 225-238: Function `show_backtest()`

**What it does:**
Opens train/backtest window (model training and strategy testing)

**Process:**
Same pattern as `show_dashboard()` but for `BacktestWindow`

**Programming Notes:**
- Passes `client` parameter (needs IPC to trigger backtests)

---

## Event Binding

### Lines 240-247: Connect Actions to Handlers

```python
start_action.triggered.connect(start_bot)
stop_action.triggered.connect(stop_bot)
run_now_action.triggered.connect(run_now)
dashboard_action.triggered.connect(show_dashboard)
logs_action.triggered.connect(show_logs)
settings_action.triggered.connect(show_settings)
backtest_action.triggered.connect(show_backtest)
```

**What it does:**
Links menu actions to Python functions

**Programming Notes:**
- `.triggered`: Qt signal emitted when action clicked
- `.connect(func)`: Binds signal to function
- No parentheses on function (pass reference, not call it)
- When user clicks → Qt calls function automatically

---

## Status Timer

### Lines 249-252: Create Update Timer

```python
status_timer = QTimer()
status_timer.timeout.connect(update_status)
status_timer.start(5000)  # Update every 5 seconds
```

**What it does:**
Automatically calls `update_status()` every 5 seconds

**Programming Notes:**
- `QTimer()`: Qt timer object
- `.timeout`: Signal emitted when timer fires
- `.start(5000)`: Start timer with 5000ms (5 second) interval
- Timer repeats indefinitely until `.stop()` called
- Runs in Qt event loop (non-blocking)

**Why 5 seconds?**
- Balance between responsiveness and CPU usage
- Bot state doesn't change often (scheduled for 3:55 PM)
- 5 sec = ~12 API calls per minute (reasonable)

---

## Initial State & Event Loop

### Lines 254-255: Initial Status Update

```python
update_status()
```

**What it does:**
Immediately fetches bot status (don't wait 5 seconds)

**Why needed?**
- Timer starts at 5 seconds (first update would be delayed)
- User sees current state immediately
- Better UX (instant feedback)

---

### Lines 257-261: Start Qt Event Loop

```python
print("✅ Menu bar icon should now be visible!")
sys.exit(app.exec())
```

**What it does:**
Starts Qt event loop and blocks until app quits

**Programming Notes:**
- `app.exec()`: Runs Qt event loop (processes events, timers, signals)
- **Blocking call:** Doesn't return until `app.quit()` called
- Returns exit code (0 = success, 1+ = error)
- `sys.exit()`: Passes exit code to shell
- Print statement helps user find icon (macOS top-right)

**Qt Event Loop:**
```
Start → Wait for events → Process event → Update GUI → Repeat
        ↑                                              ↓
        └──────────────────────────────────────────────┘
```

---

## Application Flow Diagram

```
[Start Script]
       ↓
[Check bot service running]
   ↓           ↓
 Yes          No → Exit with error
   ↓
[Create Qt app]
   ↓
[Create system tray icon]
   ↓
[Create menu]
   ↓
[Start status timer]
   ↓
[Initial status update]
   ↓
[Show tray icon]
   ↓
[Enter Qt event loop]
   ↓
[Wait for events...]
   ↓
[User clicks menu item]
   ↓
[Call handler function]
   ↓
[Update GUI]
   ↓
[Back to event loop]
```

---

## Threading Model

**Main Thread (Qt Event Loop):**
- Handles GUI rendering
- Processes menu clicks
- Runs status timer
- Calls IPC commands

**No Background Threads:**
- IPC calls are **blocking** (wait for response)
- Short operations (<1 second) OK to block
- Long operations (backtests) run in bot service, not GUI

**Why blocking IPC is OK:**
- Status updates: ~10ms
- Start/stop commands: ~50ms
- User doesn't notice <100ms freeze
- Simpler code (no threading complexity)

**If IPC was async:**
```python
# Would need callbacks or async/await
async def update_status():
    response = await client.send_command_async(...)
    # Update GUI
```

---

## Window Lifecycle

**Dashboard Window Example:**

```python
# First call to show_dashboard()
dashboard_window = DashboardWindow(client)  # Create
dashboard_window.show()                      # Show

# User closes window
# dashboard_window object still exists

# Second call to show_dashboard()
if 'dashboard_window' in globals():
    # Window exists, just show it
    dashboard_window.show()
    dashboard_window.raise_()
else:
    # Window destroyed, create new one
    dashboard_window = DashboardWindow(client)
```

**Why keep window objects?**
- Faster to show existing window than create new one
- Preserves window state (size, position, scroll)
- User doesn't lose their place

---

## Error Handling

### IPC Errors
```python
# Bot service crashes during operation
response = client.send_command({"command": "get_status"})
# Returns: {"error": "Bot service not running"}
# GUI shows red icon, disables controls
```

### Import Errors
```python
# Dashboard module missing
try:
    from gui.dashboard_window import DashboardWindow
except Exception as e:
    print(f"Error opening dashboard: {e}")
    traceback.print_exc()
# User sees error in terminal, GUI continues running
```

### Timeout Errors
```python
# Bot service hung, taking too long
response = client.send_command(..., timeout=5.0)
# After 5 seconds: {"error": "Request timed out"}
# GUI shows error toast
```

---

## macOS Integration

**System Tray Icon:**
- Shows in menu bar (top-right of screen)
- Persists after windows closed
- Native macOS look and feel

**Notifications:**
```python
tray.showMessage("Title", "Message", icon, duration)
```
- Uses macOS Notification Center
- Appears in top-right corner
- Respects system notification settings

**Application Name:**
```python
app.setApplicationName("Trading Bot")
```
- Shows in macOS menu bar
- Shows in Dock when windows open
- Shows in Activity Monitor

---

## Keyboard Shortcuts (Not Implemented)

**Could add:**
```python
start_action.setShortcut("Ctrl+S")
stop_action.setShortcut("Ctrl+X")
run_now_action.setShortcut("Ctrl+R")
```

**macOS uses:**
- `Cmd` (⌘) instead of Ctrl
- `QKeySequence.StandardKey.New` for platform-specific shortcuts

---

## Testing Checklist

**Manual Tests:**
- [ ] Icon appears in menu bar
- [ ] Clicking icon shows menu
- [ ] Status updates every 5 seconds
- [ ] Start button enables when idle
- [ ] Stop button enables when running
- [ ] Run now button enables when running
- [ ] Dashboard window opens
- [ ] Logs window opens
- [ ] Settings window opens
- [ ] Backtest window opens
- [ ] Quit exits application
- [ ] Red icon shown when bot crashes
- [ ] Notifications appear for commands

**Error Tests:**
- [ ] Start without bot service → shows error and exits
- [ ] Bot crashes mid-session → icon turns red
- [ ] IPC timeout → shows error toast
- [ ] Missing gui module → prints error, continues running

---

## Performance

**Memory Usage:**
- App: ~50 MB (PyQt6 framework)
- Per window: ~5-10 MB
- Total (all windows): ~100 MB

**CPU Usage:**
- Idle: <1%
- Status updates: ~2% spike every 5 seconds
- Window rendering: ~5-10% when animating

**Startup Time:**
- Cold start: ~2 seconds (PyQt6 import)
- Warm start: ~0.5 seconds (Python cached)

---

## Summary

`launch_gui_proper.py` is the main GUI entry point that:
- ✅ Creates macOS system tray application
- ✅ Provides real-time bot control and monitoring
- ✅ Color-coded status icons (green/yellow/red/blue)
- ✅ Opens dashboard, logs, settings, backtest windows
- ✅ Polls bot service every 5 seconds
- ✅ Graceful error handling (bot crashes, timeouts)
- ✅ Native macOS notifications
- ✅ Persists after windows closed (system tray app)

**Used by:** End users (macOS only)
**Requires:** `bot_service.py` running in background
**Launches:** Dashboard, logs, settings, backtest windows (in `gui/` folder)
