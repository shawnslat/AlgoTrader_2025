# 🚀 START HERE - Easy Two-Step Launch

## The Problem

The integrated launcher has an issue where the bot service doesn't stay running when launched as a subprocess.

## The Solution

Run the bot service and GUI **separately** in two terminal windows.

---

## Step-by-Step Instructions

### Step 1: Start Bot Service

Open a terminal and run:

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
./1_start_bot_service.sh
```

**You'll see:**
```
╔════════════════════════════════════════╗
║  Trading Bot Service                   ║
║  Keep this terminal window open!      ║
╚════════════════════════════════════════╝

Starting bot service...

2025-12-17 08:00:14,274 [INFO] IPC server started on /tmp/trader_bot.sock
2025-12-17 08:00:14,274 [INFO] Bot service running (simple wrapper mode)
```

**✅ Keep this terminal window open!** Don't close it.

---

### Step 2: Start GUI

Open a **NEW terminal** (Cmd+N) and run:

```bash
cd /Users/shawnslat/Documents/GitHub/Trader_2025
./2_start_gui.sh
```

**You'll see:**
```
╔════════════════════════════════════════╗
║  Trading Bot GUI                       ║
╚════════════════════════════════════════╝

✅ Bot service detected

Starting GUI...
Look for the menu bar icon in the top-right of your screen!
```

**🎯 Now look at the top-right of your screen!**

You should see a **colored circle** (yellow or green, NOT red).

---

## What You Should See Now

### Menu Bar Icon
- 🟡 **Yellow** = Idle (good!)
- 🟢 **Green** = Running (good!)
- ~~🔴 Red~~ = Should NOT be red anymore

### Click the Icon
You should see:
```
🟡 Bot Status: Idle (Next: 15:55:00)
────────────────────────────────
▶️  Start Bot              ← Click this!
⏸️  Stop Bot
────────────────────────────────
📊 Dashboard              ← Click this!
📈 Positions
📋 Today's Signals
📄 Live Logs              ← Click this!
⚙️  Settings
────────────────────────────────
❌ Quit
```

**All options should work now!**

---

## Try These Features

1. **Click "Start Bot"**
   - Icon should turn 🟢 green
   - Status should say "Running"

2. **Click "Dashboard"**
   - Window opens showing portfolio
   - Account summary at top
   - Position table (shows mock AAPL)
   - Signals table

3. **Click "Live Logs"**
   - Window opens with real-time logs
   - Filter dropdown works
   - Auto-scrolls to bottom

4. **Click "Settings"**
   - Config editor opens
   - Can edit tickers
   - Can adjust risk parameters

---

## To Stop

**Stop GUI:**
- Click icon → "Quit"
- Or press Ctrl+C in Terminal 2

**Stop Bot Service:**
- Press Ctrl+C in Terminal 1

---

## If Still Having Issues

### Icon is still red?

Bot service isn't running.

**Check Terminal 1:**
- Is `1_start_bot_service.sh` still running?
- Do you see "IPC server started" message?

**Test manually:**
```bash
ls -la /tmp/trader_bot.sock
# Should show: srwxr-xr-x ... /tmp/trader_bot.sock
```

### No icon at all?

GUI didn't start.

**Check Terminal 2:**
- Any error messages?
- Try running again: `./2_start_gui.sh`

### Menu won't open?

macOS permission issue.

**Fix:**
1. System Settings → Privacy & Security → Accessibility
2. Add Terminal or Python

---

## Why Two Terminals?

**Advantages:**
- ✅ See all logs in real-time
- ✅ Easy to debug
- ✅ Bot service stays running reliably
- ✅ Can restart GUI without restarting bot

**This is temporary** - once we verify it works, we'll fix the integrated launcher.

---

## Quick Reference

**Terminal 1 (keep open):**
```bash
./1_start_bot_service.sh
```

**Terminal 2 (GUI):**
```bash
./2_start_gui.sh
```

**Check status:**
```bash
python3 -c "from ipc_protocol import IPCClient; print(IPCClient().send_command({'command': 'get_status'}))"
```

---

## ✅ Checklist

After following the steps:

- [ ] Terminal 1 shows "IPC server started"
- [ ] Terminal 2 shows "✅ Bot service detected"
- [ ] Menu bar icon is visible
- [ ] Icon is 🟡 yellow or 🟢 green (not red)
- [ ] Clicking icon shows menu
- [ ] Dashboard opens when clicked
- [ ] Live Logs opens when clicked
- [ ] Settings opens when clicked

If all checked: **Success!** 🎉

If any unchecked: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

