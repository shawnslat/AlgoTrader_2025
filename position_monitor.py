"""
Position Monitor - Enforces stop losses and scaled exits on existing positions

This runs as part of each trading cycle to:
1. Check if any positions have breached their stop loss levels
2. Execute scaled exits (sell portions at profit tiers) to let winners run
3. Enforce trailing stops on final tier positions

Scaled exit tiers (configurable in config.yaml):
  Tier 1: Sell 40% at +3% (lock in early profit)
  Tier 2: Sell 30% at +6% (capture the move)
  Tier 3: Remaining 30% rides a trailing stop (let winners run)
"""

import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import pytz
import requests

logger = logging.getLogger(__name__)

ALPACA_AUTH_STATUS_CODES = {401, 403}
EXIT_STATE_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'exit_tiers.json')
EQUITY_PEAK_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'equity_peak.json')
KILL_SWITCH_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'kill_switch_tripped.json')


def _alpaca_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def _is_auth_error(exc: Exception) -> bool:
    status_code = _alpaca_status_code(exc)
    if status_code in ALPACA_AUTH_STATUS_CODES:
        return True
    message = str(exc).lower()
    return "unauthorized" in message or "forbidden" in message


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
        ),
    ):
        return True
    message = str(exc).lower()
    return any(
        token in message for token in (
            "timed out",
            "max retries exceeded",
            "failed to establish a new connection",
            "name or service not known",
            "nodename nor servname provided",
            "connection reset",
        )
    )


# ---------------------------------------------------------------------------
# Exit tier state persistence
# ---------------------------------------------------------------------------

def _load_exit_state() -> Dict:
    """Load scaled exit state from disk. Returns {symbol: {tiers_sold: [0,1,...], high_water: float, original_qty: float}}"""
    try:
        if os.path.exists(EXIT_STATE_FILE):
            with open(EXIT_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load exit state: {e}")
    return {}


def _save_exit_state(state: Dict):
    """Persist scaled exit state to disk."""
    try:
        os.makedirs(os.path.dirname(EXIT_STATE_FILE), exist_ok=True)
        with open(EXIT_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save exit state: {e}")


# ---------------------------------------------------------------------------
# Portfolio-level drawdown kill switch
# ---------------------------------------------------------------------------

def _load_equity_peak() -> Dict:
    """Load rolling peak-equity state. Returns {peak: float, peak_date: str, window_start: str}."""
    try:
        if os.path.exists(EQUITY_PEAK_FILE):
            with open(EQUITY_PEAK_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load equity peak state: {e}")
    return {}


def _save_equity_peak(state: Dict) -> None:
    try:
        os.makedirs(os.path.dirname(EQUITY_PEAK_FILE), exist_ok=True)
        tmp = EQUITY_PEAK_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, EQUITY_PEAK_FILE)
    except Exception as e:
        logger.warning(f"Could not save equity peak state: {e}")


def _write_kill_switch_flag(reason: str, equity: float, peak: float, drawdown_pct: float) -> None:
    """Write a flag file that the dashboard can read to show the tripped state."""
    payload = {
        "tripped_at": datetime.now(pytz.timezone('US/Eastern')).isoformat(timespec='seconds'),
        "reason": reason,
        "equity": equity,
        "peak": peak,
        "drawdown_pct": drawdown_pct,
    }
    try:
        os.makedirs(os.path.dirname(KILL_SWITCH_FILE), exist_ok=True)
        with open(KILL_SWITCH_FILE, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.error(f"Could not write kill switch flag: {e}")


def is_kill_switch_tripped() -> bool:
    """Check if the drawdown kill switch flag file exists."""
    return os.path.exists(KILL_SWITCH_FILE)


def clear_kill_switch() -> bool:
    """Remove the kill-switch flag (called when the user resets via dashboard).
    Also resets the equity peak so a new high-water mark begins from current equity."""
    cleared = False
    try:
        if os.path.exists(KILL_SWITCH_FILE):
            os.remove(KILL_SWITCH_FILE)
            cleared = True
        if os.path.exists(EQUITY_PEAK_FILE):
            os.remove(EQUITY_PEAK_FILE)
    except Exception as e:
        logger.error(f"Could not clear kill switch: {e}")
    return cleared


def check_drawdown_kill_switch(api, config: Dict) -> Dict:
    """
    Portfolio-level drawdown kill switch. Tracks a rolling peak-equity high-water
    mark and trips when equity falls below it by the configured threshold.

    When tripped, writes a flag file. Callers should check is_kill_switch_tripped()
    before opening new positions. Existing positions remain managed by the per-trade
    stop loss in check_and_enforce_stops -- this function does NOT auto-liquidate.

    Returns a dict describing current state for logging:
        {tripped, equity, peak, drawdown_pct, threshold_pct}
    """
    cfg = config.get('drawdown_kill_switch', {})
    if not cfg.get('enabled', False):
        return {"tripped": False, "enabled": False}

    threshold_pct = float(cfg.get('threshold_pct', 0.12))  # default 12%

    try:
        account = api.get_account()
        equity = float(account.equity)
    except Exception as e:
        if _is_transient_error(e):
            logger.warning(f"Transient error fetching account for drawdown check: {e}")
        else:
            logger.error(f"Could not fetch account equity for drawdown check: {e}")
        return {"tripped": is_kill_switch_tripped(), "enabled": True, "error": str(e)}

    state = _load_equity_peak()
    peak = float(state.get('peak', 0.0))
    today_iso = datetime.now(pytz.timezone('US/Eastern')).date().isoformat()

    # Initialize or update the peak
    if equity > peak:
        peak = equity
        state['peak'] = peak
        state['peak_date'] = today_iso
        if 'window_start' not in state:
            state['window_start'] = today_iso
        _save_equity_peak(state)

    drawdown_pct = (peak - equity) / peak if peak > 0 else 0.0
    tripped = drawdown_pct >= threshold_pct

    result = {
        "tripped": tripped,
        "enabled": True,
        "equity": equity,
        "peak": peak,
        "drawdown_pct": drawdown_pct,
        "threshold_pct": threshold_pct,
    }

    if tripped and not is_kill_switch_tripped():
        reason = (f"Drawdown {drawdown_pct:.1%} from peak ${peak:,.2f} "
                  f"(now ${equity:,.2f}) exceeded threshold {threshold_pct:.1%}")
        logger.error("=" * 60)
        logger.error(f"DRAWDOWN KILL SWITCH TRIPPED: {reason}")
        logger.error("New entries are now blocked. Reset via dashboard when ready.")
        logger.error("=" * 60)
        _write_kill_switch_flag(reason, equity, peak, drawdown_pct)
    elif tripped:
        logger.warning(f"Drawdown kill switch still tripped: equity ${equity:,.2f} vs peak ${peak:,.2f} ({drawdown_pct:.1%})")
    else:
        logger.debug(f"Drawdown OK: equity ${equity:,.2f} vs peak ${peak:,.2f} ({drawdown_pct:.1%} of {threshold_pct:.1%})")

    return result


# ---------------------------------------------------------------------------

def _submit_sell(api, symbol: str, qty: float, time_in_force: str) -> Optional[object]:
    """Submit a sell order. Returns the order object or None on failure."""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='sell',
            type='market',
            time_in_force=time_in_force,
        )
        return order
    except Exception as e:
        logger.error(f"Failed to submit sell for {symbol}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main monitor function
# ---------------------------------------------------------------------------

def check_and_enforce_stops(api, config: Dict) -> List[Dict]:
    """
    Check all open positions for stop loss, scaled profit exits, and trailing stops.

    Scaled exits sell portions of a winning position at predetermined profit levels,
    letting the remainder ride with a trailing stop. This raises average win size
    while still locking in profit early.

    Args:
        api: Alpaca REST API client
        config: Configuration dict

    Returns:
        List of actions taken (for logging)
    """
    actions = []

    stop_loss_pct = config.get('stop_loss_pct', 0.05)
    take_profit_pct = config.get('take_profit_pct', 0.10)

    # Scaled exit config
    scaled_cfg = config.get('scaled_exits', {})
    use_scaled = scaled_cfg.get('enabled', False)
    tiers = scaled_cfg.get('tiers', [])
    scaled_trail_pct = scaled_cfg.get('trail_pct', 0.03)

    # Legacy trailing stop (used when scaled exits disabled)
    trailing_stop = config.get('trailing_stop', {})
    use_trailing = trailing_stop.get('enabled', False)
    trail_pct = trailing_stop.get('trail_pct', 0.03)

    try:
        positions = api.list_positions()
    except Exception as e:
        if _is_auth_error(e):
            logger.error("Failed to fetch positions for stop checks: unauthorized. Skipping this cycle.")
        elif _is_transient_error(e):
            logger.warning(f"Connection error fetching positions for stop checks: {e}")
        else:
            logger.error(f"Failed to fetch positions: {e}")
        return actions

    if not positions:
        logger.debug("No open positions to monitor")
        return actions

    logger.info(f"Monitoring {len(positions)} open positions...")

    # Load exit tier state
    exit_state = _load_exit_state() if use_scaled else {}
    open_symbols = set()
    state_changed = False

    for pos in positions:
        symbol = pos.symbol
        open_symbols.add(symbol)
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price or 0)
        current_price = float(pos.current_price or 0)
        unrealized_pl_pct = float(pos.unrealized_plpc or 0)
        tif = 'day' if not is_crypto(symbol) else 'gtc'

        # Guard: Alpaca occasionally reports 0/None entry or current price
        # (e.g. transient API state or odd crypto symbols). Skip this position
        # rather than crash the whole cycle with a division by zero.
        if entry_price <= 0 or current_price <= 0:
            logger.warning(
                f"SKIP {symbol}: invalid price data (entry={entry_price}, current={current_price}) — "
                "position left untouched this cycle."
            )
            continue

        # Use crypto-specific stop/TP if this is a crypto position
        if is_crypto(symbol):
            crypto_cfg = config.get('crypto', {})
            pos_stop_pct = crypto_cfg.get('stop_loss_pct', stop_loss_pct)
            pos_tp_pct = crypto_cfg.get('take_profit_pct', take_profit_pct)
        else:
            pos_stop_pct = stop_loss_pct
            pos_tp_pct = take_profit_pct

        stop_price = entry_price * (1 - pos_stop_pct)

        # ---- STOP LOSS (always checked first, overrides everything) ----
        if current_price <= stop_price:
            reason = f"Price ${current_price:.2f} <= Stop ${stop_price:.2f} ({unrealized_pl_pct*100:+.1f}%)"
            logger.warning(f"STOP_LOSS {symbol}: {reason}")
            sell_qty = abs(qty) if is_crypto(symbol) else round(abs(qty), 2)
            order = _submit_sell(api, symbol, sell_qty, tif)
            actions.append({
                'symbol': symbol, 'action': 'STOP_LOSS', 'reason': reason,
                'qty': sell_qty, 'price': current_price,
                'order_id': order.id if order else None,
                'status': 'submitted' if order else 'failed',
            })
            # Clean up exit state for this position
            if symbol in exit_state:
                del exit_state[symbol]
                state_changed = True
            continue

        # ---- SCALED EXITS (tier-based partial profit taking) ----
        if use_scaled and tiers:
            gain_pct = (current_price - entry_price) / entry_price

            # Initialize state for this position if new
            if symbol not in exit_state:
                exit_state[symbol] = {
                    'tiers_sold': [],
                    'original_qty': qty,
                    'high_water': current_price,
                    'entry_price': entry_price,
                }
                state_changed = True

            sym_state = exit_state[symbol]

            # Update high water mark for trailing stop
            if current_price > sym_state.get('high_water', 0):
                sym_state['high_water'] = current_price
                state_changed = True

            original_qty = sym_state.get('original_qty', qty)

            # Check each tier
            for i, tier in enumerate(tiers):
                if i in sym_state['tiers_sold']:
                    continue  # Already sold this tier

                tier_pct = tier.get('pct', 0)
                tier_fraction = tier.get('sell_fraction', 0)
                is_trailing_tier = tier.get('trailing', False)

                if is_trailing_tier:
                    # This tier uses a trailing stop instead of a fixed target
                    high_water = sym_state.get('high_water', entry_price)
                    trail_from_high = (high_water - current_price) / high_water

                    # Only trigger trailing stop if position has been profitable
                    # (high water > entry + some minimum gain)
                    min_gain_for_trail = 0.02  # At least 2% gain before trailing kicks in
                    high_water_gain = (high_water - entry_price) / entry_price

                    if high_water_gain >= min_gain_for_trail and trail_from_high >= scaled_trail_pct:
                        sell_qty_raw = original_qty * tier_fraction
                        sell_qty = sell_qty_raw if is_crypto(symbol) else round(sell_qty_raw, 2)
                        sell_qty = min(sell_qty, qty)  # Can't sell more than we have

                        if sell_qty > 0:
                            reason = (f"SCALED_EXIT Tier {i+1} (trailing): "
                                     f"dropped {trail_from_high:.1%} from high ${high_water:.2f}, "
                                     f"selling {sell_qty} ({tier_fraction:.0%} of original)")
                            logger.info(f"{symbol}: {reason}")
                            order = _submit_sell(api, symbol, sell_qty, tif)
                            if order:
                                sym_state['tiers_sold'].append(i)
                                state_changed = True
                                actions.append({
                                    'symbol': symbol, 'action': f'SCALED_EXIT_T{i+1}_TRAIL',
                                    'reason': reason, 'qty': sell_qty,
                                    'price': current_price, 'order_id': order.id,
                                    'status': 'submitted',
                                })
                                qty -= sell_qty  # Update remaining qty for next tier checks

                elif gain_pct >= tier_pct > 0:
                    # Fixed profit tier hit
                    sell_qty_raw = original_qty * tier_fraction
                    sell_qty = sell_qty_raw if is_crypto(symbol) else round(sell_qty_raw, 2)
                    sell_qty = min(sell_qty, qty)  # Can't sell more than we have

                    if sell_qty > 0:
                        reason = (f"SCALED_EXIT Tier {i+1}: "
                                 f"gain {gain_pct:.1%} >= target {tier_pct:.1%}, "
                                 f"selling {sell_qty} ({tier_fraction:.0%} of original)")
                        logger.info(f"{symbol}: {reason}")
                        order = _submit_sell(api, symbol, sell_qty, tif)
                        if order:
                            sym_state['tiers_sold'].append(i)
                            state_changed = True
                            actions.append({
                                'symbol': symbol, 'action': f'SCALED_EXIT_T{i+1}',
                                'reason': reason, 'qty': sell_qty,
                                'price': current_price, 'order_id': order.id,
                                'status': 'submitted',
                            })
                            qty -= sell_qty  # Update remaining qty for next tier checks

            continue  # Skip legacy take profit / trailing stop when scaled exits are active

        # ---- LEGACY TAKE PROFIT (only when scaled exits disabled) ----
        target_price = entry_price * (1 + pos_tp_pct)
        if current_price >= target_price:
            reason = f"Price ${current_price:.2f} >= Target ${target_price:.2f} ({unrealized_pl_pct*100:+.1f}%)"
            logger.warning(f"TAKE_PROFIT {symbol}: {reason}")
            sell_qty = abs(qty) if is_crypto(symbol) else round(abs(qty), 2)
            order = _submit_sell(api, symbol, sell_qty, tif)
            actions.append({
                'symbol': symbol, 'action': 'TAKE_PROFIT', 'reason': reason,
                'qty': sell_qty, 'price': current_price,
                'order_id': order.id if order else None,
                'status': 'submitted' if order else 'failed',
            })
            continue

        # ---- LEGACY TRAILING STOP (only when scaled exits disabled) ----
        if use_trailing and unrealized_pl_pct > trail_pct:
            trail_trigger = entry_price * (1 + unrealized_pl_pct - trail_pct)
            if current_price <= trail_trigger:
                reason = f"Trailing stop triggered at ${current_price:.2f}"
                logger.warning(f"TRAILING_STOP {symbol}: {reason}")
                sell_qty = abs(qty) if is_crypto(symbol) else round(abs(qty), 2)
                order = _submit_sell(api, symbol, sell_qty, tif)
                actions.append({
                    'symbol': symbol, 'action': 'TRAILING_STOP', 'reason': reason,
                    'qty': sell_qty, 'price': current_price,
                    'order_id': order.id if order else None,
                    'status': 'submitted' if order else 'failed',
                })
                continue

        logger.debug(f"{symbol}: OK - ${current_price:.2f}, Stop ${stop_price:.2f}, P&L: {unrealized_pl_pct*100:+.1f}%")

    # Clean up exit state for positions that no longer exist
    if use_scaled:
        stale = [s for s in exit_state if s not in open_symbols]
        for s in stale:
            del exit_state[s]
            state_changed = True

        if state_changed:
            _save_exit_state(exit_state)

    if actions:
        logger.info(f"Position monitor took {len(actions)} actions")

    return actions


def is_crypto(symbol: str) -> bool:
    """Check if symbol is a crypto pair."""
    return '/' in symbol or symbol.endswith('USD') and symbol not in ['USD']


def get_position_summary(api) -> str:
    """Get a formatted summary of all positions."""
    try:
        positions = api.list_positions()
    except Exception as e:
        return f"Error fetching positions: {e}"

    if not positions:
        return "No open positions"

    lines = ["POSITION SUMMARY", "=" * 50]
    total_value = 0
    total_pnl = 0

    for pos in positions:
        pnl = float(pos.unrealized_pl)
        pnl_pct = float(pos.unrealized_plpc) * 100
        value = float(pos.market_value)

        lines.append(f"{'UP' if pnl >= 0 else 'DN'} {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f}")
        lines.append(f"   Current: ${float(pos.current_price):.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

        total_value += value
        total_pnl += pnl

    lines.append("=" * 50)
    lines.append(f"{'UP' if total_pnl >= 0 else 'DN'} Total: ${total_value:,.2f} | P&L: ${total_pnl:+,.2f}")

    return "\n".join(lines)
