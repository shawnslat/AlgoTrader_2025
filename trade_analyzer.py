"""
Trade Analyzer — Pulls trade history from Alpaca, FIFO matches buys to sells,
computes PnL and analytics, and generates an HTML report.

Usage:
    python trade_analyzer.py
"""

import sqlite3
import csv
import json
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'logs' / 'trades.db'
REPORT_PATH = BASE_DIR / 'logs' / 'trade_report.html'
CSV_DIR = BASE_DIR / 'logs' / 'trade_logs'
DATA_DIR = BASE_DIR / 'data'

CRYPTO_SUFFIXES = ('BTCUSD', 'ETHUSD', 'SOLUSD', 'DOGEUSD', 'SHIBUSD',
                   'AVAXUSD', 'DOTUSD', 'LINKUSD', 'MATICUSD', 'ADAUSD')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    return symbol.replace('/', '').replace('-', '').upper()


def is_crypto(symbol: str) -> bool:
    normed = normalize_symbol(symbol)
    return normed in CRYPTO_SUFFIXES or (normed.endswith('USD') and len(normed) > 4)


def symbol_to_ohlcv_filename(symbol: str) -> str:
    crypto_map = {
        'BTCUSD': 'BTC-USD', 'ETHUSD': 'ETH-USD', 'SOLUSD': 'SOL-USD',
    }
    return crypto_map.get(symbol, symbol)


def parse_timestamp(ts_str: str) -> datetime:
    ts_str = ts_str.strip()
    for fmt in (
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    ):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return pd.Timestamp(ts_str).to_pydatetime()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fills (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            alpaca_activity_id  TEXT UNIQUE,
            alpaca_order_id     TEXT,
            symbol              TEXT NOT NULL,
            raw_symbol          TEXT,
            side                TEXT NOT NULL,
            qty                 REAL NOT NULL,
            price               REAL NOT NULL,
            total_value         REAL,
            timestamp           TEXT NOT NULL,
            source              TEXT NOT NULL DEFAULT 'alpaca',
            asset_class         TEXT,
            created_at          TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
        CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp);

        CREATE TABLE IF NOT EXISTS matched_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            buy_fill_id     INTEGER NOT NULL,
            sell_fill_id    INTEGER NOT NULL,
            qty             REAL NOT NULL,
            buy_price       REAL NOT NULL,
            sell_price      REAL NOT NULL,
            buy_timestamp   TEXT NOT NULL,
            sell_timestamp  TEXT NOT NULL,
            hold_seconds    INTEGER,
            realized_pnl    REAL,
            pnl_pct         REAL,
            asset_class     TEXT,
            mfe_price       REAL,
            mae_price       REAL,
            mfe_pnl         REAL,
            mae_pnl         REAL,
            mfe_capture     REAL,
            exit_verdict    TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_matched_symbol ON matched_trades(symbol);

        CREATE TABLE IF NOT EXISTS open_positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            fill_id         INTEGER NOT NULL,
            remaining_qty   REAL NOT NULL,
            buy_price       REAL NOT NULL,
            buy_timestamp   TEXT NOT NULL,
            asset_class     TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data Import
# ---------------------------------------------------------------------------

def import_from_alpaca(api, conn: sqlite3.Connection) -> int:
    inserted = 0
    page_token = None
    while True:
        kwargs = dict(activity_types='FILL', direction='asc', page_size=100)
        if page_token:
            kwargs['page_token'] = page_token
        try:
            activities = api.get_activities(**kwargs)
        except Exception as e:
            logger.warning(f"  Alpaca API error: {e}")
            break
        if not activities:
            break
        for act in activities:
            sym_raw = act.symbol
            sym = normalize_symbol(sym_raw)
            side = act.side
            qty = float(act.qty)
            price = float(act.price)
            ts = str(act.transaction_time)
            order_id = getattr(act, 'order_id', None)
            activity_id = act.id
            asset_cls = 'crypto' if is_crypto(sym) else 'stock'
            conn.execute("""
                INSERT OR IGNORE INTO fills
                (alpaca_activity_id, alpaca_order_id, symbol, raw_symbol,
                 side, qty, price, total_value, timestamp, source, asset_class)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'alpaca', ?)
            """, (activity_id, order_id, sym, sym_raw,
                  side, qty, price, qty * price, ts, asset_cls))
            inserted += 1
        page_token = activities[-1].id
        if len(activities) < 100:
            break
    conn.commit()
    return inserted


def import_from_csvs(csv_dir: Path, conn: sqlite3.Connection) -> int:
    inserted = 0
    for csv_file in sorted(csv_dir.glob('trade_log_*.csv')):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = (row.get('ticker') or '').strip()
                ts = (row.get('timestamp') or '').strip()
                if not ticker or ts.startswith('Summary') or ticker == '':
                    continue
                sym = normalize_symbol(ticker)
                side = row['type'].strip().lower()
                price = float(row['price'])
                qty = float(row['quantity'])
                asset_cls = 'crypto' if is_crypto(sym) else 'stock'
                synth_id = f"csv_{ts}_{sym}_{side}_{qty}"
                conn.execute("""
                    INSERT OR IGNORE INTO fills
                    (alpaca_activity_id, symbol, raw_symbol, side, qty, price,
                     total_value, timestamp, source, asset_class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'csv', ?)
                """, (synth_id, sym, ticker, side, qty, price,
                      qty * price, ts, asset_cls))
                inserted += 1
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# FIFO Matching
# ---------------------------------------------------------------------------

def run_fifo_matching(conn: sqlite3.Connection):
    conn.execute("DELETE FROM matched_trades")
    conn.execute("DELETE FROM open_positions")

    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM fills ORDER BY symbol")]

    for symbol in symbols:
        fills = conn.execute("""
            SELECT id, side, qty, price, timestamp, asset_class
            FROM fills WHERE symbol = ?
            ORDER BY timestamp ASC, id ASC
        """, (symbol,)).fetchall()

        buy_queue = []  # [[fill_id, remaining_qty, price, ts, asset_class], ...]

        for fill_id, side, qty, price, ts, asset_cls in fills:
            if side == 'buy':
                buy_queue.append([fill_id, qty, price, ts, asset_cls])
            elif side == 'sell':
                sell_remaining = qty
                while sell_remaining > 1e-10 and buy_queue:
                    buy = buy_queue[0]
                    match_qty = min(sell_remaining, buy[1])

                    buy_dt = parse_timestamp(buy[3])
                    sell_dt = parse_timestamp(ts)
                    # Make both naive for subtraction
                    if buy_dt.tzinfo is not None:
                        buy_dt = buy_dt.replace(tzinfo=None)
                    if sell_dt.tzinfo is not None:
                        sell_dt = sell_dt.replace(tzinfo=None)
                    hold_secs = int((sell_dt - buy_dt).total_seconds())

                    realized_pnl = (price - buy[2]) * match_qty
                    pnl_pct = ((price - buy[2]) / buy[2]) * 100 if buy[2] > 0 else 0

                    conn.execute("""
                        INSERT INTO matched_trades
                        (symbol, buy_fill_id, sell_fill_id, qty,
                         buy_price, sell_price, buy_timestamp, sell_timestamp,
                         hold_seconds, realized_pnl, pnl_pct, asset_class)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, buy[0], fill_id, match_qty,
                          buy[2], price, buy[3], ts,
                          hold_secs, realized_pnl, pnl_pct,
                          asset_cls or buy[4]))

                    sell_remaining -= match_qty
                    buy[1] -= match_qty
                    if buy[1] <= 1e-10:
                        buy_queue.pop(0)

        for buy in buy_queue:
            if buy[1] > 1e-10:
                conn.execute("""
                    INSERT INTO open_positions
                    (symbol, fill_id, remaining_qty, buy_price, buy_timestamp, asset_class)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, buy[0], buy[1], buy[2], buy[3], buy[4]))

    conn.commit()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_summary(matched_df: pd.DataFrame, open_df: pd.DataFrame, api) -> dict:
    if matched_df.empty:
        return dict(total_trades=0, wins=0, losses=0, win_rate=0,
                    total_realized_pnl=0, avg_win=0, avg_loss=0,
                    profit_factor=0, avg_hold_hours=0, total_unrealized_pnl=0,
                    best_trade=None, worst_trade=None)

    wins = matched_df[matched_df['realized_pnl'] > 0]
    losses = matched_df[matched_df['realized_pnl'] <= 0]
    total_win = wins['realized_pnl'].sum() if len(wins) else 0
    total_loss = abs(losses['realized_pnl'].sum()) if len(losses) else 0
    pf = total_win / total_loss if total_loss > 0 else float('inf')

    unrealized = 0.0
    if not open_df.empty:
        try:
            positions = api.list_positions()
            pmap = {normalize_symbol(p.symbol): float(p.current_price) for p in positions}
            for _, row in open_df.iterrows():
                cur = pmap.get(row['symbol'], row['buy_price'])
                unrealized += (cur - row['buy_price']) * row['remaining_qty']
        except Exception:
            pass

    avg_hold = matched_df['hold_seconds'].mean() / 3600 if not matched_df.empty else 0

    best = matched_df.loc[matched_df['realized_pnl'].idxmax()].to_dict() if not matched_df.empty else None
    worst = matched_df.loc[matched_df['realized_pnl'].idxmin()].to_dict() if not matched_df.empty else None

    return dict(
        total_trades=len(matched_df), wins=len(wins), losses=len(losses),
        win_rate=len(wins) / len(matched_df) * 100,
        total_realized_pnl=matched_df['realized_pnl'].sum(),
        avg_win=wins['realized_pnl'].mean() if len(wins) else 0,
        avg_loss=losses['realized_pnl'].mean() if len(losses) else 0,
        profit_factor=pf, avg_hold_hours=avg_hold,
        total_unrealized_pnl=unrealized,
        best_trade=best, worst_trade=worst,
    )


def compute_pnl_by_asset(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    g = df.groupby('symbol').agg(
        total_pnl=('realized_pnl', 'sum'),
        count=('id', 'count'),
        wins=('realized_pnl', lambda x: (x > 0).sum()),
        avg_pnl=('realized_pnl', 'mean'),
    ).reset_index()
    g['win_rate'] = g['wins'] / g['count'] * 100
    return g.sort_values('total_pnl', ascending=False).to_dict('records')


def compute_pnl_by_day(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d['sell_dt'] = pd.to_datetime(d['sell_timestamp'], format='mixed', utc=True)
    d['dow'] = d['sell_dt'].dt.day_name()
    g = d.groupby('dow')['realized_pnl'].agg(['sum', 'count', 'mean'])
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    g = g.reindex(order).fillna(0)
    return g.to_dict('index')


def compute_pnl_by_hour(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d['sell_dt'] = pd.to_datetime(d['sell_timestamp'], format='mixed', utc=True)
    d['hour'] = d['sell_dt'].dt.hour
    g = d.groupby('hour')['realized_pnl'].agg(['sum', 'count', 'mean'])
    return g.to_dict('index')


def compute_pnl_by_duration(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d['hold_hours'] = d['hold_seconds'] / 3600
    bins = [0, 1, 4, 24, 72, 168, float('inf')]
    labels = ['<1h', '1-4h', '4-24h', '1-3d', '3-7d', '>7d']
    d['bucket'] = pd.cut(d['hold_hours'], bins=bins, labels=labels)
    g = d.groupby('bucket', observed=True)['realized_pnl'].agg(['sum', 'count', 'mean'])
    return g.to_dict('index')


def compute_streaks(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(max_win_streak=0, max_loss_streak=0, current_streak=0, current_type=None)
    d = df.sort_values('sell_timestamp')
    max_w = max_l = cur = 0
    cur_type = None
    for pnl in d['realized_pnl']:
        w = pnl > 0
        if w:
            if cur_type == 'win':
                cur += 1
            else:
                cur = 1
                cur_type = 'win'
            max_w = max(max_w, cur)
        else:
            if cur_type == 'loss':
                cur += 1
            else:
                cur = 1
                cur_type = 'loss'
            max_l = max(max_l, cur)
    return dict(max_win_streak=max_w, max_loss_streak=max_l,
                current_streak=cur, current_type=cur_type)


def compute_equity_curve(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    d = df.sort_values('sell_timestamp').copy()
    d['cum_pnl'] = d['realized_pnl'].cumsum()
    d['opt_pnl'] = d['realized_pnl'].clip(lower=0).cumsum()
    return [dict(ts=row['sell_timestamp'], cum=round(row['cum_pnl'], 2),
                 opt=round(row['opt_pnl'], 2), sym=row['symbol'],
                 pnl=round(row['realized_pnl'], 2))
            for _, row in d.iterrows()]


def compute_mfe_mae(matched_df: pd.DataFrame, conn: sqlite3.Connection):
    if matched_df.empty:
        return

    ohlcv_cache = {}
    for csv_file in DATA_DIR.glob('*.csv'):
        try:
            ohlcv = pd.read_csv(csv_file, parse_dates=['time'])
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            ohlcv_cache[normalize_symbol(csv_file.stem)] = ohlcv
        except Exception:
            continue

    for _, trade in matched_df.iterrows():
        sym = trade['symbol']
        ohlcv_key = normalize_symbol(symbol_to_ohlcv_filename(sym))
        ohlcv = ohlcv_cache.get(sym)
        if ohlcv is None:
            ohlcv = ohlcv_cache.get(ohlcv_key)
        if ohlcv is None or ohlcv.empty:
            continue

        buy_dt = pd.Timestamp(trade['buy_timestamp']).tz_localize(None) if pd.Timestamp(trade['buy_timestamp']).tzinfo is None else pd.Timestamp(trade['buy_timestamp']).tz_convert(None)
        sell_dt = pd.Timestamp(trade['sell_timestamp']).tz_localize(None) if pd.Timestamp(trade['sell_timestamp']).tzinfo is None else pd.Timestamp(trade['sell_timestamp']).tz_convert(None)
        buy_date = buy_dt.normalize()
        sell_date = sell_dt.normalize()

        mask = (ohlcv['time'] >= buy_date) & (ohlcv['time'] <= sell_date)
        period = ohlcv[mask]
        if period.empty:
            continue

        buy_price = trade['buy_price']
        sell_price = trade['sell_price']
        qty = trade['qty']
        realized_pnl = trade['realized_pnl']

        mfe_price = period['high'].max()
        mae_price = period['low'].min()
        mfe_pnl = (mfe_price - buy_price) * qty
        mae_pnl = (mae_price - buy_price) * qty
        mfe_capture = realized_pnl / mfe_pnl if mfe_pnl > 0 else 0.0
        mfe_capture = max(0.0, min(1.0, mfe_capture))

        # Post-exit price action (5 candles after sell)
        post = ohlcv[ohlcv['time'] > sell_date].head(5)
        post_max = post['high'].max() if not post.empty else sell_price

        if realized_pnl > 0:
            if mfe_capture >= 0.7:
                verdict = 'held_well'
            elif post_max > sell_price * 1.02:
                verdict = 'exited_early'
            else:
                verdict = 'good_exit'
        else:
            if mae_price < buy_price and sell_price > mae_price:
                verdict = 'recovered_partially'
            else:
                verdict = 'stop_loss_hit'

        conn.execute("""
            UPDATE matched_trades SET
                mfe_price=?, mae_price=?, mfe_pnl=?, mae_pnl=?,
                mfe_capture=?, exit_verdict=?
            WHERE id=?
        """, (mfe_price, mae_price, mfe_pnl, mae_pnl,
              mfe_capture, verdict, trade['id']))

    conn.commit()


def compute_open_positions(open_df: pd.DataFrame, api) -> list:
    if open_df.empty:
        return []
    try:
        positions = api.list_positions()
        pmap = {normalize_symbol(p.symbol): float(p.current_price) for p in positions}
    except Exception:
        pmap = {}

    result = []
    for _, row in open_df.iterrows():
        cur = pmap.get(row['symbol'], row['buy_price'])
        unrealized = (cur - row['buy_price']) * row['remaining_qty']
        pnl_pct = ((cur - row['buy_price']) / row['buy_price'] * 100) if row['buy_price'] > 0 else 0
        result.append(dict(
            symbol=row['symbol'], qty=row['remaining_qty'],
            buy_price=row['buy_price'], current_price=cur,
            unrealized_pnl=round(unrealized, 2), pnl_pct=round(pnl_pct, 2),
            buy_timestamp=row['buy_timestamp'],
            asset_class=row['asset_class'],
        ))
    return result


def run_analysis(conn: sqlite3.Connection, api) -> dict:
    matched_df = pd.read_sql_query("SELECT * FROM matched_trades", conn)
    open_df = pd.read_sql_query("SELECT * FROM open_positions", conn)

    # MFE/MAE (updates DB in place)
    compute_mfe_mae(matched_df, conn)
    # Re-read after MFE update
    matched_df = pd.read_sql_query("SELECT * FROM matched_trades", conn)

    return dict(
        summary=compute_summary(matched_df, open_df, api),
        pnl_by_asset=compute_pnl_by_asset(matched_df),
        pnl_by_day=compute_pnl_by_day(matched_df),
        pnl_by_hour=compute_pnl_by_hour(matched_df),
        pnl_by_duration=compute_pnl_by_duration(matched_df),
        streaks=compute_streaks(matched_df),
        equity_curve=compute_equity_curve(matched_df),
        open_positions=compute_open_positions(open_df, api),
        trades=matched_df.to_dict('records'),
    )


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _fmt_pnl(val: float) -> str:
    cls = 'win' if val >= 0 else 'loss'
    return f'<span class="{cls}">${val:+,.2f}</span>'


def _fmt_pct(val: float) -> str:
    cls = 'win' if val >= 0 else 'loss'
    return f'<span class="{cls}">{val:+.1f}%</span>'


def _verdict_badge(v: str) -> str:
    if not v:
        return ''
    label = v.replace('_', ' ').title()
    return f'<span class="badge badge-{v}">{label}</span>'


def _hold_fmt(secs: float) -> str:
    if pd.isna(secs) or secs is None:
        return '-'
    h = secs / 3600
    if h < 1:
        return f'{secs / 60:.0f}m'
    if h < 24:
        return f'{h:.1f}h'
    return f'{h / 24:.1f}d'


def generate_report(results: dict, output_path: str):
    s = results['summary']
    ec = results['equity_curve']
    trades = results['trades']
    open_pos = results['open_positions']
    pnl_asset = results['pnl_by_asset']
    pnl_day = results['pnl_by_day']
    pnl_hour = results['pnl_by_hour']
    streaks = results['streaks']

    # Equity curve data
    ec_labels = json.dumps([e['ts'][:10] for e in ec])
    ec_actual = json.dumps([e['cum'] for e in ec])
    ec_optimal = json.dumps([e['opt'] for e in ec])

    # PnL by asset chart
    asset_labels = json.dumps([a['symbol'] for a in pnl_asset])
    asset_values = json.dumps([round(a['total_pnl'], 2) for a in pnl_asset])
    asset_colors = json.dumps(['#4CAF50' if a['total_pnl'] >= 0 else '#f44336' for a in pnl_asset])

    # Day heatmap
    day_cells = ''
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        info = pnl_day.get(day, {'sum': 0, 'count': 0, 'mean': 0})
        val = info.get('sum', 0) if isinstance(info, dict) else 0
        cnt = info.get('count', 0) if isinstance(info, dict) else 0
        intensity = min(abs(val) / 50, 1.0) if val != 0 else 0
        color = f'rgba(76,175,80,{intensity:.2f})' if val >= 0 else f'rgba(244,67,54,{intensity:.2f})'
        day_cells += f'<td style="background:{color};text-align:center;padding:12px">'
        day_cells += f'<div style="font-weight:bold">{day[:3]}</div>'
        day_cells += f'<div>${val:+,.2f}</div><div style="font-size:0.8em;opacity:0.7">{int(cnt)} trades</div></td>'

    # Hour heatmap
    hour_cells = ''
    for h in range(24):
        info = pnl_hour.get(h, {'sum': 0, 'count': 0})
        val = info.get('sum', 0) if isinstance(info, dict) else 0
        cnt = info.get('count', 0) if isinstance(info, dict) else 0
        if cnt == 0:
            hour_cells += f'<td style="text-align:center;padding:6px;opacity:0.3"><div>{h:02d}</div><div>-</div></td>'
        else:
            intensity = min(abs(val) / 30, 1.0)
            color = f'rgba(76,175,80,{intensity:.2f})' if val >= 0 else f'rgba(244,67,54,{intensity:.2f})'
            hour_cells += f'<td style="background:{color};text-align:center;padding:6px">'
            hour_cells += f'<div>{h:02d}</div><div>${val:+,.2f}</div></td>'

    # Trade log rows
    trade_rows = ''
    for t in sorted(trades, key=lambda x: x.get('sell_timestamp', ''), reverse=True):
        pnl_cls = 'win' if t['realized_pnl'] >= 0 else 'loss'
        cap = f"{t['mfe_capture']:.0%}" if t.get('mfe_capture') is not None else '-'
        trade_rows += f"""<tr class="trade-row {pnl_cls}">
            <td>{t['symbol']}</td>
            <td>${t['buy_price']:,.2f}</td>
            <td>${t['sell_price']:,.2f}</td>
            <td>{t['qty']:.4f}</td>
            <td class="{pnl_cls}">${t['realized_pnl']:+,.2f}</td>
            <td class="{pnl_cls}">{t['pnl_pct']:+.1f}%</td>
            <td>{_hold_fmt(t['hold_seconds'])}</td>
            <td>{cap}</td>
            <td>{_verdict_badge(t.get('exit_verdict'))}</td>
        </tr>"""

    # Open positions rows
    open_rows = ''
    for p in open_pos:
        pcls = 'win' if p['unrealized_pnl'] >= 0 else 'loss'
        open_rows += f"""<tr>
            <td>{p['symbol']}</td>
            <td>{p['qty']:.4f}</td>
            <td>${p['buy_price']:,.2f}</td>
            <td>${p['current_price']:,.2f}</td>
            <td class="{pcls}">${p['unrealized_pnl']:+,.2f}</td>
            <td class="{pcls}">{p['pnl_pct']:+.1f}%</td>
        </tr>"""

    # PnL by asset rows
    asset_rows = ''
    for a in pnl_asset:
        pcls = 'win' if a['total_pnl'] >= 0 else 'loss'
        asset_rows += f"""<tr>
            <td>{a['symbol']}</td>
            <td>{int(a['count'])}</td>
            <td>{a['win_rate']:.0f}%</td>
            <td class="{pcls}">${a['total_pnl']:+,.2f}</td>
            <td class="{pcls}">${a['avg_pnl']:+,.2f}</td>
        </tr>"""

    pf_display = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else "∞"
    best_sym = s['best_trade']['symbol'] if s['best_trade'] else '-'
    best_pnl = s['best_trade']['realized_pnl'] if s['best_trade'] else 0
    worst_sym = s['worst_trade']['symbol'] if s['worst_trade'] else '-'
    worst_pnl = s['worst_trade']['realized_pnl'] if s['worst_trade'] else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Analyzer Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {{
    --bg-primary: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-card-hover: #1e1e35;
    --border: #2a2a40;
    --text: #e0e0e0;
    --text-dim: #888;
    --green: #4CAF50;
    --red: #f44336;
    --yellow: #FFC107;
    --blue: #2196F3;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: var(--bg-primary); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; }}
h1 {{ font-size: 1.6em; margin-bottom: 5px; }}
h2 {{ font-size: 1.2em; margin-bottom: 15px; color: var(--text-dim); }}
.header {{ text-align: center; padding: 30px 0 20px; }}
.header small {{ color: var(--text-dim); }}
.grid {{ display: grid; gap: 15px; margin-bottom: 25px; }}
.grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
.grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }}
.card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
.card:hover {{ background: var(--bg-card-hover); }}
.stat-label {{ font-size: 0.8em; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-value {{ font-size: 1.8em; font-weight: 700; margin-top: 4px; }}
.win {{ color: var(--green); }}
.loss {{ color: var(--red); }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th {{ text-align: left; padding: 10px; border-bottom: 2px solid var(--border); color: var(--text-dim); font-size: 0.8em; text-transform: uppercase; }}
td {{ padding: 10px; border-bottom: 1px solid var(--border); }}
.trade-row:hover {{ background: var(--bg-card-hover); }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600; }}
.badge-held_well {{ background: rgba(76,175,80,0.15); color: var(--green); }}
.badge-good_exit {{ background: rgba(33,150,243,0.15); color: var(--blue); }}
.badge-exited_early {{ background: rgba(255,193,7,0.15); color: var(--yellow); }}
.badge-stop_loss_hit {{ background: rgba(244,67,54,0.15); color: var(--red); }}
.badge-recovered_partially {{ background: rgba(255,152,0,0.15); color: #FF9800; }}
.chart-container {{ position: relative; height: 300px; }}
.section {{ margin-bottom: 30px; }}
@media (max-width: 768px) {{
    .grid-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2 {{ grid-template-columns: 1fr; }}
    .stat-value {{ font-size: 1.3em; }}
    table {{ font-size: 0.75em; }}
    td, th {{ padding: 6px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>Trade Analyzer Report</h1>
    <small>Generated {datetime.now().strftime('%Y-%m-%d %H:%M ET')} | Alpaca Paper Trading</small>
</div>

<!-- Summary Cards -->
<div class="grid grid-4">
    <div class="card">
        <div class="stat-label">Total Trades</div>
        <div class="stat-value">{s['total_trades']}</div>
    </div>
    <div class="card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value {'win' if s['win_rate'] >= 50 else 'loss'}">{s['win_rate']:.1f}%</div>
        <div style="color:var(--text-dim);font-size:0.85em">{s['wins']}W / {s['losses']}L</div>
    </div>
    <div class="card">
        <div class="stat-label">Realized PnL</div>
        <div class="stat-value {'win' if s['total_realized_pnl'] >= 0 else 'loss'}">${s['total_realized_pnl']:+,.2f}</div>
    </div>
    <div class="card">
        <div class="stat-label">Unrealized PnL</div>
        <div class="stat-value {'win' if s['total_unrealized_pnl'] >= 0 else 'loss'}">${s['total_unrealized_pnl']:+,.2f}</div>
    </div>
    <div class="card">
        <div class="stat-label">Profit Factor</div>
        <div class="stat-value">{pf_display}</div>
    </div>
    <div class="card">
        <div class="stat-label">Avg Hold</div>
        <div class="stat-value">{s['avg_hold_hours']:.1f}h</div>
    </div>
    <div class="card">
        <div class="stat-label">Best Trade</div>
        <div class="stat-value win">${best_pnl:+,.2f}</div>
        <div style="color:var(--text-dim);font-size:0.85em">{best_sym}</div>
    </div>
    <div class="card">
        <div class="stat-label">Worst Trade</div>
        <div class="stat-value loss">${worst_pnl:+,.2f}</div>
        <div style="color:var(--text-dim);font-size:0.85em">{worst_sym}</div>
    </div>
    <div class="card">
        <div class="stat-label">Win Streak</div>
        <div class="stat-value win">{streaks['max_win_streak']}</div>
    </div>
    <div class="card">
        <div class="stat-label">Loss Streak</div>
        <div class="stat-value loss">{streaks['max_loss_streak']}</div>
    </div>
</div>

<!-- Charts -->
<div class="grid grid-2">
    <div class="card section">
        <h2>Equity Curve</h2>
        <div class="chart-container"><canvas id="equityChart"></canvas></div>
    </div>
    <div class="card section">
        <h2>PnL by Ticker</h2>
        <div class="chart-container"><canvas id="assetChart"></canvas></div>
    </div>
</div>

<!-- Heatmaps -->
<div class="card section">
    <h2>PnL by Day of Week</h2>
    <table><tr>{day_cells}</tr></table>
</div>

<div class="card section">
    <h2>PnL by Hour (UTC)</h2>
    <div style="overflow-x:auto"><table><tr>{hour_cells}</tr></table></div>
</div>

<!-- PnL by Ticker Table -->
<div class="card section">
    <h2>Performance by Ticker</h2>
    <table>
        <thead><tr><th>Ticker</th><th>Trades</th><th>Win Rate</th><th>Total PnL</th><th>Avg PnL</th></tr></thead>
        <tbody>{asset_rows}</tbody>
    </table>
</div>

<!-- Open Positions -->
{'<div class="card section"><h2>Open Positions</h2><table><thead><tr><th>Ticker</th><th>Qty</th><th>Entry</th><th>Current</th><th>Unrealized</th><th>PnL %</th></tr></thead><tbody>' + open_rows + '</tbody></table></div>' if open_rows else ''}

<!-- Trade Log -->
<div class="card section">
    <h2>Trade Log</h2>
    <div style="overflow-x:auto">
    <table>
        <thead><tr>
            <th>Ticker</th><th>Entry</th><th>Exit</th><th>Qty</th>
            <th>PnL</th><th>PnL %</th><th>Hold</th><th>MFE Cap</th><th>Verdict</th>
        </tr></thead>
        <tbody>{trade_rows}</tbody>
    </table>
    </div>
</div>

<script>
const chartDefaults = {{
    color: '#e0e0e0',
    borderColor: '#2a2a40',
}};
Chart.defaults.color = '#aaa';
Chart.defaults.borderColor = '#2a2a40';

new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: {ec_labels},
        datasets: [{{
            label: 'Actual PnL',
            data: {ec_actual},
            borderColor: '#4CAF50',
            backgroundColor: 'rgba(76,175,80,0.08)',
            fill: true, tension: 0.3, pointRadius: 2
        }}, {{
            label: 'Optimal PnL',
            data: {ec_optimal},
            borderColor: '#FFC107',
            borderDash: [5, 5],
            fill: false, tension: 0.3, pointRadius: 0
        }}]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ boxWidth: 12 }} }} }},
        scales: {{
            y: {{ ticks: {{ callback: v => '$' + v.toFixed(0) }} }}
        }}
    }}
}});

new Chart(document.getElementById('assetChart'), {{
    type: 'bar',
    data: {{
        labels: {asset_labels},
        datasets: [{{
            label: 'Total PnL',
            data: {asset_values},
            backgroundColor: {asset_colors},
            borderRadius: 4
        }}]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ ticks: {{ callback: v => '$' + v.toFixed(0) }} }}
        }}
    }}
}});
</script>

<div style="text-align:center;padding:30px;color:var(--text-dim);font-size:0.8em">
    Trade Analyzer | Trader_2025 | Data from Alpaca Paper Trading API
</div>

</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Trade Analyzer - Trader_2025")
    print("=" * 60)

    # Load config
    config_path = BASE_DIR / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Init Alpaca API
    api = tradeapi.REST(
        config['alpaca']['api_key'],
        config['alpaca']['api_secret'],
        config['alpaca']['base_url']
    )

    # Init DB
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)

    # Import
    print("\n[1/5] Importing fills from Alpaca API...")
    alpaca_count = import_from_alpaca(api, conn)
    print(f"       {alpaca_count} fill records from Alpaca")

    print("[2/5] Importing fills from CSV trade logs...")
    csv_count = import_from_csvs(CSV_DIR, conn)
    print(f"       {csv_count} fill records from CSVs")

    total = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    print(f"       Total fills in database: {total}")

    # FIFO matching
    print("\n[3/5] Running FIFO trade matching...")
    run_fifo_matching(conn)
    matched = conn.execute("SELECT COUNT(*) FROM matched_trades").fetchone()[0]
    open_ct = conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0]
    print(f"       Matched trades: {matched}")
    print(f"       Open positions: {open_ct}")

    # Analysis
    print("\n[4/5] Running analysis (PnL, streaks, MFE/MAE)...")
    results = run_analysis(conn, api)

    # Report
    print("\n[5/5] Generating HTML report...")
    generate_report(results, str(REPORT_PATH))
    print(f"       Saved: {REPORT_PATH}")

    # Meta
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('last_run', ?)",
                 (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

    # Summary
    s = results['summary']
    print("\n" + "=" * 60)
    print(f"  Total Trades:    {s['total_trades']}")
    print(f"  Win Rate:        {s['win_rate']:.1f}%  ({s['wins']}W / {s['losses']}L)")
    print(f"  Realized PnL:    ${s['total_realized_pnl']:+,.2f}")
    print(f"  Unrealized PnL:  ${s['total_unrealized_pnl']:+,.2f}")
    pf = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else "inf"
    print(f"  Profit Factor:   {pf}")
    print(f"  Avg Hold:        {s['avg_hold_hours']:.1f} hours")
    print("=" * 60)


if __name__ == '__main__':
    main()
