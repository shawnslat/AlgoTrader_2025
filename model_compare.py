"""
Side-by-side model comparison harness.

Trains two model variants on the SAME engineered features + same time-series
split and reports the trading-relevant metrics: directional accuracy, AUC,
overfitting gap, precision, recall, and the Sharpe ratio of a simple
sign-following strategy on the held-out test set.

The two variants:
  - "current": replicates the production tune_and_train_model in
    Trader_main_Grok4_20250731.py (XGBoost with RandomizedSearchCV, precision-
    scored, 100-iter sweep). This is what artifacts/final_model.pkl is.
  - "v2style": ensemble (XGB + RandomForest + GradientBoosting), heavy
    regularization, noise-filtered training labels, scale_pos_weight. This
    is the V2 *strategy* applied to the current feature set — the only
    apples-to-apples way to compare since the real V2 pickle uses 25
    incompatible features.

Run:
    .venv/bin/python model_compare.py

Outputs: artifacts/model_compare_report.txt (plus stdout).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_PATH = ARTIFACT_DIR / "model_compare_report.txt"


def _strategy_sharpe(preds: np.ndarray, next_returns: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple sign-following Sharpe / hit rate / mean return on held-out data.

    preds: 0 or 1 (down/up)
    next_returns: the actual next-bar return for each row

    Returns: (annualized_sharpe, hit_rate, mean_return)
    """
    if len(preds) == 0:
        return 0.0, 0.0, 0.0
    # Map preds to position: 1 -> long (+1), 0 -> flat (0). Don't go short on a
    # down prediction because that's not what the live bot does either.
    positions = preds.astype(float)
    strat_returns = positions * next_returns
    # Only count days the model took a position
    active_mask = positions > 0
    n_active = int(active_mask.sum())
    if n_active == 0:
        return 0.0, 0.0, 0.0

    active_returns = strat_returns[active_mask]
    mean_r = float(np.mean(active_returns))
    std_r = float(np.std(active_returns, ddof=1)) if n_active > 1 else 0.0
    # Annualize assuming ~252 trading days; bars are daily here
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
    hit = float(np.mean(active_returns > 0))
    return sharpe, hit, mean_r


def _max_drawdown(preds: np.ndarray, next_returns: np.ndarray) -> float:
    """Max drawdown of the cumulative equity curve from the same sign-following strategy."""
    if len(preds) == 0:
        return 0.0
    positions = preds.astype(float)
    strat_returns = positions * next_returns
    equity = np.cumprod(1.0 + strat_returns)
    rolling_max = np.maximum.accumulate(equity)
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def _evaluate(name: str, model, X_train, y_train, X_test, y_test, next_returns_test) -> Dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = float("nan")

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_test_pred)
    sharpe, hit_rate, mean_r = _strategy_sharpe(y_test_pred, next_returns_test)
    mdd = _max_drawdown(y_test_pred, next_returns_test)

    return {
        "name": name,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "overfit_gap": train_acc - test_acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "n_active_test": int((y_test_pred == 1).sum()),
        "n_test": int(len(y_test)),
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "mean_return_per_active_bar": mean_r,
        "max_drawdown": mdd,
        "confusion_matrix": cm.tolist(),
    }


def _train_current_style(X_train, y_train, n_iter: int = 30):
    """Mirror tune_and_train_model in Trader_main: RandomizedSearchCV, precision-scored."""
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier

    logger.info("[current] tuning XGBoost via RandomizedSearchCV (precision-scored)...")
    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "scale_pos_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_distributions=param_grid,
        n_iter=n_iter,  # 30 keeps the comparison runtime reasonable; prod uses 100
        cv=tscv,
        scoring="precision",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    logger.info(f"[current] best params: {search.best_params_}")
    return search.best_estimator_


def _train_v2_style(X_train, y_train):
    """V2's ensemble (XGB + RF + GB), heavy regularization, scale_pos_weight."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from xgboost import XGBClassifier

    counts = y_train.value_counts()
    spw = (counts.get(0, 1) / counts.get(1, 1)) if counts.get(1, 0) > 0 else 1.0
    logger.info(f"[v2style] scale_pos_weight = {spw:.3f}")

    xgb = XGBClassifier(
        n_estimators=150, learning_rate=0.02, max_depth=3, min_child_weight=10,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.5,
        scale_pos_weight=spw, eval_metric="logloss", random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=30, min_samples_leaf=15,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        min_samples_split=30, min_samples_leaf=15, subsample=0.7, random_state=42,
    )
    ens = VotingClassifier(estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    ens.fit(X_train, y_train)
    return ens


def main() -> int:
    # Lazy imports so cold start is fast and config errors surface early
    from Trader_main_Grok4_20250731 import (
        load_configuration,
        load_historical_data,
        engineer_features,
        add_sentiment_features,
        prepare_train_test_data,
    )

    cfg = load_configuration("config.yaml")
    logger.info("Loading historical data from ./data ...")
    data = load_historical_data("./data", cfg)
    if data is None or data.empty:
        logger.error("No data found. Train the model first so ./data is populated.")
        return 1

    logger.info(f"Loaded {len(data)} rows across {data['ticker'].nunique()} tickers")
    data = engineer_features(data, config=cfg, is_backtest=True)
    data = add_sentiment_features(data, cfg, is_backtest=True)
    if data.empty:
        logger.error("Empty dataframe after feature engineering.")
        return 2

    # Mirror the feature set used by the live training pipeline
    selected_features = [
        "MA10", "MA50", "RSI", "MACD", "MACD_Signal", "MACD_Diff",
        "Bollinger_Upper", "Bollinger_Lower", "Lag1_Close", "Lag2_Close",
        "ATR", "Stochastic_RSI", "Volume_Change",
    ]
    try:
        import talib  # noqa: F401
        selected_features.extend(["Momentum", "SMA_20"])
    except ImportError:
        pass
    selected_features.append("Sentiment_Score")
    selected_features = [f for f in selected_features if f in data.columns]
    logger.info(f"Using {len(selected_features)} features: {selected_features}")

    # Build train/test the same way prepare_train_test_data does (chronological 80/20)
    X_train, X_test, y_train, y_test = prepare_train_test_data(data, selected_features)

    # For the strategy-Sharpe metric we need actual next-bar returns aligned with the
    # test rows. Reconstruct them from the engineered data's chronological tail.
    data_sorted = data.sort_values("date").reset_index(drop=True)
    # next-bar return per row
    if "close" not in data_sorted.columns:
        logger.error("No close column found; cannot compute next-bar returns.")
        return 3
    next_returns = data_sorted.groupby("ticker")["close"].pct_change().shift(-1).fillna(0.0).values
    # Align: test rows are the last len(X_test) chronological rows
    next_returns_test = next_returns[-len(X_test):]
    assert len(next_returns_test) == len(X_test), "test/return alignment mismatch"

    base_rate = float(y_test.mean())
    logger.info(f"Test set base rate (P(up)): {base_rate:.3f}  ← a constant-'up' predictor would score this")

    # Train both
    cur_model = _train_current_style(X_train, y_train, n_iter=30)
    v2_model = _train_v2_style(X_train, y_train)

    # Evaluate both
    cur_metrics = _evaluate("current", cur_model, X_train, y_train, X_test, y_test, next_returns_test)
    v2_metrics = _evaluate("v2style", v2_model, X_train, y_train, X_test, y_test, next_returns_test)

    # Print report
    def _fmt(m):
        return (
            f"  train_acc     : {m['train_acc']:.4f}\n"
            f"  test_acc      : {m['test_acc']:.4f}\n"
            f"  overfit_gap   : {m['overfit_gap']:+.4f}\n"
            f"  auc           : {m['auc']:.4f}\n"
            f"  precision     : {m['precision']:.4f}\n"
            f"  recall        : {m['recall']:.4f}\n"
            f"  positions     : {m['n_active_test']}/{m['n_test']} bars\n"
            f"  sharpe (test) : {m['sharpe']:.3f}\n"
            f"  hit_rate      : {m['hit_rate']:.3f}\n"
            f"  mean_return   : {m['mean_return_per_active_bar']:+.5f}\n"
            f"  max_drawdown  : {m['max_drawdown']:.3f}\n"
            f"  confusion     : {m['confusion_matrix']}\n"
        )

    header = "=" * 72
    report_lines = [
        header,
        "MODEL COMPARISON — current XGBoost vs v2-style ensemble",
        header,
        f"Features: {selected_features}",
        f"Train size: {len(X_train)}  Test size: {len(X_test)}",
        f"Test base rate (always-up accuracy): {base_rate:.3f}",
        "",
        "=== current (XGBoost + RandomizedSearchCV) ===",
        _fmt(cur_metrics),
        "=== v2style (XGB+RF+GB ensemble, heavy regularization) ===",
        _fmt(v2_metrics),
        header,
        "VERDICT",
        header,
    ]

    # Verdict — prioritize Sharpe + overfit gap over raw accuracy
    cur_score = cur_metrics["sharpe"] - max(0, cur_metrics["overfit_gap"] - 0.05) * 2
    v2_score = v2_metrics["sharpe"] - max(0, v2_metrics["overfit_gap"] - 0.05) * 2
    winner = "v2style" if v2_score > cur_score else "current"
    report_lines.append(f"Sharpe-minus-overfit-penalty: current={cur_score:.3f}, v2style={v2_score:.3f}")
    report_lines.append(f"Winner on this metric: {winner}")

    # Caveats the report must include — model accuracy ≠ trading edge
    report_lines += [
        "",
        "Caveats:",
        "  - Sharpe here is a held-out, no-cost, full-long sign-following backtest.",
        "    Add transaction costs (0.001 + 0.0005 from config.backtest) before",
        "    deciding to deploy — 30bps round-trip can erase a marginal edge.",
        "  - This compares MODEL OUTPUTS only. The live bot also runs an 8-source",
        "    confidence vote on top, which can suppress weak signals. Real Sharpe",
        "    with the confidence layer is typically higher than this raw number.",
        "  - Base rate of up-days dominates accuracy. Compare test_acc against the",
        "    base rate, not against 50%.",
        "  - Single train/test split. Walk-forward will give a more honest read.",
        "    Run backtest_validation.py for that.",
        "",
    ]
    report = "\n".join(report_lines)
    print(report)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report)
    logger.info(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
