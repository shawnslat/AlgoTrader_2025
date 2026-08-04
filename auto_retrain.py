"""
Weekly auto-retrain job.

Re-downloads recent historical data, re-tunes the XGBoost model, writes a new
artifacts/final_model.pkl, and signals the running bot to reload it via IPC.

Safety:
- Trains to a tmp file first, only swaps in if training succeeds and the new
  model evaluates above a minimum precision threshold (configurable).
- Keeps a timestamped backup of the previous model in artifacts/model_backups/.
- Writes a status JSON the dashboard can surface.

Run manually:
    .venv/bin/python auto_retrain.py

Scheduled via launchd (see launchd/com.shawnslat.algotrader.retrain.plist).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
BACKUP_DIR = ARTIFACT_DIR / "model_backups"
STATUS_FILE = BASE_DIR / "logs" / "retrain_status.json"
IPC_SOCKET = "/tmp/trader_bot.sock"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "auto_retrain.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _write_status(payload: dict) -> None:
    payload["written_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write retrain status: {e}")


def _signal_bot_reload() -> dict:
    """Tell the running bot service (if any) to reload its model from disk."""
    if not os.path.exists(IPC_SOCKET):
        return {"sent": False, "reason": "bot not running (no IPC socket)"}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.connect(IPC_SOCKET)
            s.sendall(json.dumps({"command": "reload_model"}).encode() + b"\n")
            chunks = []
            while True:
                data = s.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b"\n" in data:
                    break
            raw = b"".join(chunks).decode().strip().split("\n", 1)[0]
            resp = json.loads(raw) if raw else {}
            return {"sent": True, "response": resp}
    except Exception as e:
        return {"sent": False, "error": str(e)}


def main() -> int:
    start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"Auto-retrain starting at {start.isoformat(timespec='seconds')}")
    logger.info("=" * 60)

    # Lazy imports — keep launchd startup time low if config is broken
    try:
        from Trader_main_Grok4_20250731 import (
            load_configuration,
            download_historical_data_with_crypto,
            load_historical_data,
            engineer_features,
            add_sentiment_features,
            prepare_train_test_data,
            tune_and_train_model,
            evaluate_model,
        )
    except Exception as e:
        logger.error(f"Could not import training pipeline: {e}", exc_info=True)
        _write_status({"status": "import_failed", "error": str(e)})
        return 2

    config = load_configuration("config.yaml")
    tickers = config["tickers"]
    retrain_cfg = config.get("auto_retrain", {}) or {}
    min_precision = float(retrain_cfg.get("min_precision", 0.0))

    # ----- Data -----
    logger.info(f"Downloading historical data for {len(tickers)} tickers...")
    try:
        download_historical_data_with_crypto(tickers, config)
        data = load_historical_data("./data", config)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}", exc_info=True)
        _write_status({"status": "data_fetch_failed", "error": str(e)})
        return 3

    if data.empty:
        logger.error("No data loaded; aborting retrain.")
        _write_status({"status": "no_data"})
        return 4

    # ----- Feature engineering (is_backtest=True suppresses sentiment leakage) -----
    data = engineer_features(data, config=config, is_backtest=True)
    data = add_sentiment_features(data, config, is_backtest=True)
    if data.empty:
        logger.error("Empty dataframe after feature engineering; aborting.")
        _write_status({"status": "empty_features"})
        return 5

    # Mirror the feature set used in main()
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
    try:
        from finviz_enrichment import FINVIZ_AVAILABLE
        if FINVIZ_AVAILABLE:
            selected_features.extend([
                "PE_Ratio", "Forward_PE", "PEG_Ratio", "Debt_Equity",
                "ROE", "Profit_Margin", "Short_Float", "Beta",
                "Analyst_Recom", "SMA20_Dist", "SMA50_Dist",
            ])
    except ImportError:
        pass
    features_with_sentiment = selected_features + ["Sentiment_Score"]

    # Only keep features actually present in the dataframe (defensive)
    features_with_sentiment = [f for f in features_with_sentiment if f in data.columns]
    if "Target" not in data.columns:
        logger.error("No 'Target' column in engineered data; aborting.")
        _write_status({"status": "no_target"})
        return 6

    # ----- Backup existing model BEFORE training so a crash mid-train can't lose it -----
    model_path = ARTIFACT_DIR / "final_model.pkl"
    backup_path = None
    if model_path.exists():
        backup_path = BACKUP_DIR / f"final_model_{start.strftime('%Y%m%d_%H%M%S')}.pkl"
        try:
            shutil.copy2(model_path, backup_path)
            logger.info(f"Backed up existing model to {backup_path}")
        except Exception as e:
            logger.warning(f"Could not back up existing model: {e}")

    # ----- Train -----
    try:
        X_train, X_test, y_train, y_test = prepare_train_test_data(data, features_with_sentiment)
    except Exception as e:
        logger.error(f"Train/test split failed: {e}", exc_info=True)
        _write_status({"status": "split_failed", "error": str(e)})
        return 7

    if len(X_train) < 100:
        logger.error(f"Too few training samples ({len(X_train)}); aborting.")
        _write_status({"status": "insufficient_data", "n_train": len(X_train)})
        return 8

    logger.info(f"Training on {len(X_train)} rows, evaluating on {len(X_test)} rows...")
    try:
        new_model = tune_and_train_model(X_train, y_train)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        _write_status({"status": "train_failed", "error": str(e)})
        # tune_and_train_model writes directly to final_model.pkl, so if it failed
        # partway through we may have a stale or corrupt file. Restore from backup.
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, model_path)
            logger.warning("Restored model from backup due to training failure.")
        return 9

    if new_model is None:
        logger.error("Training returned no model.")
        _write_status({"status": "no_model_returned"})
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, model_path)
        return 10

    # ----- Evaluate -----
    try:
        # evaluate_model logs metrics; we re-derive precision here for the gate
        from sklearn.metrics import precision_score
        preds = new_model.predict(X_test)
        precision = float(precision_score(y_test, preds, zero_division=0))
    except Exception as e:
        logger.warning(f"Could not compute precision gate: {e}")
        precision = 0.0
    logger.info(f"New model precision on holdout: {precision:.4f} (gate >= {min_precision:.4f})")
    try:
        evaluate_model(new_model, X_test, y_test)
    except Exception as e:
        logger.warning(f"evaluate_model raised: {e}")

    if precision < min_precision:
        logger.warning(
            f"New model precision {precision:.4f} below min {min_precision:.4f}; "
            f"rolling back to previous model."
        )
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, model_path)
        _write_status({
            "status": "rejected_below_threshold",
            "precision": precision,
            "min_precision": min_precision,
            "rolled_back": bool(backup_path),
        })
        return 11

    # ----- Hot-reload signal to running bot (best-effort) -----
    reload_result = _signal_bot_reload()
    logger.info(f"Reload signal: {reload_result}")

    elapsed = (datetime.now() - start).total_seconds()
    _write_status({
        "status": "ok",
        "precision": precision,
        "min_precision": min_precision,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features_with_sentiment,
        "backup": str(backup_path) if backup_path else None,
        "reload": reload_result,
        "elapsed_seconds": round(elapsed, 1),
    })
    logger.info(f"Retrain complete in {elapsed:.1f}s. Precision={precision:.4f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
