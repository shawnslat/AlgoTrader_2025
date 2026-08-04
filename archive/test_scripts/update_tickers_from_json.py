#!/usr/bin/env python3
"""
Update config.yaml tickers from a Dexter-produced JSON file.

Expected JSON shape:
{
  "tickers": ["AAPL", "MSFT", "NVDA"]
}
"""
import argparse
import json
from pathlib import Path
import sys
import yaml


def load_tickers(json_path: Path):
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        sys.exit(f"Failed to read {json_path}: {exc}")
    tickers = data.get("tickers") or []
    if not isinstance(tickers, list):
        sys.exit("JSON must contain a list under key 'tickers'.")
    cleaned = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        s = t.strip().upper()
        if s:
            cleaned.append(s)
    cleaned = list(dict.fromkeys(cleaned))  # dedupe, preserve order
    if not cleaned:
        sys.exit("No valid tickers found in JSON.")
    return cleaned


def update_config(config_path: Path, tickers):
    try:
        config = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        sys.exit(f"Failed to read {config_path}: {exc}")
    if not isinstance(config, dict):
        sys.exit(f"{config_path} is not a YAML dict.")
    config["tickers"] = tickers
    try:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    except Exception as exc:
        sys.exit(f"Failed to write {config_path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Update config.yaml tickers from Dexter JSON.")
    parser.add_argument("--json", default="tickers_auto.json", help="Path to Dexter JSON (default: tickers_auto.json)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    args = parser.parse_args()

    json_path = Path(args.json)
    config_path = Path(args.config)

    if not json_path.exists():
        sys.exit(f"{json_path} not found. Run Dexter and save tickers JSON first.")
    if not config_path.exists():
        sys.exit(f"{config_path} not found.")

    tickers = load_tickers(json_path)
    update_config(config_path, tickers)
    print(f"Updated {config_path} tickers -> {tickers}")


if __name__ == "__main__":
    main()
