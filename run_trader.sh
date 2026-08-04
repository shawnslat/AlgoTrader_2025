#!/bin/bash
# Wrapper script to run the trader with automatic model retraining
# This will download data for all tickers (including new crypto), train, backtest, then trade

python3 Trader_main_Grok4_20250731.py --retrain
