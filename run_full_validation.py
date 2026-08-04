#!/usr/bin/env python3
"""
Full Strategy Validation Script

This script runs comprehensive validation tests to detect overfitting:
1. Out-of-sample testing (train pre-2024, test 2024-2025)
2. Walk-forward analysis with rolling windows
3. Monte Carlo simulation
4. Feature importance analysis
5. Comprehensive risk metrics

Usage:
    python3 run_full_validation.py [--split-date YYYY-MM-DD] [--initial-capital AMOUNT]

Example:
    python3 run_full_validation.py --split-date 2024-01-01 --initial-capital 10000
"""

import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import main trader functions
from Trader_main_Grok4_20250731 import (
    load_configuration,
    download_historical_data_with_crypto,
    load_historical_data,
    engineer_features,
    add_sentiment_features,
    tune_and_train_model,
    load_q_table,
    generate_signals,
    simulate_trades,
    ARTIFACT_DIR
)

# Setup logging early (before any logger calls)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ARTIFACT_DIR, 'validation.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Check if talib is available (after logger is defined)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not installed. Using core features only.")

# Import validation module
from backtest_validation import (
    run_out_of_sample_backtest,
    run_walk_forward_analysis,
    run_monte_carlo_simulation,
    analyze_feature_importance,
    calculate_comprehensive_metrics,
    plot_out_of_sample_comparison,
    save_validation_report
)


def get_selected_features():
    """
    Get the list of selected features for the model.
    Matches the feature selection in Trader_main_Grok4_20250731.py
    """
    selected_features = [
        'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_Upper',
        'Bollinger_Lower', 'Lag1_Close', 'Lag2_Close', 'ATR', 'Stochastic_RSI', 'Volume_Change'
    ]

    # Add TA-Lib features only if available
    if TALIB_AVAILABLE:
        selected_features.extend(['Momentum', 'SMA_20'])
        logger.info("TA-Lib is available. Adding Momentum and SMA_20 features.")
    else:
        logger.info("TA-Lib not available. Using core features only.")

    return selected_features


def backtest_wrapper(model, data, selected_features, config, q_table, initial_capital):
    """
    Wrapper around the existing backtest logic to work with validation module.
    """
    # Generate signals
    data_with_signals = generate_signals(
        model,
        data,
        selected_features,
        config,
        q_table,
        threshold=0.5,
        positions_snapshot={}
    )

    if data_with_signals.empty:
        logger.warning("No signals generated for backtest")
        return pd.DataFrame(), []

    # Simulate trades
    capital_history, drawdown_info = simulate_trades(
        data_with_signals,
        initial_capital,
        config=config,
        max_position=100,
        stop_loss_pct=config.get('stop_loss_pct', 0.05),
        take_profit_pct=config.get('take_profit_pct', 0.10),
        buying_power_pct=config.get('buying_power_pct', 50),
        transaction_cost_pct=0.001,
        slippage_pct=0.0001
    )

    return capital_history, drawdown_info


def main():
    parser = argparse.ArgumentParser(description='Run full strategy validation')
    parser.add_argument('--split-date', type=str, default='2024-01-01',
                        help='Date to split train/test data (default: 2024-01-01)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtesting (default: 10000)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading historical data (use existing data)')
    parser.add_argument('--skip-walkforward', action='store_true',
                        help='Skip walk-forward analysis (it takes a long time)')
    parser.add_argument('--skip-montecarlo', action='store_true',
                        help='Skip Monte Carlo simulation')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE STRATEGY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Split date: {args.split_date}")
    logger.info(f"Initial capital: ${args.initial_capital:,.2f}")

    try:
        # Load configuration
        logger.info("\n1. Loading configuration...")
        config = load_configuration()

        # Download data (if not skipped)
        if not args.skip_download:
            logger.info("\n2. Downloading historical data...")
            tickers = config.get('tickers', [])

            # Add crypto tickers if enabled
            crypto_config = config.get('crypto', {})
            if crypto_config.get('enabled', False):
                crypto_tickers = crypto_config.get('tickers', [])
                tickers.extend(crypto_tickers)
                logger.info(f"Crypto trading enabled. Added {len(crypto_tickers)} crypto tickers.")

            download_historical_data_with_crypto(tickers, config)
        else:
            logger.info("\n2. Skipping data download (using existing data)")

        # Load and prepare data
        logger.info("\n3. Loading and engineering features...")
        data = load_historical_data("./data", config)

        if data.empty:
            logger.error("No data loaded. Exiting.")
            return

        data = engineer_features(data)
        data = add_sentiment_features(data, config)

        # Select features
        logger.info("\n4. Selecting features...")
        selected_features = get_selected_features()
        logger.info(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")

        # Load Q-table
        q_table = load_q_table()

        # =============================================================================
        # OUT-OF-SAMPLE TESTING
        # =============================================================================
        logger.info("\n5. Running out-of-sample validation...")
        oos_results = run_out_of_sample_backtest(
            data=data,
            train_model_func=tune_and_train_model,
            backtest_func=backtest_wrapper,
            selected_features=selected_features,
            config=config,
            q_table=q_table,
            split_date=args.split_date,
            initial_capital=args.initial_capital
        )

        if oos_results and 'in_sample_history' in oos_results and 'out_sample_history' in oos_results:
            plot_out_of_sample_comparison(
                oos_results['in_sample_history'],
                oos_results['out_sample_history'],
                ARTIFACT_DIR
            )

        # =============================================================================
        # WALK-FORWARD ANALYSIS
        # =============================================================================
        wf_results = {}
        if not args.skip_walkforward:
            logger.info("\n6. Running walk-forward analysis...")
            logger.info("   (This may take 10-30 minutes depending on data size)")
            wf_results = run_walk_forward_analysis(
                data=data,
                train_model_func=tune_and_train_model,
                backtest_func=backtest_wrapper,
                selected_features=selected_features,
                config=config,
                q_table=q_table,
                train_window_days=365,  # Train on 1 year
                test_window_days=90,     # Test on 3 months
                step_days=30,            # Roll forward 1 month
                initial_capital=args.initial_capital
            )
        else:
            logger.info("\n6. Skipping walk-forward analysis (use --skip-walkforward=false to enable)")

        # =============================================================================
        # MONTE CARLO SIMULATION
        # =============================================================================
        mc_results = {}
        if not args.skip_montecarlo and oos_results:
            logger.info("\n7. Running Monte Carlo simulation...")
            # Use out-of-sample results for Monte Carlo
            mc_results = run_monte_carlo_simulation(
                capital_history=oos_results['out_sample_history'],
                num_simulations=1000,
                num_periods=252  # 1 year
            )
        else:
            logger.info("\n7. Skipping Monte Carlo simulation")

        # =============================================================================
        # FEATURE IMPORTANCE ANALYSIS
        # =============================================================================
        logger.info("\n8. Analyzing feature importance...")

        # Load the trained model from artifacts
        import joblib
        model_path = os.path.join(ARTIFACT_DIR, 'final_model.pkl')

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            feature_importance = analyze_feature_importance(
                model=model,
                feature_names=selected_features,
                top_n=20,
                artifact_dir=ARTIFACT_DIR
            )
        else:
            logger.warning("No trained model found. Skipping feature importance analysis.")
            feature_importance = pd.DataFrame()

        # =============================================================================
        # SAVE COMPREHENSIVE REPORT
        # =============================================================================
        logger.info("\n9. Generating validation report...")
        save_validation_report(
            oos_results=oos_results,
            wf_results=wf_results,
            mc_results=mc_results,
            feature_importance=feature_importance,
            artifact_dir=ARTIFACT_DIR
        )

        # =============================================================================
        # SUMMARY AND RECOMMENDATIONS
        # =============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE - SUMMARY AND RECOMMENDATIONS")
        logger.info("=" * 80)

        if oos_results:
            degradation = oos_results.get('degradation', {})
            return_deg = degradation.get('return_degradation', 0)
            oos_return = oos_results.get('out_of_sample', {}).get('total_return', 0)
            oos_sharpe = oos_results.get('out_of_sample', {}).get('sharpe_ratio', 0)

            logger.info(f"\n📊 OUT-OF-SAMPLE PERFORMANCE:")
            logger.info(f"   Return: {oos_return:.2%}")
            logger.info(f"   Sharpe Ratio: {oos_sharpe:.2f}")
            logger.info(f"   Degradation from in-sample: {return_deg:.2%}")

            # Red flags
            red_flags = []

            if return_deg > 0.5:
                red_flags.append("🚩 SEVERE overfitting: >50% performance drop out-of-sample")
            elif return_deg > 0.2:
                red_flags.append("⚠️  MODERATE overfitting: >20% performance drop")

            if oos_return < 0:
                red_flags.append("🚩 Strategy LOSES money out-of-sample")

            if oos_sharpe > 3.0:
                red_flags.append("⚠️  Suspiciously high Sharpe ratio (>3.0) - possible overfitting")

            if oos_sharpe < 0.5:
                red_flags.append("⚠️  Poor risk-adjusted returns (Sharpe < 0.5)")

            if red_flags:
                logger.warning("\n⚠️  RED FLAGS DETECTED:")
                for flag in red_flags:
                    logger.warning(f"   {flag}")
            else:
                logger.info("\n✅ No major red flags detected")

        if wf_results and 'summary' in wf_results:
            wf_win_rate = wf_results['summary'].get('win_rate', 0)
            wf_avg_return = wf_results['summary'].get('avg_return', 0)

            logger.info(f"\n📈 WALK-FORWARD ANALYSIS:")
            logger.info(f"   Win rate: {wf_win_rate:.1%}")
            logger.info(f"   Average return per period: {wf_avg_return:.2%}")

            if wf_win_rate < 0.5:
                logger.warning("   ⚠️  Strategy loses in >50% of forward periods")

        if mc_results:
            prob_profit = mc_results.get('prob_profit', 0)
            logger.info(f"\n🎲 MONTE CARLO SIMULATION:")
            logger.info(f"   Probability of profit: {prob_profit:.1%}")

            if prob_profit < 0.6:
                logger.warning("   ⚠️  Low probability of profit (<60%)")

        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS:")
        logger.info("=" * 80)

        if oos_results and oos_results.get('degradation', {}).get('return_degradation', 0) > 0.3:
            logger.info("1. 🔧 REDUCE OVERFITTING:")
            logger.info("   - Remove low-importance features (check feature_importance.csv)")
            logger.info("   - Simplify model (reduce max_depth, increase reg_lambda)")
            logger.info("   - Use fewer technical indicators")
            logger.info("   - Increase minimum sample requirements")

        if oos_results and oos_results.get('out_of_sample', {}).get('sharpe_ratio', 0) < 1.0:
            logger.info("2. 💡 IMPROVE RISK-ADJUSTED RETURNS:")
            logger.info("   - Tighten stop-loss parameters")
            logger.info("   - Reduce position sizes")
            logger.info("   - Add volatility filters (skip trades when VIX is high)")
            logger.info("   - Consider regime filtering (only trade in bull markets)")

        if wf_results and wf_results.get('summary', {}).get('win_rate', 0) < 0.5:
            logger.info("3. 📉 LOW CONSISTENCY:")
            logger.info("   - Strategy performs inconsistently across time periods")
            logger.info("   - Consider fundamental changes to trading logic")
            logger.info("   - May need different approach (mean reversion vs trend following)")

        logger.info("\n📁 All results saved to 'artifacts/' directory:")
        logger.info("   - validation_report.txt (comprehensive summary)")
        logger.info("   - feature_importance.csv (feature analysis)")
        logger.info("   - out_of_sample_comparison.png (equity curves)")
        logger.info("   - validation.log (detailed logs)")

        logger.info("\n✅ Validation complete!")

    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
