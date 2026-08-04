"""
Enhanced Backtesting Validation Module

This module implements proper out-of-sample testing, walk-forward analysis,
and comprehensive risk metrics to detect overfitting and validate strategy robustness.

Key Features:
- Out-of-sample testing (train on pre-2024, test on 2024-2025)
- Walk-forward analysis with rolling windows
- Comprehensive risk metrics (Sharpe, Sortino, Calmar ratios)
- Monte Carlo simulation for robustness testing
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import os

logger = logging.getLogger(__name__)

# =============================================================================
# Risk Metrics Calculations
# =============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio.

    Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
    Annualized assuming 252 trading days.

    Good Sharpe ratio: 1.0-1.5 is acceptable, >2.0 is excellent, >3.0 may indicate overfitting.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility).

    Sortino = (Mean Return - Risk-Free Rate) / Downside Deviation
    Better measure than Sharpe for strategies with asymmetric returns.
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    return sortino


def calculate_calmar_ratio(capital_history: pd.DataFrame, years: float = None) -> float:
    """
    Calculate Calmar Ratio.

    Calmar = Annualized Return / Maximum Drawdown
    Measures return per unit of downside risk.
    Good ratio: >0.5, Excellent: >1.0
    """
    if capital_history.empty or 'capital' not in capital_history.columns:
        return 0.0

    # Calculate annualized return
    if years is None:
        years = len(capital_history) / 252

    if years <= 0:
        return 0.0

    total_return = (capital_history['capital'].iloc[-1] / capital_history['capital'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1

    # Calculate max drawdown
    running_max = capital_history['capital'].cummax()
    drawdown = (capital_history['capital'] - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    if max_drawdown == 0:
        return 0.0

    calmar = annualized_return / max_drawdown
    return calmar


def calculate_max_drawdown(capital_history: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics.

    Returns:
        Dictionary with max_drawdown, max_drawdown_pct, drawdown_duration
    """
    if capital_history.empty or 'capital' not in capital_history.columns:
        return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'drawdown_duration': 0}

    running_max = capital_history['capital'].cummax()
    drawdown = capital_history['capital'] - running_max
    drawdown_pct = drawdown / running_max

    max_dd = abs(drawdown.min())
    max_dd_pct = abs(drawdown_pct.min())

    # Calculate drawdown duration (days underwater)
    is_underwater = drawdown < 0
    drawdown_duration = is_underwater.sum()

    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'drawdown_duration': drawdown_duration
    }


def calculate_win_rate(capital_history: pd.DataFrame) -> Dict[str, float]:
    """Calculate win rate and profit factor from daily returns."""
    if capital_history.empty:
        return {'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0}

    returns = capital_history['capital'].pct_change().dropna()

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def calculate_comprehensive_metrics(capital_history: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """Calculate all performance metrics in one place."""
    if capital_history.empty:
        return {}

    returns = capital_history['capital'].pct_change().fillna(0)

    total_return = (capital_history['capital'].iloc[-1] / initial_capital) - 1
    years = len(capital_history) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(capital_history, years)
    dd_metrics = calculate_max_drawdown(capital_history)
    win_metrics = calculate_win_rate(capital_history)

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': dd_metrics['max_drawdown'],
        'max_drawdown_pct': dd_metrics['max_drawdown_pct'],
        'drawdown_duration': dd_metrics['drawdown_duration'],
        'win_rate': win_metrics['win_rate'],
        'profit_factor': win_metrics['profit_factor'],
        'avg_win': win_metrics['avg_win'],
        'avg_loss': win_metrics['avg_loss'],
        'volatility': returns.std() * np.sqrt(252),
        'num_trades': len(capital_history)
    }

    return metrics


# =============================================================================
# Out-of-Sample Testing
# =============================================================================

def split_train_test_by_date(data: pd.DataFrame, split_date: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train (before split_date) and test (after split_date).

    This is critical for detecting overfitting:
    - Train on historical data (e.g., pre-2024)
    - Test on recent unseen data (e.g., 2024-2025)
    """
    if 'date' not in data.columns:
        logger.error("Data must have 'date' column for time-based split")
        return data, pd.DataFrame()

    data['date'] = pd.to_datetime(data['date'])
    split_timestamp = pd.to_datetime(split_date)

    train_data = data[data['date'] < split_timestamp].copy()
    test_data = data[data['date'] >= split_timestamp].copy()

    logger.info(f"Train set: {len(train_data)} rows ({train_data['date'].min()} to {train_data['date'].max()})")
    logger.info(f"Test set: {len(test_data)} rows ({test_data['date'].min()} to {test_data['date'].max()})")

    return train_data, test_data


def run_out_of_sample_backtest(
    data: pd.DataFrame,
    train_model_func,
    backtest_func,
    selected_features: List[str],
    config: Dict,
    q_table,
    split_date: str = '2024-01-01',
    initial_capital: float = 10000
) -> Dict:
    """
    Run proper out-of-sample backtest.

    Process:
    1. Split data into train (pre-2024) and test (2024-2025)
    2. Train model ONLY on train data
    3. Backtest on BOTH train and test data separately
    4. Compare in-sample vs out-of-sample performance

    Returns:
        Dictionary with in_sample_metrics, out_of_sample_metrics, and degradation analysis
    """
    logger.info("=" * 80)
    logger.info("RUNNING OUT-OF-SAMPLE BACKTEST")
    logger.info("=" * 80)

    # Split data
    train_data, test_data = split_train_test_by_date(data, split_date)

    if test_data.empty:
        logger.warning("No test data available for out-of-sample testing")
        return {}

    # Prepare train/test sets for model training
    X_train = train_data[selected_features]
    y_train = train_data['Target']
    X_test = test_data[selected_features]
    y_test = test_data['Target']

    # Train model on training data only
    logger.info("Training model on in-sample data...")
    model = train_model_func(X_train, y_train)

    # Evaluate model accuracy on both sets
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    logger.info(f"Model accuracy - In-sample: {train_accuracy:.2%}, Out-of-sample: {test_accuracy:.2%}")

    # Backtest on in-sample data
    logger.info("\nBacktesting on IN-SAMPLE data (training period)...")
    in_sample_results = backtest_func(model, train_data, selected_features, config, q_table, initial_capital)
    in_sample_metrics = calculate_comprehensive_metrics(in_sample_results[0], initial_capital)

    # Backtest on out-of-sample data
    logger.info("\nBacktesting on OUT-OF-SAMPLE data (test period)...")
    out_sample_results = backtest_func(model, test_data, selected_features, config, q_table, initial_capital)
    out_sample_metrics = calculate_comprehensive_metrics(out_sample_results[0], initial_capital)

    # Calculate degradation
    degradation = {
        'return_degradation': in_sample_metrics['total_return'] - out_sample_metrics['total_return'],
        'sharpe_degradation': in_sample_metrics['sharpe_ratio'] - out_sample_metrics['sharpe_ratio'],
        'accuracy_degradation': train_accuracy - test_accuracy
    }

    # Log comparison
    logger.info("\n" + "=" * 80)
    logger.info("OUT-OF-SAMPLE VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"IN-SAMPLE (Train):")
    logger.info(f"  Total Return: {in_sample_metrics['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {in_sample_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {in_sample_metrics['max_drawdown_pct']:.2%}")
    logger.info(f"  Model Accuracy: {train_accuracy:.2%}")

    logger.info(f"\nOUT-OF-SAMPLE (Test):")
    logger.info(f"  Total Return: {out_sample_metrics['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {out_sample_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {out_sample_metrics['max_drawdown_pct']:.2%}")
    logger.info(f"  Model Accuracy: {test_accuracy:.2%}")

    logger.info(f"\nDEGRADATION:")
    logger.info(f"  Return drop: {degradation['return_degradation']:.2%}")
    logger.info(f"  Sharpe drop: {degradation['sharpe_degradation']:.2f}")
    logger.info(f"  Accuracy drop: {degradation['accuracy_degradation']:.2%}")

    # Overfitting warnings
    if degradation['return_degradation'] > 0.5:
        logger.warning("⚠️  SEVERE OVERFITTING: >50% return drop out-of-sample!")
    elif degradation['return_degradation'] > 0.2:
        logger.warning("⚠️  MODERATE OVERFITTING: >20% return drop out-of-sample")

    if out_sample_metrics['total_return'] < 0:
        logger.warning("⚠️  STRATEGY LOSES MONEY OUT-OF-SAMPLE!")

    return {
        'in_sample': in_sample_metrics,
        'out_of_sample': out_sample_metrics,
        'degradation': degradation,
        'in_sample_history': in_sample_results[0],
        'out_sample_history': out_sample_results[0]
    }


# =============================================================================
# Walk-Forward Analysis
# =============================================================================

def run_walk_forward_analysis(
    data: pd.DataFrame,
    train_model_func,
    backtest_func,
    selected_features: List[str],
    config: Dict,
    q_table,
    train_window_days: int = 365,
    test_window_days: int = 90,
    step_days: int = 30,
    initial_capital: float = 10000
) -> Dict:
    """
    Run walk-forward analysis with rolling windows.

    Process:
    1. Train on window 1, test on next period
    2. Roll forward, train on window 2, test on next period
    3. Continue rolling through entire dataset
    4. Aggregate out-of-sample results

    This is the GOLD STANDARD for validating trading strategies.

    Args:
        train_window_days: Size of training window (e.g., 365 days)
        test_window_days: Size of test window (e.g., 90 days)
        step_days: How far to roll forward each iteration (e.g., 30 days)
    """
    logger.info("=" * 80)
    logger.info("RUNNING WALK-FORWARD ANALYSIS")
    logger.info("=" * 80)

    if 'date' not in data.columns:
        logger.error("Data must have 'date' column")
        return {}

    data = data.sort_values('date').copy()
    data['date'] = pd.to_datetime(data['date'])

    min_date = data['date'].min()
    max_date = data['date'].max()

    walk_forward_results = []
    current_start = min_date

    iteration = 1

    while True:
        train_end = current_start + timedelta(days=train_window_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_days)

        if test_end > max_date:
            break

        # Extract windows
        train_window = data[(data['date'] >= current_start) & (data['date'] < train_end)]
        test_window = data[(data['date'] >= test_start) & (data['date'] < test_end)]

        if train_window.empty or test_window.empty:
            current_start += timedelta(days=step_days)
            continue

        logger.info(f"\nIteration {iteration}:")
        logger.info(f"  Train: {current_start.date()} to {train_end.date()} ({len(train_window)} rows)")
        logger.info(f"  Test:  {test_start.date()} to {test_end.date()} ({len(test_window)} rows)")

        try:
            # Train model on this window
            X_train = train_window[selected_features]
            y_train = train_window['Target']

            model = train_model_func(X_train, y_train)

            # Test on forward period
            test_results = backtest_func(model, test_window, selected_features, config, q_table, initial_capital)
            test_metrics = calculate_comprehensive_metrics(test_results[0], initial_capital)

            walk_forward_results.append({
                'iteration': iteration,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'metrics': test_metrics,
                'capital_history': test_results[0]
            })

            logger.info(f"  Test Return: {test_metrics['total_return']:.2%}, Sharpe: {test_metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")

        current_start += timedelta(days=step_days)
        iteration += 1

    if not walk_forward_results:
        logger.warning("No walk-forward results generated")
        return {}

    # Aggregate results
    all_returns = [r['metrics']['total_return'] for r in walk_forward_results]
    all_sharpes = [r['metrics']['sharpe_ratio'] for r in walk_forward_results]

    positive_periods = sum(1 for r in all_returns if r > 0)
    total_periods = len(all_returns)

    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total iterations: {total_periods}")
    logger.info(f"Profitable periods: {positive_periods} ({positive_periods/total_periods:.1%})")
    logger.info(f"Average return: {np.mean(all_returns):.2%}")
    logger.info(f"Median return: {np.median(all_returns):.2%}")
    logger.info(f"Std dev of returns: {np.std(all_returns):.2%}")
    logger.info(f"Average Sharpe: {np.mean(all_sharpes):.2f}")
    logger.info(f"Median Sharpe: {np.median(all_sharpes):.2f}")

    if positive_periods / total_periods < 0.5:
        logger.warning("⚠️  STRATEGY LOSES MONEY IN >50% OF FORWARD PERIODS!")

    return {
        'results': walk_forward_results,
        'summary': {
            'total_iterations': total_periods,
            'profitable_periods': positive_periods,
            'win_rate': positive_periods / total_periods,
            'avg_return': np.mean(all_returns),
            'median_return': np.median(all_returns),
            'std_return': np.std(all_returns),
            'avg_sharpe': np.mean(all_sharpes),
            'median_sharpe': np.median(all_sharpes)
        }
    }


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def run_monte_carlo_simulation(
    capital_history: pd.DataFrame,
    num_simulations: int = 1000,
    num_periods: int = 252
) -> Dict:
    """
    Run Monte Carlo simulation by bootstrapping historical returns.

    This tests robustness by:
    1. Sampling historical daily returns randomly
    2. Creating synthetic equity curves
    3. Analyzing distribution of outcomes

    Helps answer: "What range of outcomes is likely?"
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING MONTE CARLO SIMULATION ({num_simulations} runs)")
    logger.info("=" * 80)

    if capital_history.empty:
        return {}

    # Calculate historical returns
    returns = capital_history['capital'].pct_change().dropna()

    if len(returns) == 0:
        logger.warning("No returns to simulate")
        return {}

    initial_capital = capital_history['capital'].iloc[0]

    # Run simulations
    final_capitals = []
    max_drawdowns = []
    sharpe_ratios = []

    for i in range(num_simulations):
        # Bootstrap sample returns
        simulated_returns = np.random.choice(returns, size=num_periods, replace=True)

        # Build equity curve
        equity_curve = initial_capital * (1 + simulated_returns).cumprod()

        final_capitals.append(equity_curve.iloc[-1])

        # Calculate drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdowns.append(abs(drawdown.min()))

        # Calculate Sharpe
        if simulated_returns.std() > 0:
            sharpe = (simulated_returns.mean() / simulated_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)

    # Analyze distribution
    final_capitals = np.array(final_capitals)
    max_drawdowns = np.array(max_drawdowns)
    sharpe_ratios = np.array(sharpe_ratios)

    percentiles = [5, 25, 50, 75, 95]
    final_percentiles = np.percentile(final_capitals, percentiles)
    dd_percentiles = np.percentile(max_drawdowns, percentiles)

    prob_profit = (final_capitals > initial_capital).sum() / num_simulations
    prob_loss_50pct = (final_capitals < initial_capital * 0.5).sum() / num_simulations

    logger.info("\nMONTE CARLO RESULTS:")
    logger.info(f"Probability of profit: {prob_profit:.1%}")
    logger.info(f"Probability of >50% loss: {prob_loss_50pct:.1%}")
    logger.info(f"\nFinal Capital Percentiles:")
    for p, val in zip(percentiles, final_percentiles):
        logger.info(f"  {p}th percentile: ${val:,.2f} ({(val/initial_capital - 1):.1%})")

    logger.info(f"\nMax Drawdown Percentiles:")
    for p, val in zip(percentiles, dd_percentiles):
        logger.info(f"  {p}th percentile: {val:.1%}")

    logger.info(f"\nSharpe Ratio: Mean={np.mean(sharpe_ratios):.2f}, Median={np.median(sharpe_ratios):.2f}")

    return {
        'prob_profit': prob_profit,
        'prob_loss_50pct': prob_loss_50pct,
        'final_capital_percentiles': dict(zip(percentiles, final_percentiles)),
        'max_drawdown_percentiles': dict(zip(percentiles, dd_percentiles)),
        'mean_sharpe': np.mean(sharpe_ratios),
        'median_sharpe': np.median(sharpe_ratios)
    }


# =============================================================================
# Feature Importance Analysis
# =============================================================================

def analyze_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    top_n: int = 20,
    artifact_dir: str = 'artifacts'
) -> pd.DataFrame:
    """
    Analyze and visualize feature importance from XGBoost model.

    Helps identify:
    - Which features are actually predictive
    - Which features are noise (can be removed to reduce overfitting)
    """
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 80)

    importance_scores = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save to CSV
    importance_path = os.path.join(artifact_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"\nFeature importance saved to {importance_path}")

    # Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(artifact_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Feature importance plot saved to {plot_path}")

    # Identify low-importance features (candidates for removal)
    threshold = 0.01
    low_importance = importance_df[importance_df['importance'] < threshold]

    if len(low_importance) > 0:
        logger.info(f"\n⚠️  {len(low_importance)} features have very low importance (<{threshold}):")
        logger.info(f"    Consider removing: {', '.join(low_importance['feature'].tolist()[:10])}")

    return importance_df


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_out_of_sample_comparison(
    in_sample_history: pd.DataFrame,
    out_sample_history: pd.DataFrame,
    artifact_dir: str = 'artifacts'
):
    """Plot in-sample vs out-of-sample equity curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # In-sample
    in_sample_history.index = pd.to_datetime(in_sample_history.index)
    ax1.plot(in_sample_history.index, in_sample_history['capital'], color='blue', linewidth=2)
    ax1.set_title('In-Sample Performance (Training Period)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)

    # Out-of-sample
    out_sample_history.index = pd.to_datetime(out_sample_history.index)
    ax2.plot(out_sample_history.index, out_sample_history['capital'], color='orange', linewidth=2)
    ax2.set_title('Out-of-Sample Performance (Test Period)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Capital ($)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(artifact_dir, 'out_of_sample_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Out-of-sample comparison plot saved to {plot_path}")


def save_validation_report(
    oos_results: Dict,
    wf_results: Dict,
    mc_results: Dict,
    feature_importance: pd.DataFrame,
    artifact_dir: str = 'artifacts'
):
    """Save comprehensive validation report to text file."""
    report_path = os.path.join(artifact_dir, 'validation_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STRATEGY VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Out-of-sample section
        if oos_results:
            f.write("OUT-OF-SAMPLE TESTING\n")
            f.write("-" * 80 + "\n")
            f.write(f"In-Sample Return: {oos_results['in_sample']['total_return']:.2%}\n")
            f.write(f"Out-of-Sample Return: {oos_results['out_of_sample']['total_return']:.2%}\n")
            f.write(f"Return Degradation: {oos_results['degradation']['return_degradation']:.2%}\n")
            f.write(f"In-Sample Sharpe: {oos_results['in_sample']['sharpe_ratio']:.2f}\n")
            f.write(f"Out-of-Sample Sharpe: {oos_results['out_of_sample']['sharpe_ratio']:.2f}\n\n")

        # Walk-forward section
        if wf_results and 'summary' in wf_results:
            f.write("WALK-FORWARD ANALYSIS\n")
            f.write("-" * 80 + "\n")
            summary = wf_results['summary']
            f.write(f"Total Iterations: {summary['total_iterations']}\n")
            f.write(f"Profitable Periods: {summary['profitable_periods']} ({summary['win_rate']:.1%})\n")
            f.write(f"Average Return: {summary['avg_return']:.2%}\n")
            f.write(f"Median Return: {summary['median_return']:.2%}\n")
            f.write(f"Average Sharpe: {summary['avg_sharpe']:.2f}\n\n")

        # Monte Carlo section
        if mc_results:
            f.write("MONTE CARLO SIMULATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Probability of Profit: {mc_results['prob_profit']:.1%}\n")
            f.write(f"Probability of >50% Loss: {mc_results['prob_loss_50pct']:.1%}\n")
            f.write(f"Mean Sharpe Ratio: {mc_results['mean_sharpe']:.2f}\n\n")

        # Feature importance section
        if feature_importance is not None and not feature_importance.empty:
            f.write("TOP 10 FEATURES\n")
            f.write("-" * 80 + "\n")
            for idx, row in feature_importance.head(10).iterrows():
                f.write(f"{row['feature']}: {row['importance']:.4f}\n")

    logger.info(f"Validation report saved to {report_path}")
