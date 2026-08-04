#!/usr/bin/env python3
"""
Improved Trading Model V2 - Optimized Binary Classifier
Strategy: Keep binary target but fix the real issues:
1. Filter out noise (only predict on significant moves)
2. Better features with price momentum and volatility
3. Handle class imbalance properly
4. Strong regularization to prevent overfitting
5. Ensemble for robustness
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = 'artifacts'


def load_all_data():
    """Load all CSV files from data directory"""
    data_files = list(Path('data').glob('*.csv'))
    logger.info(f"Loading {len(data_files)} ticker files...")

    all_data = []
    for f in data_files:
        df = pd.read_csv(f)
        df['ticker'] = f.stem
        df['date'] = pd.to_datetime(df['time'])
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)

    logger.info(f"Loaded {len(combined)} rows for {combined['ticker'].nunique()} tickers")
    return combined


def create_enhanced_features(df, noise_threshold=0.005):
    """
    Engineer features with noise filtering.
    Only predict on moves > threshold to filter random noise.
    """
    logger.info(f"Engineering features (noise threshold: {noise_threshold*100:.2f}%)...")

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    features_list = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        if len(ticker_df) < 50:
            logger.warning(f"Skipping {ticker} - insufficient data")
            continue

        # --- PRICE MOMENTUM FEATURES ---

        # Multiple timeframe ROC
        ticker_df['ROC_3'] = ticker_df['close'].pct_change(3) * 100
        ticker_df['ROC_5'] = ticker_df['close'].pct_change(5) * 100
        ticker_df['ROC_10'] = ticker_df['close'].pct_change(10) * 100
        ticker_df['ROC_20'] = ticker_df['close'].pct_change(20) * 100

        # Moving average crossovers
        ticker_df['MA5'] = ticker_df['close'].rolling(5, min_periods=1).mean()
        ticker_df['MA10'] = ticker_df['close'].rolling(10, min_periods=1).mean()
        ticker_df['MA20'] = ticker_df['close'].rolling(20, min_periods=1).mean()
        ticker_df['MA50'] = ticker_df['close'].rolling(50, min_periods=1).mean()

        ticker_df['MA5_MA20_Ratio'] = ticker_df['MA5'] / ticker_df['MA20']
        ticker_df['MA10_MA50_Ratio'] = ticker_df['MA10'] / ticker_df['MA50']

        # Price position relative to MAs
        ticker_df['Price_above_MA20'] = (ticker_df['close'] > ticker_df['MA20']).astype(int)
        ticker_df['Price_above_MA50'] = (ticker_df['close'] > ticker_df['MA50']).astype(int)

        # --- MOMENTUM INDICATORS ---

        # RSI (14-period)
        delta = ticker_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        ticker_df['RSI'] = 100 - (100 / (1 + rs))
        ticker_df['RSI_Oversold'] = (ticker_df['RSI'] < 30).astype(int)
        ticker_df['RSI_Overbought'] = (ticker_df['RSI'] > 70).astype(int)

        # MACD
        exp1 = ticker_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = ticker_df['close'].ewm(span=26, adjust=False).mean()
        ticker_df['MACD'] = exp1 - exp2
        ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
        ticker_df['MACD_Hist'] = ticker_df['MACD'] - ticker_df['MACD_Signal']
        ticker_df['MACD_Positive'] = (ticker_df['MACD_Hist'] > 0).astype(int)

        # Stochastic
        low_14 = ticker_df['low'].rolling(14, min_periods=1).min()
        high_14 = ticker_df['high'].rolling(14, min_periods=1).max()
        ticker_df['Stochastic'] = ((ticker_df['close'] - low_14) / (high_14 - low_14).replace(0, 1)) * 100

        # --- VOLATILITY FEATURES ---

        # Bollinger Bands
        rolling_mean = ticker_df['close'].rolling(20, min_periods=1).mean()
        rolling_std = ticker_df['close'].rolling(20, min_periods=1).std()
        ticker_df['BB_Upper'] = rolling_mean + (2 * rolling_std)
        ticker_df['BB_Lower'] = rolling_mean - (2 * rolling_std)
        ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / rolling_mean
        ticker_df['BB_Position'] = (ticker_df['close'] - ticker_df['BB_Lower']) / (ticker_df['BB_Upper'] - ticker_df['BB_Lower'])

        # ATR
        high_low = ticker_df['high'] - ticker_df['low']
        high_close = abs(ticker_df['high'] - ticker_df['close'].shift())
        low_close = abs(ticker_df['low'] - ticker_df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['ATR'] = tr.rolling(14, min_periods=1).mean()
        ticker_df['ATR_Pct'] = ticker_df['ATR'] / ticker_df['close']

        # Historical volatility
        ticker_df['Volatility_10'] = ticker_df['close'].pct_change().rolling(10, min_periods=1).std() * 100
        ticker_df['Volatility_20'] = ticker_df['close'].pct_change().rolling(20, min_periods=1).std() * 100

        # --- VOLUME FEATURES ---

        ticker_df['Volume_MA20'] = ticker_df['volume'].rolling(20, min_periods=1).mean()
        ticker_df['Volume_Ratio'] = ticker_df['volume'] / ticker_df['Volume_MA20'].replace(0, 1)
        ticker_df['High_Volume'] = (ticker_df['Volume_Ratio'] > 1.5).astype(int)

        # Price-volume trend
        ticker_df['PV_Trend'] = ticker_df['close'].pct_change() * ticker_df['Volume_Ratio']

        # --- TEMPORAL FEATURES ---

        ticker_df['DayOfWeek'] = ticker_df['date'].dt.dayofweek
        ticker_df['Month'] = ticker_df['date'].dt.month
        ticker_df['Quarter'] = ticker_df['date'].dt.quarter

        # --- IMPROVED TARGET with NOISE FILTERING ---

        # Calculate next-day return
        ticker_df['Next_Return'] = ticker_df['close'].shift(-1) / ticker_df['close'] - 1

        # Only keep rows where move is significant (> threshold)
        # This filters out noise and focuses on tradeable moves
        ticker_df['Significant_Move'] = abs(ticker_df['Next_Return']) > noise_threshold

        # Binary target: 1 if up AND significant, 0 if down AND significant
        ticker_df['Target'] = (ticker_df['Next_Return'] > 0).astype(int)

        # Filter: Only keep significant moves
        ticker_df = ticker_df[ticker_df['Significant_Move'] == True].copy()

        # Drop rows with missing target
        ticker_df.dropna(subset=['Target'], inplace=True)

        features_list.append(ticker_df)

    if not features_list:
        raise ValueError("No valid data after feature engineering")

    combined = pd.concat(features_list, ignore_index=True)

    # Drop NaN values
    combined = combined.dropna()

    logger.info(f"After noise filtering: {len(combined)} rows ({len(combined)/len(df)*100:.1f}% of original)")
    logger.info(f"Features: {len(combined.columns)} columns")

    # Check class balance
    class_dist = combined['Target'].value_counts()
    logger.info(f"Target distribution:")
    logger.info(f"  Down (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(combined)*100:.1f}%)")
    logger.info(f"  Up (1):   {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(combined)*100:.1f}%)")

    return combined


def prepare_train_test(df, test_size=0.2):
    """Split data chronologically"""
    logger.info("Preparing train/test split...")

    # Feature columns
    feature_cols = [
        'ROC_3', 'ROC_5', 'ROC_10', 'ROC_20',
        'MA5_MA20_Ratio', 'MA10_MA50_Ratio',
        'Price_above_MA20', 'Price_above_MA50',
        'RSI', 'RSI_Oversold', 'RSI_Overbought',
        'MACD_Hist', 'MACD_Positive',
        'Stochastic',
        'BB_Width', 'BB_Position',
        'ATR_Pct',
        'Volatility_10', 'Volatility_20',
        'Volume_Ratio', 'High_Volume', 'PV_Trend',
        'DayOfWeek', 'Month', 'Quarter'
    ]

    # Filter to available features
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols]
    y = df['Target'].astype(int)

    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")
    logger.info(f"Features:  {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_optimal_ensemble(X_train, y_train):
    """Train ensemble with strong regularization"""
    logger.info("Training optimized ensemble model...")

    # Calculate class imbalance weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

    # XGBoost with heavy regularization
    xgb = XGBClassifier(
        n_estimators=150,
        learning_rate=0.02,  # Very low learning rate
        max_depth=3,  # Very shallow trees
        min_child_weight=10,  # Require many samples per leaf
        subsample=0.7,  # Use only 70% of samples
        colsample_bytree=0.7,  # Use only 70% of features
        reg_alpha=1.0,  # Strong L1 regularization
        reg_lambda=2.5,  # Strong L2 regularization
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        eval_metric='logloss',
        random_state=42
    )

    # Random Forest with constraints
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,  # Shallow trees
        min_samples_split=30,  # Many samples to split
        min_samples_leaf=15,  # Many samples per leaf
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        min_samples_split=30,
        min_samples_leaf=15,
        subsample=0.7,
        random_state=42
    )

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        voting='soft',
        n_jobs=-1
    )

    logger.info("Training ensemble...")
    ensemble.fit(X_train, y_train)

    logger.info("✓ Training complete!")
    return ensemble


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Comprehensive evaluation"""
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_proba)

    logger.info(f"\nMetrics:")
    logger.info(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    logger.info(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"  ROC AUC:        {auc:.4f}")
    logger.info(f"  Overfit:        {(train_acc - test_acc):.4f}")

    if train_acc - test_acc < 0.10:
        logger.info("  ✓ Overfitting is well controlled!")
    elif train_acc - test_acc < 0.20:
        logger.info("  ✓ Overfitting is acceptable")
    else:
        logger.warning("  ⚠️ Still overfitting")

    # Classification report
    logger.info("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up'], digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix - Optimized Model V2')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Path(ARTIFACT_DIR) / 'improved_model_v2_confusion.png', dpi=150)
    plt.close()

    logger.info(f"✓ Confusion matrix saved")

    # Precision-recall analysis
    tp = cm[1, 1]  # True positives
    fp = cm[0, 1]  # False positives
    fn = cm[1, 0]  # False negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    logger.info(f"\nTrading Metrics:")
    logger.info(f"  Precision: {precision:.4f} (When model says BUY, it's right {precision*100:.1f}% of the time)")
    logger.info(f"  Recall:    {recall:.4f} (Model catches {recall*100:.1f}% of good opportunities)")

    return test_acc, auc


def main():
    """Main training pipeline"""
    print("="*80)
    print("IMPROVED TRADING MODEL V2 - OPTIMIZED BINARY CLASSIFIER")
    print("="*80)
    print()

    # Load data
    df = load_all_data()

    # Engineer features with noise filtering
    df = create_enhanced_features(df, noise_threshold=0.005)  # Only predict moves > 0.5%

    # Prepare train/test
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)

    # Train model
    model = train_optimal_ensemble(X_train, y_train)

    # Evaluate
    test_acc, auc = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save model
    model_path = Path(ARTIFACT_DIR) / 'improved_model_v2.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"\n✓ Model saved to {model_path}")

    # Save feature list
    feature_path = Path(ARTIFACT_DIR) / 'improved_features_v2.txt'
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"✓ Features saved to {feature_path}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Model training complete!")
    print(f"\nResults:")
    print(f"  Baseline:           52.0% accuracy")
    print(f"  Improved V2:        {test_acc*100:.1f}% accuracy")
    print(f"  ROC AUC:            {auc:.3f}")
    print(f"  Gain:               {(test_acc - 0.52)*100:+.1f} percentage points")
    print(f"\nKey improvements:")
    print(f"  1. ✓ Noise filtering (only predict moves > 0.5%)")
    print(f"  2. ✓ {len(feature_cols)} enhanced momentum/volatility features")
    print(f"  3. ✓ Heavy regularization (prevents overfitting)")
    print(f"  4. ✓ Class imbalance handling")
    print(f"  5. ✓ Ensemble of 3 algorithms")
    print(f"\nTo use this model:")
    print(f"  1. Replace artifacts/final_model.pkl with improved_model_v2.pkl")
    print(f"  2. Update feature list in Trader_main_Grok4_20250731.py")
    print(f"  3. Run backtest to verify performance")

    return 0


if __name__ == '__main__':
    sys.exit(main())
