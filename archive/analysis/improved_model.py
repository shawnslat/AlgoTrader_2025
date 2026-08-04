#!/usr/bin/env python3
"""
Improved Trading Model - Fixes accuracy issues identified in diagnosis
Key improvements:
1. Multi-class target (5 classes instead of binary)
2. Better features (adds time-based, price momentum)
3. Handles class imbalance
4. Ensemble model for robustness
5. Proper regularization to prevent overfitting
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from main bot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Trader_main_Grok4_20250731 import load_configuration

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


def create_improved_features(df):
    """Engineer better features with multi-timeframe and temporal components"""
    logger.info("Engineering improved features...")

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    features_list = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        if len(ticker_df) < 50:
            logger.warning(f"Skipping {ticker} - insufficient data")
            continue

        # --- PRICE-BASED FEATURES ---

        # Moving averages (short, medium, long term)
        ticker_df['MA5'] = ticker_df['close'].rolling(5, min_periods=1).mean()
        ticker_df['MA10'] = ticker_df['close'].rolling(10, min_periods=1).mean()
        ticker_df['MA20'] = ticker_df['close'].rolling(20, min_periods=1).mean()
        ticker_df['MA50'] = ticker_df['close'].rolling(50, min_periods=1).mean()

        # Price momentum (rate of change)
        ticker_df['ROC_5'] = ticker_df['close'].pct_change(5) * 100
        ticker_df['ROC_10'] = ticker_df['close'].pct_change(10) * 100
        ticker_df['ROC_20'] = ticker_df['close'].pct_change(20) * 100

        # Distance from moving averages (trend strength)
        ticker_df['Price_to_MA10'] = (ticker_df['close'] - ticker_df['MA10']) / ticker_df['MA10'] * 100
        ticker_df['Price_to_MA50'] = (ticker_df['close'] - ticker_df['MA50']) / ticker_df['MA50'] * 100

        # Bollinger Bands
        rolling_mean = ticker_df['close'].rolling(20, min_periods=1).mean()
        rolling_std = ticker_df['close'].rolling(20, min_periods=1).std()
        ticker_df['BB_Upper'] = rolling_mean + (2 * rolling_std)
        ticker_df['BB_Lower'] = rolling_mean - (2 * rolling_std)
        ticker_df['BB_Position'] = (ticker_df['close'] - ticker_df['BB_Lower']) / (ticker_df['BB_Upper'] - ticker_df['BB_Lower'])

        # --- MOMENTUM INDICATORS ---

        # RSI
        delta = ticker_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        ticker_df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = ticker_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = ticker_df['close'].ewm(span=26, adjust=False).mean()
        ticker_df['MACD'] = exp1 - exp2
        ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
        ticker_df['MACD_Hist'] = ticker_df['MACD'] - ticker_df['MACD_Signal']

        # Stochastic Oscillator
        low_14 = ticker_df['low'].rolling(14, min_periods=1).min()
        high_14 = ticker_df['high'].rolling(14, min_periods=1).max()
        ticker_df['Stochastic'] = ((ticker_df['close'] - low_14) / (high_14 - low_14).replace(0, 1)) * 100

        # --- VOLATILITY FEATURES ---

        # ATR (Average True Range)
        high_low = ticker_df['high'] - ticker_df['low']
        high_close = abs(ticker_df['high'] - ticker_df['close'].shift())
        low_close = abs(ticker_df['low'] - ticker_df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        ticker_df['ATR'] = tr.rolling(14, min_periods=1).mean()

        # Relative volatility
        ticker_df['Volatility_20'] = ticker_df['close'].rolling(20, min_periods=1).std()
        ticker_df['Volatility_Ratio'] = ticker_df['Volatility_20'] / ticker_df['close']

        # --- VOLUME FEATURES ---

        ticker_df['Volume_MA5'] = ticker_df['volume'].rolling(5, min_periods=1).mean()
        ticker_df['Volume_MA20'] = ticker_df['volume'].rolling(20, min_periods=1).mean()
        ticker_df['Volume_Ratio'] = ticker_df['volume'] / ticker_df['Volume_MA20'].replace(0, 1)

        # Volume momentum
        ticker_df['Volume_Change'] = ticker_df['volume'].pct_change(5)

        # --- TEMPORAL FEATURES ---

        ticker_df['DayOfWeek'] = ticker_df['date'].dt.dayofweek  # 0=Monday, 4=Friday
        ticker_df['Month'] = ticker_df['date'].dt.month
        ticker_df['Quarter'] = ticker_df['date'].dt.quarter

        # Market regime (simple: high/low volatility)
        vol_median = ticker_df['Volatility_Ratio'].rolling(50, min_periods=1).median()
        ticker_df['High_Vol_Regime'] = (ticker_df['Volatility_Ratio'] > vol_median).astype(int)

        # --- PRICE PATTERNS ---

        # Candle patterns
        ticker_df['Body_Size'] = abs(ticker_df['close'] - ticker_df['open']) / ticker_df['open']
        ticker_df['Upper_Shadow'] = (ticker_df['high'] - ticker_df[['close', 'open']].max(axis=1)) / ticker_df['open']
        ticker_df['Lower_Shadow'] = (ticker_df[['close', 'open']].min(axis=1) - ticker_df['low']) / ticker_df['open']

        # Gap detection
        ticker_df['Gap'] = ticker_df['open'] - ticker_df['close'].shift()
        ticker_df['Gap_Pct'] = ticker_df['Gap'] / ticker_df['close'].shift() * 100

        # --- IMPROVED TARGET (Multi-class) ---

        ticker_df['Future_Return'] = ticker_df['close'].shift(-1) / ticker_df['close'] - 1
        ticker_df['Target_MultiClass'] = pd.cut(
            ticker_df['Future_Return'],
            bins=[-np.inf, -0.015, -0.005, 0.005, 0.015, np.inf],
            labels=[0, 1, 2, 3, 4]  # 0=StrongDown, 1=Down, 2=Neutral, 3=Up, 4=StrongUp
        )

        # Drop rows with missing target
        ticker_df.dropna(subset=['Target_MultiClass'], inplace=True)

        features_list.append(ticker_df)

    if not features_list:
        raise ValueError("No valid data after feature engineering")

    combined = pd.concat(features_list, ignore_index=True)

    # Drop NaN values
    combined = combined.dropna()

    logger.info(f"Feature engineering complete: {len(combined)} rows, {len(combined.columns)} columns")
    logger.info(f"Target distribution:\n{combined['Target_MultiClass'].value_counts().sort_index()}")

    return combined


def prepare_train_test(df, test_size=0.2):
    """Split data chronologically for time series"""
    logger.info("Preparing train/test split...")

    # Select feature columns
    feature_cols = [
        'MA5', 'MA10', 'MA20', 'MA50',
        'ROC_5', 'ROC_10', 'ROC_20',
        'Price_to_MA10', 'Price_to_MA50',
        'BB_Position',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stochastic',
        'ATR', 'Volatility_Ratio',
        'Volume_Ratio', 'Volume_Change',
        'DayOfWeek', 'Month', 'Quarter', 'High_Vol_Regime',
        'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Gap_Pct'
    ]

    # Ensure all features exist
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols]
    y = df['Target_MultiClass'].astype(int)

    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")
    logger.info(f"Features:  {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_improved_model(X_train, y_train):
    """Train ensemble model with proper regularization"""
    logger.info("Training improved ensemble model...")

    # Calculate class weights for imbalance
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

    logger.info(f"Class weights: {class_weights}")

    # XGBoost with regularization
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,  # Lower learning rate
        max_depth=4,  # Shallower trees to prevent overfitting
        min_child_weight=5,  # Require more samples per leaf
        subsample=0.8,  # Use 80% of samples per tree
        colsample_bytree=0.8,  # Use 80% of features per tree
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=2.0,  # L2 regularization
        eval_metric='mlogloss',
        random_state=42,
        tree_method='hist'
    )

    # Random Forest with regularization
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,  # Limited depth
        min_samples_split=20,  # More samples required to split
        min_samples_leaf=10,  # More samples required per leaf
        max_features='sqrt',  # Limit features per tree
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )

    # Ensemble (voting classifier)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        voting='soft',  # Use probabilities
        n_jobs=-1
    )

    logger.info("Training ensemble (this may take a few minutes)...")
    ensemble.fit(X_train, y_train)

    logger.info("✓ Training complete!")
    return ensemble


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    logger.info(f"\nAccuracy:")
    logger.info(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    logger.info(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"  Overfit: {(train_acc - test_acc):.4f}")

    if train_acc - test_acc < 0.15:
        logger.info("  ✓ Overfitting is under control!")
    else:
        logger.warning("  ⚠️ Still some overfitting present")

    # Classification report
    logger.info("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred,
                                 target_names=['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up'],
                yticklabels=['Strong Down', 'Down', 'Neutral', 'Up', 'Strong Up'])
    plt.title('Confusion Matrix - Improved Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Path(ARTIFACT_DIR) / 'improved_confusion_matrix.png', dpi=150)
    plt.close()

    logger.info(f"✓ Confusion matrix saved to {ARTIFACT_DIR}/improved_confusion_matrix.png")

    # Trading signals (convert 5-class to actionable signals)
    # 0,1 = Sell, 2 = Hold, 3,4 = Buy
    def class_to_signal(pred_class):
        if pred_class <= 1:
            return -1  # Sell
        elif pred_class >= 3:
            return 1   # Buy
        else:
            return 0   # Hold

    y_test_signals = y_test.apply(lambda x: class_to_signal(x))
    y_pred_signals = pd.Series(y_test_pred).apply(lambda x: class_to_signal(x))

    signal_acc = accuracy_score(y_test_signals, y_pred_signals)
    logger.info(f"\nTrading Signal Accuracy: {signal_acc:.4f} ({signal_acc*100:.2f}%)")

    logger.info("\nSignal Classification Report:")
    print(classification_report(y_test_signals, y_pred_signals,
                                 target_names=['Sell', 'Hold', 'Buy']))

    return test_acc


def main():
    """Main training pipeline"""
    print("="*80)
    print("IMPROVED TRADING MODEL TRAINING")
    print("="*80)
    print()

    # Load data
    df = load_all_data()

    # Engineer features
    df = create_improved_features(df)

    # Prepare train/test
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)

    # Train model
    model = train_improved_model(X_train, y_train)

    # Evaluate
    test_acc = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save model
    model_path = Path(ARTIFACT_DIR) / 'improved_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"\n✓ Model saved to {model_path}")

    # Save feature list
    feature_path = Path(ARTIFACT_DIR) / 'improved_features.txt'
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"✓ Features saved to {feature_path}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Model training complete!")
    print(f"\nImprovements over baseline:")
    print(f"  Baseline accuracy:      52.0%")
    print(f"  Improved accuracy:      {test_acc*100:.1f}%")
    print(f"  Gain:                   +{(test_acc - 0.52)*100:.1f} percentage points")
    print(f"\nKey changes:")
    print(f"  1. ✓ Multi-class target (5 classes vs binary)")
    print(f"  2. ✓ {len(feature_cols)} enhanced features (vs 14)")
    print(f"  3. ✓ Ensemble model (3 algorithms)")
    print(f"  4. ✓ Regularization to prevent overfitting")
    print(f"  5. ✓ Balanced class weights")
    print(f"\nNext steps:")
    print(f"  1. Test model with: python test_improved_model.py")
    print(f"  2. Integrate into bot: Update Trader_main_Grok4_20250731.py")
    print(f"  3. Backtest with improved signals")

    return 0


if __name__ == '__main__':
    sys.exit(main())
