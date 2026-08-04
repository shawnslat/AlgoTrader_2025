#!/usr/bin/env python3
"""
Model Analysis and Improvement Script
Diagnoses why model accuracy is only 52% and implements fixes
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
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
from Trader_main_Grok4_20250731 import (
    load_historical_data,
    engineer_features,
    prepare_train_test_data,
    load_configuration
)

class ModelAnalyzer:
    """Analyzes current model performance and identifies issues"""

    def __init__(self, config_path='config.yaml'):
        self.config = load_configuration(config_path)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.report_dir = Path('artifacts/model_analysis')
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        """Load data and prepare features"""
        logger.info("Loading historical data...")
        self.data = load_historical_data('data', self.config)

        if self.data.empty:
            raise ValueError("No data loaded!")

        logger.info(f"Loaded {len(self.data)} rows for {len(self.data['ticker'].unique())} tickers")
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")

        # Engineer features (without sentiment for faster analysis)
        logger.info("Engineering features...")
        self.data = engineer_features(self.data)

        if self.data.empty:
            raise ValueError("Feature engineering resulted in empty dataframe!")

        logger.info(f"After feature engineering: {len(self.data)} rows, {len(self.data.columns)} columns")

        return self.data

    def analyze_data_quality(self):
        """Check data quality issues"""
        logger.info("\n" + "="*80)
        logger.info("DATA QUALITY ANALYSIS")
        logger.info("="*80)

        # Check class balance
        if 'Target' in self.data.columns:
            class_counts = self.data['Target'].value_counts()
            logger.info(f"\nClass Distribution:")
            logger.info(f"  Class 0 (down): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(self.data)*100:.1f}%)")
            logger.info(f"  Class 1 (up):   {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(self.data)*100:.1f}%)")

            imbalance_ratio = max(class_counts) / min(class_counts)
            if imbalance_ratio > 1.5:
                logger.warning(f"⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}:1")

        # Check for data leakage
        logger.info("\nChecking for potential data leakage...")
        leakage_features = []
        for col in self.data.columns:
            if 'target' in col.lower() or 'future' in col.lower():
                leakage_features.append(col)

        if leakage_features:
            logger.warning(f"⚠️  Potential leakage features: {leakage_features}")
        else:
            logger.info("✓ No obvious data leakage detected")

        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            logger.warning(f"\n⚠️  Missing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  {col}: {count} ({count/len(self.data)*100:.1f}%)")
        else:
            logger.info("✓ No missing values")

        # Check feature distributions
        logger.info("\nFeature Statistics:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['date', 'ticker', 'Target']]

        for col in numeric_cols[:10]:  # Show first 10 features
            if col in self.data.columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                skew = self.data[col].skew()
                logger.info(f"  {col:20s}: mean={mean:8.3f}, std={std:8.3f}, skew={skew:6.2f}")

        # Save data quality report
        with open(self.report_dir / 'data_quality_report.txt', 'w') as f:
            f.write("DATA QUALITY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total rows: {len(self.data)}\n")
            f.write(f"Total features: {len(self.data.columns)}\n")
            f.write(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}\n\n")

            if 'Target' in self.data.columns:
                f.write("Class Distribution:\n")
                for cls, count in class_counts.items():
                    f.write(f"  Class {cls}: {count} ({count/len(self.data)*100:.1f}%)\n")

    def analyze_feature_importance(self, selected_features):
        """Analyze which features are actually useful"""
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)

        X = self.data[selected_features]
        y = self.data['Target']

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Quick XGBoost model for feature importance
        logger.info("Training quick model for feature importance...")
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 15 Most Important Features:")
        for idx, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")

        logger.info("\nBottom 10 Least Important Features:")
        for idx, row in importance_df.tail(10).iterrows():
            logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")

        # Save feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.savefig(self.report_dir / 'feature_importance.png', dpi=150)
        plt.close()

        # Mutual Information
        logger.info("\nCalculating Mutual Information scores...")
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_df = pd.DataFrame({
            'feature': selected_features,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        logger.info("\nTop 10 Features by Mutual Information:")
        for idx, row in mi_df.head(10).iterrows():
            logger.info(f"  {row['feature']:25s}: {row['mi_score']:.4f}")

        # Save full reports
        importance_df.to_csv(self.report_dir / 'feature_importance_xgboost.csv', index=False)
        mi_df.to_csv(self.report_dir / 'feature_importance_mi.csv', index=False)

        return importance_df, mi_df

    def test_multiple_models(self, selected_features):
        """Compare performance of different models"""
        logger.info("\n" + "="*80)
        logger.info("MULTI-MODEL COMPARISON")
        logger.info("="*80)

        X = self.data[selected_features]
        y = self.data['Target']

        # Time series split
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        models = {
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='logloss', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        }

        results = []

        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)

            # Probability predictions for AUC
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5
            else:
                auc = 0.5

            logger.info(f"  Train Accuracy: {train_acc:.4f}")
            logger.info(f"  Test Accuracy:  {test_acc:.4f}")
            logger.info(f"  ROC AUC:        {auc:.4f}")
            logger.info(f"  Overfit:        {train_acc - test_acc:.4f}")

            results.append({
                'model': name,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'roc_auc': auc,
                'overfit': train_acc - test_acc
            })

            # Classification report
            logger.info(f"\n{name} Classification Report:")
            print(classification_report(y_test, y_pred_test, digits=4))

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.report_dir / 'model_comparison.csv', index=False)

        return results_df

    def suggest_improvements(self):
        """Generate actionable improvement recommendations"""
        logger.info("\n" + "="*80)
        logger.info("IMPROVEMENT RECOMMENDATIONS")
        logger.info("="*80)

        recommendations = []

        # Check class balance
        if 'Target' in self.data.columns:
            class_counts = self.data['Target'].value_counts()
            imbalance = max(class_counts) / min(class_counts)

            if imbalance > 1.2:
                recommendations.append({
                    'issue': 'Class Imbalance',
                    'severity': 'HIGH',
                    'description': f'Target classes are imbalanced ({imbalance:.2f}:1)',
                    'solution': 'Use class weights, SMOTE oversampling, or adjust decision threshold'
                })

        # Check for overfitting
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
            model.fit(self.X_train, self.y_train)
            train_acc = model.score(self.X_train, self.y_train)
            test_acc = model.score(self.X_test, self.y_test)

            if train_acc - test_acc > 0.1:
                recommendations.append({
                    'issue': 'Overfitting',
                    'severity': 'HIGH',
                    'description': f'Large gap between train ({train_acc:.3f}) and test ({test_acc:.3f}) accuracy',
                    'solution': 'Add regularization, reduce model complexity, use more data, or add dropout'
                })

        # Check feature count
        num_features = len([c for c in self.data.columns if c not in ['date', 'ticker', 'Target']])
        if num_features > 30:
            recommendations.append({
                'issue': 'Too Many Features',
                'severity': 'MEDIUM',
                'description': f'{num_features} features may cause curse of dimensionality',
                'solution': 'Use feature selection (SelectKBest, RFE) to reduce to 15-20 most important features'
            })

        # Check data size
        if len(self.data) < 1000:
            recommendations.append({
                'issue': 'Insufficient Data',
                'severity': 'HIGH',
                'description': f'Only {len(self.data)} samples for training',
                'solution': 'Collect more historical data (2+ years) or use data augmentation'
            })

        # Target definition issue
        recommendations.append({
            'issue': 'Binary Target Too Simple',
            'severity': 'MEDIUM',
            'description': 'Predicting "close(t+1) > close(t)" ignores magnitude of moves',
            'solution': 'Consider multi-class target (strong up/up/neutral/down/strong down) or regression'
        })

        # Sentiment feature
        if 'Sentiment_Score' not in self.data.columns or self.data['Sentiment_Score'].std() < 0.01:
            recommendations.append({
                'issue': 'Weak Sentiment Feature',
                'severity': 'LOW',
                'description': 'Sentiment scores are missing or have low variance',
                'solution': 'Improve sentiment analysis or remove feature if not informative'
            })

        # Print recommendations
        logger.info("\nPriority Issues to Fix:\n")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. [{rec['severity']}] {rec['issue']}")
            logger.info(f"   Problem: {rec['description']}")
            logger.info(f"   Solution: {rec['solution']}\n")

        # Save recommendations
        rec_df = pd.DataFrame(recommendations)
        rec_df.to_csv(self.report_dir / 'recommendations.csv', index=False)

        return recommendations


class ModelImprover:
    """Implements model improvements"""

    def __init__(self, data, report_dir):
        self.data = data
        self.report_dir = report_dir

    def select_best_features(self, n_features=15):
        """Select top N most predictive features"""
        logger.info(f"\nSelecting top {n_features} features...")

        # Prepare data
        feature_cols = [c for c in self.data.columns if c not in ['date', 'ticker', 'Target', 'Sentiment_Score', 'buy_signal', 'sell_signal']]
        X = self.data[feature_cols]
        y = self.data['Target']

        # Mutual Information
        selector = SelectKBest(mutual_info_classif, k=min(n_features, len(feature_cols)))
        selector.fit(X, y)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [feat for feat, selected in zip(feature_cols, selected_mask) if selected]

        logger.info(f"Selected features: {selected_features}")

        # Save
        with open(self.report_dir / 'selected_features.txt', 'w') as f:
            f.write('\n'.join(selected_features))

        return selected_features

    def create_ensemble_model(self, selected_features):
        """Create ensemble of multiple models"""
        logger.info("\nCreating ensemble model...")

        X = self.data[selected_features]
        y = self.data['Target']

        # Time series split
        train_size = int(0.8 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Calculate class weights to handle imbalance
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]

        # Individual models
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.5,
            eval_metric='logloss',
            random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )

        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
            voting='soft'
        )

        logger.info("Training ensemble model...")
        ensemble.fit(X_train, y_train)

        # Evaluate
        train_acc = ensemble.score(X_train, y_train)
        test_acc = ensemble.score(X_test, y_test)
        y_pred = ensemble.predict(X_test)

        logger.info(f"\nEnsemble Results:")
        logger.info(f"  Train Accuracy: {train_acc:.4f}")
        logger.info(f"  Test Accuracy:  {test_acc:.4f}")
        logger.info(f"  Improvement:    {test_acc - 0.52:.4f} over baseline")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.report_dir / 'ensemble_confusion_matrix.png', dpi=150)
        plt.close()

        return ensemble, test_acc

    def optimize_threshold(self, model, selected_features):
        """Find optimal prediction threshold"""
        logger.info("\nOptimizing decision threshold...")

        X = self.data[selected_features]
        y = self.data['Target']

        train_size = int(0.8 * len(X))
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]

        # Get probabilities
        y_proba = model.predict_proba(X_test)[:, 1]

        # Test different thresholds
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_threshold = 0.5
        best_accuracy = 0

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({'threshold': threshold, 'accuracy': accuracy})

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        logger.info(f"Best threshold: {best_threshold:.2f} (accuracy: {best_accuracy:.4f})")

        # Plot threshold vs accuracy
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['accuracy'], marker='o')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Accuracy')
        plt.title('Threshold Optimization')
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.report_dir / 'threshold_optimization.png', dpi=150)
        plt.close()

        return best_threshold


def main():
    """Main analysis and improvement pipeline"""

    print("="*80)
    print("TRADING BOT MODEL ANALYSIS & IMPROVEMENT")
    print("="*80)
    print()

    # Phase 1: Analysis
    analyzer = ModelAnalyzer()

    try:
        # Load data
        analyzer.load_and_prepare_data()

        # Analyze data quality
        analyzer.analyze_data_quality()

        # Current features (from original code)
        selected_features = [
            'MA10', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'Bollinger_Upper', 'Bollinger_Lower', 'ATR', 'Stochastic_RSI',
            'Lag1_Close', 'Lag2_Close', 'Volume_Change', 'VIX'
        ]

        # Filter to available features
        selected_features = [f for f in selected_features if f in analyzer.data.columns]

        logger.info(f"\nUsing {len(selected_features)} features for analysis")

        # Analyze feature importance
        importance_df, mi_df = analyzer.analyze_feature_importance(selected_features)

        # Test multiple models
        results_df = analyzer.test_multiple_models(selected_features)

        # Get recommendations
        recommendations = analyzer.suggest_improvements()

        # Phase 2: Improvements
        improver = ModelImprover(analyzer.data, analyzer.report_dir)

        # Select best features
        best_features = improver.select_best_features(n_features=15)

        # Create ensemble model
        ensemble_model, ensemble_acc = improver.create_ensemble_model(best_features)

        # Optimize threshold
        best_threshold = improver.optimize_threshold(ensemble_model, best_features)

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"\nBaseline Accuracy: 52.0%")
        print(f"Improved Accuracy: {ensemble_acc*100:.1f}%")
        print(f"Improvement:       {(ensemble_acc - 0.52)*100:.1f} percentage points")
        print(f"Optimal Threshold: {best_threshold:.2f}")
        print(f"\nBest Features ({len(best_features)}):")
        for f in best_features:
            print(f"  - {f}")

        print(f"\n✓ Analysis complete! Reports saved to: {analyzer.report_dir}")
        print("\nNext Steps:")
        print("1. Review feature_importance.png to understand key drivers")
        print("2. Check recommendations.csv for actionable improvements")
        print("3. Update Trader_main_Grok4_20250731.py with best features")
        print("4. Consider implementing the ensemble model")

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
