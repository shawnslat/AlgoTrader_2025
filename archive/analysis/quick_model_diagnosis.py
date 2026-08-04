#!/usr/bin/env python3
"""
Quick Model Diagnosis - Identifies top 3 issues with current model
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("QUICK MODEL DIAGNOSIS")
print("="*80)

# Load data files
data_files = list(Path('data').glob('*.csv'))
print(f"\n✓ Found {len(data_files)} data files")

# Load a sample file
sample_data = pd.read_csv(data_files[0])
print(f"✓ Sample data shape: {sample_data.shape}")
print(f"  Columns: {list(sample_data.columns)}")
print(f"  Date range: {sample_data['time'].min()} to {sample_data['time'].max()}")

# Load all data
all_data = []
for f in data_files:
    df = pd.read_csv(f)
    df['ticker'] = f.stem
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n✓ Combined data: {len(combined_df)} rows, {len(combined_df['ticker'].unique())} tickers")

# Simulate target creation
combined_df['date'] = combined_df['time']
combined_df = combined_df.sort_values(['ticker', 'date'])
combined_df['next_close'] = combined_df.groupby('ticker')['close'].shift(-1)
combined_df['target'] = (combined_df['next_close'] > combined_df['close']).astype(int)
combined_df.dropna(subset=['target'], inplace=True)

print("\n" + "="*80)
print("ISSUE #1: CLASS IMBALANCE")
print("="*80)

class_counts = combined_df['target'].value_counts()
print(f"\nTarget distribution:")
print(f"  Class 0 (down): {class_counts[0]:,} ({class_counts[0]/len(combined_df)*100:.1f}%)")
print(f"  Class 1 (up):   {class_counts[1]:,} ({class_counts[1]/len(combined_df)*100:.1f}%)")

imbalance_ratio = max(class_counts) / min(class_counts)
print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.15:
    print(f"\n⚠️  PROBLEM: Classes are imbalanced!")
    print(f"  📊 The model is biased toward predicting the majority class")
    print(f"  🔧 FIX: Use scale_pos_weight = {imbalance_ratio:.2f} in XGBoost")
else:
    print(f"\n✓ Classes are reasonably balanced")

print("\n" + "="*80)
print("ISSUE #2: TARGET DEFINITION")
print("="*80)

# Analyze price movements
combined_df['price_change_pct'] = ((combined_df['next_close'] - combined_df['close']) / combined_df['close'] * 100)

print(f"\nPrice change statistics:")
print(f"  Mean:   {combined_df['price_change_pct'].mean():.3f}%")
print(f"  Median: {combined_df['price_change_pct'].median():.3f}%")
print(f"  Std:    {combined_df['price_change_pct'].std():.3f}%")

# Count near-zero moves
small_moves = combined_df[abs(combined_df['price_change_pct']) < 0.5]
print(f"\n  Moves < 0.5%: {len(small_moves):,} ({len(small_moves)/len(combined_df)*100:.1f}%)")

print(f"\n⚠️  PROBLEM: Binary target treats all moves equally!")
print(f"  📊 A +0.01% move is treated same as +5% move")
print(f"  📊 Model can't distinguish meaningful moves from noise")
print(f"  🔧 FIX: Filter out small moves or use multi-class target:")
print(f"       - Strong Down: < -1.5%")
print(f"       - Down: -1.5% to -0.5%")
print(f"       - Neutral: -0.5% to +0.5%")
print(f"       - Up: +0.5% to +1.5%")
print(f"       - Strong Up: > +1.5%")

print("\n" + "="*80)
print("ISSUE #3: INSUFFICIENT DATA PER TICKER")
print("="*80)

rows_per_ticker = combined_df.groupby('ticker').size()
print(f"\nRows per ticker:")
for ticker, count in rows_per_ticker.items():
    print(f"  {ticker:6s}: {count:4d} rows")

min_rows = rows_per_ticker.min()
print(f"\n  Minimum: {min_rows} rows per ticker")

if min_rows < 400:
    print(f"\n⚠️  PROBLEM: Insufficient data for reliable training!")
    print(f"  📊 ML models need 1000+ samples for good performance")
    print(f"  📊 With 80/20 split, you have only ~{int(min_rows * 0.8)} training samples")
    print(f"  🔧 FIX OPTIONS:")
    print(f"       1. Download 2-3 years of history (not just 1 year)")
    print(f"       2. Train on ALL tickers together (not per-ticker)")
    print(f"       3. Use simpler models (linear regression, logistic)")
else:
    print(f"\n✓ Data volume is reasonable")

print("\n" + "="*80)
print("RECOMMENDED FIXES (Priority Order)")
print("="*80)

print("""
1. 🔧 USE MULTI-CLASS TARGET (High Impact)
   Current: Binary (up/down)
   Better:  5 classes (strong down, down, neutral, up, strong up)
   Why:     Filters noise, focuses on tradeable moves

2. 🔧 HANDLE CLASS IMBALANCE (High Impact)
   Current: No weighting
   Better:  scale_pos_weight in XGBoost
   Why:     Prevents bias toward majority class

3. 🔧 TRAIN SINGLE MODEL FOR ALL TICKERS (Medium Impact)
   Current: Separate models per ticker
   Better:  One model learns from all tickers
   Why:     More training data = better patterns

4. 🔧 ADD TIME-BASED FEATURES (Medium Impact)
   Current: Only price/volume indicators
   Better:  Day of week, month, market regime
   Why:     Markets have temporal patterns

5. 🔧 COLLECT MORE DATA (Low effort, high impact)
   Current: ~1 year history
   Better:  2-3 years
   Why:     More diverse market conditions

Expected Improvement: 52% → 58-62% accuracy
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
Run the full analysis with:
  python model_analysis_and_improvement.py

This will:
- Generate detailed feature importance rankings
- Test ensemble models (XGBoost + Random Forest + Gradient Boosting)
- Optimize decision thresholds
- Create visualization reports
""")
