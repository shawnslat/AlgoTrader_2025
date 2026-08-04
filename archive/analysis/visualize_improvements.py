#!/usr/bin/env python3
"""
Visualization of Model Improvements
Creates comparison charts showing baseline vs improved performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Data
metrics = {
    'Baseline Model': {
        'Test Accuracy': 52.0,
        'Train Accuracy': 88.0,
        'Overfitting': 36.0,
        'ROC AUC': 49.2,
        'Precision': 55.0,
        'Recall': 54.0
    },
    'Improved Model V2': {
        'Test Accuracy': 55.4,
        'Train Accuracy': 70.2,
        'Overfitting': 14.7,
        'ROC AUC': 54.0,
        'Precision': 58.4,
        'Recall': 73.0
    }
}

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
categories = ['Test Accuracy', 'Train Accuracy']
baseline_acc = [metrics['Baseline Model']['Test Accuracy'],
                metrics['Baseline Model']['Train Accuracy']]
improved_acc = [metrics['Improved Model V2']['Test Accuracy'],
                metrics['Improved Model V2']['Train Accuracy']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline', color='#ff6b6b', alpha=0.8)
bars2 = ax1.bar(x + width/2, improved_acc, width, label='Improved V2', color='#51cf66', alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 100)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# 2. Overfitting Reduction
ax2 = plt.subplot(2, 3, 2)
overfitting = [metrics['Baseline Model']['Overfitting'],
               metrics['Improved Model V2']['Overfitting']]
colors = ['#ff6b6b', '#51cf66']

bars = ax2.bar(['Baseline', 'Improved V2'], overfitting, color=colors, alpha=0.8)
ax2.set_ylabel('Overfitting Gap (%)', fontsize=12)
ax2.set_title('Overfitting Reduction', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 40)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

ax2.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Acceptable (<15%)')
ax2.legend(fontsize=9)

# 3. ROC AUC Comparison
ax3 = plt.subplot(2, 3, 3)
auc_values = [metrics['Baseline Model']['ROC AUC'],
              metrics['Improved Model V2']['ROC AUC']]
colors = ['#ff6b6b', '#51cf66']

bars = ax3.bar(['Baseline', 'Improved V2'], auc_values, color=colors, alpha=0.8)
ax3.set_ylabel('ROC AUC (%)', fontsize=12)
ax3.set_title('ROC AUC Score', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax3.legend(fontsize=9)

# 4. Precision & Recall
ax4 = plt.subplot(2, 3, 4)
categories = ['Precision', 'Recall']
baseline_pr = [metrics['Baseline Model']['Precision'],
               metrics['Baseline Model']['Recall']]
improved_pr = [metrics['Improved Model V2']['Precision'],
               metrics['Improved Model V2']['Recall']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, baseline_pr, width, label='Baseline', color='#ff6b6b', alpha=0.8)
bars2 = ax4.bar(x + width/2, improved_pr, width, label='Improved V2', color='#51cf66', alpha=0.8)

ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_title('Trading Metrics (Buy Signals)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend(fontsize=10)
ax4.set_ylim(0, 100)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# 5. Feature Count Comparison
ax5 = plt.subplot(2, 3, 5)
feature_counts = [14, 25]
colors = ['#ff6b6b', '#51cf66']

bars = ax5.bar(['Baseline', 'Improved V2'], feature_counts, color=colors, alpha=0.8)
ax5.set_ylabel('Number of Features', fontsize=12)
ax5.set_title('Feature Engineering', fontsize=14, fontweight='bold')
ax5.set_ylim(0, 30)

for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 6. Key Improvements Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
KEY IMPROVEMENTS

✅ Test Accuracy: 52.0% → 55.4%
   (+3.4 percentage points)

✅ Overfitting: 36.0% → 14.7%
   (-21.3 percentage points)

✅ Precision: 55.0% → 58.4%
   (+3.4 percentage points)

✅ Recall: 54.0% → 73.0%
   (+19.0 percentage points)

✅ Features: 14 → 25
   (+11 enhanced features)

METHODS USED:
• Noise filtering (>0.5% moves)
• Enhanced feature engineering
• Ensemble (XGB+RF+GB)
• Heavy regularization
• Class imbalance handling
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Trading Bot Model Improvement Analysis', fontsize=18, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = Path('artifacts/model_improvement_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to {output_path}")

# Create a second figure showing feature importance comparison
fig2, ax = plt.subplots(figsize=(12, 8))

# Feature categories
categories = ['Price\nMomentum', 'Moving\nAverages', 'Momentum\nIndicators',
              'Volatility', 'Volume', 'Temporal']
baseline_features = [0, 2, 6, 3, 1, 0]  # Approximate
improved_features = [4, 4, 6, 5, 3, 3]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_features, width, label='Baseline',
               color='#ff6b6b', alpha=0.8)
bars2 = ax.bar(x + width/2, improved_features, width, label='Improved V2',
               color='#51cf66', alpha=0.8)

ax.set_ylabel('Number of Features', fontsize=12)
ax.set_xlabel('Feature Category', fontsize=12)
ax.set_title('Feature Distribution by Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.set_ylim(0, 8)

for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
output_path2 = Path('artifacts/feature_distribution.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Feature distribution saved to {output_path2}")

print("\n" + "="*60)
print("VISUALIZATIONS CREATED SUCCESSFULLY")
print("="*60)
print("\nGenerated files:")
print(f"  1. {output_path}")
print(f"  2. {output_path2}")
print("\nThese charts show:")
print("  • Accuracy improvements")
print("  • Overfitting reduction")
print("  • Trading metrics (precision/recall)")
print("  • Feature engineering enhancements")
