# Troubleshooting Guide

## Validation Failed: 'NoneType' object has no attribute 'score'

### Problem
```
WARNING:Trader_main_Grok4_20250731:No training data available. Model training skipped.
ERROR:__main__:Error during validation: 'NoneType' object has no attribute 'score'
```

### Root Cause
The validation script needs a trained model to validate, but:
- No data in `data/` directory, OR
- Data doesn't have enough samples to split into train/test, OR
- **Split date is too early for available data** (most common!)

**Data Limitation:** Historical data only spans from **2024-01-03** to present. If you try to split at 2024-01-01 (default), there's NO training data before that date.

### Solution

#### ✅ RECOMMENDED: Use Later Split Date

The validation works best with a split date in the middle of your data range:

```bash
# Use October 2024 as split (trains on Jan-Sep, tests on Oct-Dec)
./run_validation.sh --split-date 2024-10-01 --skip-walkforward

# Or run directly:
python3 run_full_validation.py --split-date 2024-10-01 --skip-walkforward
```

This splits 2024 data:
- **Training:** Jan-Sep 2024 (9 months)
- **Testing:** Oct-Dec 2024 (3 months)

#### Option 2: Run Full Backtest First (if you haven't)

If you haven't run a backtest yet, do that first:

**From GUI (Recommended):**
```bash
# 1. Start bot service
./1_start_bot_service.sh

# 2. In another terminal, start GUI
./2_start_gui.sh

# 3. In GUI:
#    - Open "Train & Backtest" window
#    - Click "▶ Run Download + Train + Backtest"
#    - Wait for completion (5-15 minutes)
#    - Then click "🔍 Run Full Validation"
```

**From Command Line:**
```bash
# 1. Run full backtest pipeline
python3 Trader_main_Grok4_20250731.py --retrain

# This will:
#   - Download historical data → data/
#   - Engineer features
#   - Train XGBoost model → artifacts/final_model.pkl
#   - Run backtest → artifacts/backtesting_results.csv

# 2. Then run validation with correct split date
./run_validation.sh --split-date 2024-10-01 --skip-walkforward
```

---

## Common Issues

### Issue: "No data available"

**Symptoms:**
```
WARNING: No training data available. Model training skipped.
```

**Solution:**
```bash
# Check if data directory exists and has files
ls -lh data/

# If empty, download data manually
python3 Trader_main_Grok4_20250731.py --retrain
```

---

### Issue: "TA-Lib not installed"

**Symptoms:**
```
WARNING: TA-Lib is not installed. Technical indicators will not work.
```

**Impact:**
- Not critical - model will use 13 core features instead of 15
- Some advanced technical indicators won't be available

**Solution (Optional):**
```bash
# macOS
brew install ta-lib
pip3 install TA-Lib

# Linux
sudo apt-get install ta-lib
pip3 install TA-Lib

# If installation fails, just ignore - bot works without it
```

---

### Issue: GUI Validation Times Out

**Symptoms:**
```
Validation timed out after 30 minutes.
```

**Solution:**
```bash
# Run from command line instead with skip options
./run_validation.sh --skip-walkforward --skip-montecarlo

# This runs in 2-5 minutes instead of 10-30
```

---

### Issue: Validation Shows Severe Overfitting

**Symptoms:**
```
OUT-OF-SAMPLE PERFORMANCE:
  Return: -15.3%
  Degradation: 215.7%

⚠️ SEVERE OVERFITTING: >50% return drop out-of-sample!
```

**This is EXPECTED** with current 52% model accuracy.

**Solution:** Follow the improvement steps in `QUICK_START.md`:

1. **Add Confidence Filtering**
   ```python
   # Only trade when model is >70% confident
   if model_confidence >= 0.70:
       execute_trade()
   ```

2. **Remove Low-Importance Features**
   ```bash
   # Check which features matter
   cat artifacts/feature_importance.csv

   # Remove features with importance < 0.01
   # Retrain with only top 5-7 features
   ```

3. **Re-validate**
   ```bash
   ./run_validation.sh --skip-walkforward
   ```

---

### Issue: Major Dips in Backtest Chart

**Symptoms:**
- Equity curve has valleys >50% drawdown
- Backtest shows 200% returns but unstable

**Diagnosis:**
This indicates overfitting. The validation will confirm this.

**Solutions:**

**Short-term:** Reduce risk
```yaml
# In config.yaml
risk_per_trade_pct: 0.1  # Reduce from 0.2
max_position_pct: 3.0     # Reduce from 5.0
stop_loss_pct: 0.03       # Tighten from 0.05
```

**Long-term:** Fix the strategy
1. Implement confidence filtering
2. Simplify model (fewer features, less depth)
3. Add volatility filters
4. Re-validate to confirm improvement

---

### Issue: Backtest Never Completes

**Symptoms:**
- Backtest runs for >30 minutes
- No progress updates

**Solution:**
```bash
# 1. Check if it's actually running
ps aux | grep python

# 2. Check logs for errors
tail -f trader_bot.log

# 3. If stuck, restart:
pkill -f "python.*Trader"
./1_start_bot_service.sh
```

---

### Issue: Model Accuracy is Only 52%

**Symptoms:**
```
Classification Report:
  accuracy: 0.52  (barely better than coin flip)
```

**Diagnosis:**
Your model can't predict market movements. This is the CORE PROBLEM.

**Why This Happens:**
- Too many features (overfitting)
- Not enough data
- Markets are noisy (hard to predict)
- Wrong features (not predictive)

**Solutions:**

**Option A: Improve Model (Hard)**
1. Get more historical data (3+ years)
2. Add better features (order flow, volume profiles)
3. Use ensemble methods
4. Implement regime detection

**Option B: Trade Less Selectively (Easier)**
1. Only trade when model is very confident (>70%)
2. This filters out noise, keeps only high-quality signals
3. 52% overall → potentially 65-70% on high-confidence trades

**Option C: Different Approach**
1. Momentum/trend following (simpler, more robust)
2. Mean reversion on specific patterns
3. Pair trading / statistical arbitrage

**Recommended: Try Option B first** - it's the easiest way to improve without rebuilding everything.

---

## Workflow: First Time Setup

Here's the correct order to run everything:

### 1. Initial Backtest (Required)
```bash
# Start bot service
./1_start_bot_service.sh

# In GUI: Run Download + Train + Backtest
# OR from command line:
python3 Trader_main_Grok4_20250731.py --retrain
```

**This creates:**
- `data/` - Historical price data
- `artifacts/final_model.pkl` - Trained model
- `artifacts/backtesting_results.csv` - Backtest results
- `artifacts/classification_report.txt` - Model accuracy

### 2. Review Initial Results
```bash
# Check model accuracy
cat artifacts/classification_report.txt

# View backtest chart in GUI
# Look for major dips/valleys
```

### 3. Run Validation
```bash
# Quick validation (5-10 min)
./run_validation.sh --skip-walkforward

# Full validation (10-30 min)
./run_validation.sh
```

**This creates:**
- `artifacts/validation_report.txt` - Overfitting analysis
- `artifacts/out_of_sample_comparison.png` - Performance charts
- `artifacts/feature_importance.csv` - Feature rankings

### 4. Analyze Results
```bash
# Read validation report
cat artifacts/validation_report.txt

# Check for red flags:
#   - Out-of-sample return < 0%
#   - Degradation > 50%
#   - Sharpe ratio > 3.0
```

### 5. Improve Strategy
Based on validation results:
- If overfit → reduce features, add confidence filter
- If underfitting → add more data, better features
- If unstable → reduce position sizes, tighten stops

### 6. Re-validate
```bash
# After making changes
python3 Trader_main_Grok4_20250731.py --retrain
./run_validation.sh --skip-walkforward
```

### 7. Iterate
Repeat steps 5-6 until validation shows:
- ✅ Out-of-sample return > 0%
- ✅ Degradation < 20%
- ✅ Sharpe ratio 1.0-2.0
- ✅ Max drawdown < 30%

---

## Quick Reference

### Run Backtest
```bash
./1_start_bot_service.sh
# Then in GUI: "Run Download + Train + Backtest"
```

### Run Validation
```bash
# After backtest completes:
./run_validation.sh --skip-walkforward
```

### Check Results
```bash
# Model accuracy
cat artifacts/classification_report.txt

# Validation summary
cat artifacts/validation_report.txt

# Feature importance
cat artifacts/feature_importance.csv
```

### View in GUI
```bash
./2_start_gui.sh
# Open "Train & Backtest" window
# Check tabs:
#   - Classification Report
#   - Backtest Chart
#   - Validation Report
#   - Out-of-Sample Test
#   - Feature Importance
```

---

## Getting Help

**Still stuck?** Check these files:
- `QUICK_START.md` - Quick start guide
- `VALIDATION_GUIDE.md` - Detailed validation guide
- `WHATS_NEW.md` - Feature overview
- `artifacts/validation.log` - Detailed error logs
- `trader_bot.log` - Main bot logs

**Common mistakes:**
- ❌ Running validation before backtest (need model first!)
- ❌ Not waiting for backtest to complete
- ❌ Expecting validation to fix overfitting (it only detects it)
- ❌ Ignoring validation warnings (they're usually right!)

**Key insight:**
> Validation doesn't make your strategy better - it tells you the TRUTH about whether it works.
>
> If validation shows overfitting, your strategy is curve-fitted.
> If it shows good out-of-sample performance, you might have a real edge.
