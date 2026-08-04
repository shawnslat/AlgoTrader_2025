# Archive Directory

This directory contains development artifacts, test scripts, and deprecated code that are not needed for daily operation but kept for reference.

## Directory Structure

### 📊 analysis/
Model improvement and analysis scripts from December 2025 accuracy improvement project.

**Files:**
- `improved_model.py` - First attempt at improved model (multi-class, achieved 36% accuracy)
- `improved_model_v2.py` - Optimized binary classifier (achieved 55.4% accuracy) ✅
- `model_analysis_and_improvement.py` - Comprehensive model diagnostic tool
- `quick_model_diagnosis.py` - Fast 5-second diagnostic script
- `visualize_improvements.py` - Creates performance comparison charts

**Status:** Reference only. The improved model V2 has been trained and saved to `artifacts/improved_model_v2.pkl`.

**To use:** See `docs/MODEL_IMPROVEMENT_REPORT.md` for full details on the improvements.

---

### 🧪 test_scripts/
Utility and testing scripts used during development.

**Files:**
- `fix_sentiment_fallback.py` - Automated script that applied sentiment fallback fix (Dec 17, 2025)
- `test_sentiment_fix.py` - Unit test for sentiment fallback mechanism
- `update_tickers_from_json.py` - Helper to sync tickers from Dexter output
- `organize_files.sh` - This cleanup script

**Status:** Reference only. These were one-time utilities.

---

### 🗄️ old_gui_scripts/
Deprecated GUI launchers and simplified bot service.

**Files:**
- `trader_gui.py` - Original GUI launcher (replaced by `launch_gui_proper.py`)
- `bot_service_simple.py` - Mock bot service for GUI testing (replaced by full `bot_service.py`)
- `run_gui.sh` - Old GUI startup script (replaced by `2_start_gui.sh`)
- `test_gui.sh` - Old GUI test script (deprecated)

**Why deprecated:**
- `trader_gui.py`: Had PyQt6 segfault issues on macOS, replaced by inline version in `launch_gui_proper.py`
- `bot_service_simple.py`: Was mock version, now using full `bot_service.py` with real trading logic
- Shell scripts: Replaced by numbered scripts (`1_start_bot_service.sh`, `2_start_gui.sh`)

**Status:** Do not use. Kept for reference only.

---

## Key Achievements Archived Here

### Model Accuracy Improvement (Dec 17, 2025)
- **Baseline:** 52.0% accuracy
- **Improved:** 55.4% accuracy
- **Gain:** +3.4 percentage points
- **Method:** Enhanced features, ensemble model, noise filtering, regularization

See: `docs/MODEL_IMPROVEMENT_REPORT.md`

### Sentiment Fallback Fix (Dec 17, 2025)
- **Problem:** Sentiment analysis failure blocked all trading
- **Solution:** Added try-except fallback to continue with neutral sentiment
- **Impact:** Bot now trades even when sentiment APIs fail

See: `docs/SENTIMENT_FIX_SUMMARY.md`

---

## When to Use These Files

### Analysis Scripts
Use if you want to:
- Re-analyze model performance
- Retrain improved model with fresh data
- Generate new performance visualizations
- Diagnose model issues

**Command:**
```bash
cd archive/analysis
python quick_model_diagnosis.py  # Fast diagnostic
python model_analysis_and_improvement.py  # Full analysis
python improved_model_v2.py  # Retrain optimized model
```

### Test Scripts
Use if you want to:
- Test sentiment fallback mechanism
- Update tickers from Dexter
- Debug specific components

**Command:**
```bash
cd archive/test_scripts
python test_sentiment_fix.py
```

### Old GUI Scripts
**Do not use.** These are superseded by current implementations.

---

## Cleaning Up

If you want to permanently remove these files (not recommended):

```bash
# ⚠️ WARNING: This deletes all archived files permanently
rm -rf archive/

# Better: Keep the archive but compress it
tar -czf archive_backup_$(date +%Y%m%d).tar.gz archive/
# Then optionally delete the folder
```

---

## Related Documentation

- **Main README:** `../README.md` - Project overview and usage
- **Model Report:** `../docs/MODEL_IMPROVEMENT_REPORT.md` - Technical details on model improvements
- **Sentiment Fix:** `../docs/SENTIMENT_FIX_SUMMARY.md` - Sentiment fallback implementation
- **GUI Guide:** `../docs/GUI_README.md` - GUI usage instructions

---

**Last Updated:** December 17, 2025
**Purpose:** Historical reference and development artifacts
**Status:** Not required for production use
