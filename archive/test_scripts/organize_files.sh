#!/bin/bash
# Organize Trader_2025 directory

echo "🗂️  Organizing Trader_2025 directory..."

# Create archive structure
mkdir -p archive/analysis
mkdir -p archive/documentation
mkdir -p archive/test_scripts
mkdir -p archive/old_gui_scripts

echo "📦 Moving analysis files..."
mv MODEL_IMPROVEMENT_REPORT.md archive/documentation/
mv NO_TRADES_ANALYSIS.md archive/documentation/
mv QUICK_START_IMPROVED_MODEL.md archive/documentation/
mv SENTIMENT_FIX_SUMMARY.md archive/documentation/

echo "📦 Moving model improvement scripts..."
mv improved_model.py archive/analysis/
mv improved_model_v2.py archive/analysis/
mv model_analysis_and_improvement.py archive/analysis/
mv quick_model_diagnosis.py archive/analysis/
mv visualize_improvements.py archive/analysis/

echo "📦 Moving test scripts..."
mv test_sentiment_fix.py archive/test_scripts/
mv fix_sentiment_fallback.py archive/test_scripts/

echo "📦 Moving old GUI scripts..."
mv run_gui.sh archive/old_gui_scripts/
mv test_gui.sh archive/old_gui_scripts/
mv trader_gui.py archive/old_gui_scripts/
mv bot_service_simple.py archive/old_gui_scripts/

echo "📦 Moving helper scripts..."
mv update_tickers_from_json.py archive/test_scripts/

echo "✅ Organization complete!"
echo ""
echo "Directory structure:"
echo "  Root: Core bot files only"
echo "  archive/documentation: Reports and summaries"
echo "  archive/analysis: Model improvement scripts"
echo "  archive/test_scripts: Test and utility scripts"
echo "  archive/old_gui_scripts: Deprecated GUI launchers"
