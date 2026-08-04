#!/bin/bash
# Wrapper script to run comprehensive strategy validation
#
# This runs all validation tests including:
# - Out-of-sample testing
# - Walk-forward analysis
# - Monte Carlo simulation
# - Feature importance analysis
#
# Usage:
#   ./run_validation.sh                    # Run full validation
#   ./run_validation.sh --skip-walkforward  # Skip walk-forward (faster)

PYTHON="python3"

echo "🔍 Starting comprehensive strategy validation..."
echo ""

# Check if Python 3 is available
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ ERROR: Python 3 not found"
    exit 1
fi

# Run validation
$PYTHON run_full_validation.py "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Validation complete! Check artifacts/ directory for results."
else
    echo ""
    echo "❌ Validation failed. Check artifacts/validation.log for details."
fi

exit $EXIT_CODE
