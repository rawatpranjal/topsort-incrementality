#!/bin/bash

# ARDL Analysis Runner Script
# Usage: ./run_analysis.sh [options]

echo "=================================="
echo "Running ARDL Analysis"
echo "=================================="

# Default settings
MODELS="purchases aov revenue"
MAX_LAGS=7
OUTPUT="results_$(date +%Y%m%d_%H%M%S).txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODELS="revenue"
            MAX_LAGS=3
            shift
            ;;
        --full)
            MAX_LAGS=14
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run analysis
echo "Settings:"
echo "  Models: $MODELS"
echo "  Max lags: $MAX_LAGS"
echo "  Output: $OUTPUT"
echo ""

python3 main.py \
    --models $MODELS \
    --max-lags $MAX_LAGS \
    --output $OUTPUT

echo ""
echo "Analysis complete!"
echo "Results saved to: $OUTPUT"

# Show summary
echo ""
echo "Quick Summary:"
grep -A 2 "FINAL SUMMARY" $OUTPUT | tail -n 20