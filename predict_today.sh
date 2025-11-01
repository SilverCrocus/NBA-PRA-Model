#!/bin/bash
# Auto-predict for games TODAY (whenever odds are available)

echo "=============================================="
echo "NBA PRA Predictions - Auto Date Detection"
echo "=============================================="

# Get today's date in EST
TODAY=$(TZ='America/New_York' date +%Y-%m-%d)
echo "Today (EST): $TODAY"

# Try today first
echo ""
echo "Attempting to fetch odds for $TODAY..."
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/run_daily.py --skip-training --date "$TODAY"

# Check if predictions were created with odds
PRED_FILE="production/outputs/predictions/predictions_${TODAY}.csv"

if [ -f "$PRED_FILE" ]; then
    # Count how many rows have odds
    ODDS_COUNT=$(python3 -c "import pandas as pd; df=pd.read_csv('$PRED_FILE'); print(df['pra_line'].notna().sum())")

    echo ""
    echo "=============================================="
    echo "RESULTS:"
    echo "  Predictions with odds: $ODDS_COUNT"

    if [ "$ODDS_COUNT" -gt "0" ]; then
        echo "  ✓ SUCCESS - Odds populated!"
        echo "  File: $PRED_FILE"
    else
        echo "  ✗ No odds found for $TODAY"
        echo "  (Games may have already started or odds not yet available)"
    fi
    echo "=============================================="
else
    echo ""
    echo "✗ Prediction file not created"
fi
