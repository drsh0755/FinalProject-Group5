#!/bin/bash
# Convenience script to train TFT model

set -e

# Default values
CONFIG="training/train_config.yaml"
DATA="data/processed/features_with_sentiment.csv"
USE_DDG_DA=false
DDG_DA_MODEL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --use-ddg-da)
            USE_DDG_DA=true
            shift
            ;;
        --ddg-da-model)
            DDG_DA_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 -m training.train_tft --config $CONFIG --data $DATA"

if [ "$USE_DDG_DA" = true ]; then
    CMD="$CMD --use-ddg-da"
    if [ -n "$DDG_DA_MODEL" ]; then
        CMD="$CMD --ddg-da-model $DDG_DA_MODEL"
    fi
fi

echo "Running training with command:"
echo "$CMD"
echo ""

# Run training
eval $CMD
