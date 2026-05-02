#!/bin/bash
num=$1
for ((i=1; i<=num; i++)); do
    echo "▶️ Running inference $i/$num..."
    if ! python3 scripts/inference.py; then
        echo "❌ Run $i failed!"
        exit 1
    fi
done
echo "✅ All $num runs completed!"