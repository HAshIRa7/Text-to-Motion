#!/bin/bash
for i in {1..10}; do
    echo "▶️ Running inference $i/10..."
    python scripts/inference.py
    # Bash automatically waits for the command to finish before the next loop
done
echo "✅ All 10 runs completed!"
