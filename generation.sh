#!/bin/bash
for i in {1..20}; do
    echo "▶️ Running inference $i/20..."
    python scripts/inference.py
    # Bash automatically waits for the command to finish before the next loop
done
echo "✅ All 20 runs completed!"
