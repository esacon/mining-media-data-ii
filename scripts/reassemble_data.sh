#!/bin/bash

# Script to reassemble split data files
# This script combines the split parts of playerLogs_game2_playerbasedlines.jsonl

echo "Reassembling playerLogs_game2_playerbasedlines.jsonl..."

# Check if the parts exist
if [ ! -f "src/data/dataset_game2/playerLogs_game2_playerbasedlines_part_aa" ]; then
    echo "Error: Split parts not found. Make sure you're running this from the project root."
    exit 1
fi

# Combine the parts
cat src/data/dataset_game2/playerLogs_game2_playerbasedlines_part_* > src/data/dataset_game2/playerLogs_game2_playerbasedlines.jsonl

echo "Reassembly complete. Original file restored to src/data/dataset_game2/playerLogs_game2_playerbasedlines.jsonl"
echo "File size: $(ls -lh src/data/dataset_game2/playerLogs_game2_playerbasedlines.jsonl | awk '{print $5}')"
