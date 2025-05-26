# Data Directory

This directory contains the datasets for the Mining Media Data II project.

## Dataset Structure

- `dataset_1/`: Game 1 raw data in CSV and Excel formats
- `dataset_2/`: Game 2 player logs (split into multiple parts due to size constraints)
- `dataset_3/`: Bulk matches and players data in JSON format

## Large File Handling

Due to GitHub's file size limitations, the large `playerLogs_game2_playerbasedlines.jsonl` file (4.4GB) has been split into smaller parts:

- `playerLogs_game2_playerbasedlines_part_aa` (1.5GB)
- `playerLogs_game2_playerbasedlines_part_ab` (1.5GB)  
- `playerLogs_game2_playerbasedlines_part_ac` (1.4GB)

### Reassembling the Split File

To reassemble the original file, run the following script from the project root:

```bash
./scripts/reassemble_data.sh
```

This will create the original `playerLogs_game2_playerbasedlines.jsonl` file in `src/data/dataset_2/`.

### Manual Reassembly

Alternatively, you can manually reassemble the file using:

```bash
cat src/data/dataset_2/playerLogs_game2_playerbasedlines_part_* > src/data/dataset_2/playerLogs_game2_playerbasedlines.jsonl
```

## File Sizes

- `dataset_1/rawdata_game1.csv`: ~5MB
- `dataset_1/rawdata_game1.xlsx`: ~2MB
- `dataset_2/playerLogs_game2_playerbasedlines.jsonl`: 4.4GB (when reassembled)
- `dataset_3/bulkmatches.json`: 336MB
- `dataset_3/bulkplayers.json`: 934MB

All large files (>100MB) are stored using Git LFS. 