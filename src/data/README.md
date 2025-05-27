# Data Directory: Project Datasets

This directory houses the datasets integral to the Mining Media Data II project. Each dataset has been meticulously analyzed for its structure, data types, and content, providing a foundational understanding for research and analysis.

## 📊 Dataset Overview

A high-level summary of the datasets included:

| Dataset       | Description                    | Total Size | Number of Files            | Total Records / Entities |
| :------------ | :----------------------------- | :--------- | :------------------------- | :----------------------- |
| **dataset_1** | Game 1 raw interaction data    | ~7.6 MB    | 2                          | 153,929 rows             |
| **dataset_2** | Game 2 detailed player logs    | ~4.5 GB    | 4 (1 reassembled, 3 split) | ~27,566 players          |
| **dataset_3** | Bulk match and player profiles | ~1.2 GB    | 2                          | Indexed data             |

---

## 📁 Dataset Structure and Details

Detailed information about each dataset's purpose, format, files, schema, and sample records.

### Dataset 1: Game 1 Raw Data

**Purpose**: Captures fundamental game interaction data, including device tracking.

**Format**: Available in both CSV and Excel formats, containing identical content.

#### Files:

- `rawdata_game1.csv`: 4.58 MB, 153,929 rows
- `rawdata_game1.xlsx`: 3.05 MB, 153,929 rows

#### Schema:

| Column   | Type    | Description                                   | Sample Values              |
| :------- | :------ | :-------------------------------------------- | :------------------------- |
| `device` | `int64` | Unique identifier for the game device.        | `352610060979119`          |
| `score`  | `int64` | The score achieved by the player in the game. | `7`, `0`, `6`              |
| `time`   | `int64` | Unix timestamp of the event.                  | `1421157320`, `1421157288` |

#### Sample Record (from CSV/Excel):

```json
{
  "device": 352610060979119,
  "score": 7,
  "time": 1421157320
}
```

---

### Dataset 2: Game 2 Player Logs

**Purpose**: Provides highly detailed player behavior and event tracking information.

**Format**: JSON Lines (JSONL) - Due to its substantial size, this dataset is split into multiple files.

#### Files:

- `playerLogs_game2_playerbasedlines.jsonl`: 4.47 GB - _This is the reassembled file. For primary analysis, it's often more efficient to process the split parts directly to manage memory._
- `playerLogs_game2_playerbasedlines_part_aa`: 1.50 GB - _Primary analysis source with 27,566 complete records._
- `playerLogs_game2_playerbasedlines_part_ab`: 1.50 GB - _Split part (partial records)._
- `playerLogs_game2_playerbasedlines_part_ac`: 1.47 GB - _Split part (partial records)._

#### Schema:

Each record in the JSONL files represents player data, typically structured as follows:

| Field     | Type     | Description                                                                |
| :-------- | :------- | :------------------------------------------------------------------------- |
| `uid`     | `string` | Unique player identifier (40-character hexadecimal string).                |
| `records` | `array`  | An array containing a chronological sequence of player events and actions. |

#### Player Event Structure:

Each object within the `records` array corresponds to a specific player event and contains the following fields:

| Field           | Type      | Description                                                           |
| :-------------- | :-------- | :-------------------------------------------------------------------- |
| `time`          | `integer` | Unix timestamp of the event in milliseconds.                          |
| `date`          | `string`  | ISO 8601 formatted date and time of the event.                        |
| `clientIp`      | `string`  | Anonymized IP address of the player's client.                         |
| `app`           | `string`  | Application identifier (e.g., "DO").                                  |
| `appVersion`    | `string`  | Version string of the application.                                    |
| `device`        | `string`  | Type of device used (e.g., "iPod", "iPhone").                         |
| `event`         | `string`  | Type of event recorded (e.g., "softPurchase", "progress", "install"). |
| `platform`      | `string`  | Operating platform ("iOS", "Android").                                |
| `properties`    | `object`  | Event-specific data, varying based on the `event` type.               |
| `uid`           | `string`  | Player identifier (redundant with top-level `uid`).                   |
| `calcDate`      | `string`  | Calculated date for analytical purposes.                              |
| `localTime`     | `integer` | Local timestamp in milliseconds.                                      |
| `queueDuration` | `string`  | Duration in queue (as string).                                        |

#### Sample Player Record:

```json
{
  "uid": "0001E7ED9ECB34E9A1D31DE15B334E32001B32BD",
  "records": [
    {
      "time": 1406267278630,
      "date": "2014-07-25T05:47:58.63Z",
      "clientIp": "66.56.37.0",
      "app": "DO",
      "appVersion": "2.4.67.7442",
      "device": "iPod",
      "event": "softPurchase",
      "platform": "iOS",
      "properties": {
        "balance": 2465,
        "currency": "soft",
        "id": "scooter_clone_suspension_1_2",
        "price": 400,
        "remaining": 2065,
        "type": "vehicle_upgrade"
      },
      "uid": "0001E7ED9ECB34E9A1D31DE15B334E32001B32BD",
      "calcDate": "2014-07-25T05:47:54.0000000",
      "localTime": "1406267271000",
      "queueDuration": "4000"
    }
  ]
}
```

---

### Dataset 3: Bulk Matches and Players

**Purpose**: Contains aggregated game match data and comprehensive player profiles.

**Format**: Large JSON objects, structured for indexed access.

#### Files:

- `bulkmatches.json`: 320.86 MB - Contains match data indexed by match ID.
- `bulkplayers.json`: 890.47 MB - Contains player data indexed by player ID.

#### Structure:

Both files utilize numeric string keys (e.g., `"1"`, `"2"`, `"3"`) as primary indices for accessing individual records.

**`bulkmatches.json`**:
Each key represents a **Match ID** (as a numeric string). The corresponding value is a match object detailing:

| Field       | Type      | Description                                  |
| :---------- | :-------- | :------------------------------------------- |
| `server`    | `string`  | Server hostname where the match was played   |
| `port`      | `integer` | Server port number                           |
| `official`  | `boolean` | Whether this was an official match           |
| `group`     | `string`  | Match group identifier                       |
| `date`      | `integer` | Unix timestamp of the match                  |
| `timeLimit` | `integer` | Time limit for the match in minutes          |
| `duration`  | `integer` | Actual match duration in milliseconds        |
| `mapId`     | `integer` | Identifier for the map used                  |
| `teams`     | `array`   | Array of team objects with scores and splats |

**`bulkplayers.json`**:
Each key represents a **Player ID** (as a numeric string). The corresponding value is an array of player objects, which typically include:

| Field    | Type      | Description                         |
| :------- | :-------- | :---------------------------------- |
| `auth`   | `boolean` | Whether the player is authenticated |
| `name`   | `string`  | Player display name                 |
| `flair`  | `integer` | Player flair/badge identifier       |
| `degree` | `integer` | Player degree/level                 |
| `score`  | `integer` | Player score in the match           |
| `points` | `integer` | Points earned by the player         |
| `team`   | `integer` | Team identifier (0, 1, or 2)        |
| `events` | `string`  | Encoded event data for the player   |

#### Sample Match Record:

```json
{
  "1": {
    "server": "tagpro-chord.koalabeast.com",
    "port": 8008,
    "official": true,
    "group": "",
    "date": 1432576197,
    "timeLimit": 12,
    "duration": 22562,
    "mapId": 6,
    "teams": [
      {
        "name": "Red",
        "score": 1,
        "splats": "mp59z/DaK5eWsvU1WURi5P0ufn/XUS4TVqOBt9RPBWsOipTXullw2rVSDH6Vlb9WI7eqkqNWFxhriYFTv7maMN8VZw7enoWTKJw4l6I+yqEnqHt19HzBOe4g3N4xtqKWsVoruZLaRU3hH6t5iRLWVAk5G7zzrBJNiFcu9Hw="
      },
      {
        "name": "Blue",
        "score": 3,
        "splats": "psKLSdkU5Ql2Mj1JaTciTKnUgn4nLT/TWKiVq85VLOXqarSbLNq/JC1gb8akuRMqNCKbufHK/rJn+GMzCjSZGyHMpY9eTlirJsySkz45T7LDpXOR9EM5agBbPRtwdnm4CTeEmqEiAsp42SVkbpK5wCk2IySdcSKhRq1WU4miQXbQQq3qjGhTT7yJl2YMu21eaBYI"
      }
    ]
  }
}
```

#### Sample Player Record:

```json
{
  "1": [
    {
      "auth": true,
      "name": "RoDyMaRy",
      "flair": 105,
      "degree": 99,
      "score": 31,
      "points": 25,
      "team": 1,
      "events": "AREVYIpPCAe0QCBSIAisARCvARK/ARCjARArIAgDARAfARCjARArARArCAbrQCDMCAdDAKHyQKELAFRmCAQPQKD3ARV6ARCjIAgrAREkARCjARBnAREbIAg9ARAhARArIAQBECYAUIoBETQBECsAUAsBEA8BECsBEdAgCTAAUS4AURtAF/EAU9YIBD1AoMlAGDIgDC0ICAHqAKCQQKELIAmIAFC7AFFU"
    }
  ]
}
```

---

## 🔧 Data Analysis Tools

### Automated Analysis Script

The `src/data/_analysis/data_format_summary.py` script is provided for comprehensive, automated analysis of dataset formats.

To run the script:

```bash
# Execute from the project root directory
python src/data/_analysis/data_format_summary.py
```

**Key Features**:

- Supports analysis of CSV, Excel, JSON, and JSONL file formats.
- Efficiently samples data to infer types and structures without loading entire large files.
- Accurately identifies data types, column/field names, and provides sample values.
- Intelligently handles large split files (e.g., `_part_aa` files).
- Uses streaming analysis for very large files (>100MB) to manage memory efficiently.
- Generates a detailed and structured summary of each dataset.

### Analysis Output

The analysis results are saved to `src/data/_analysis/data_format_summary.json` with the following details:

- File sizes and total record/row counts.
- In-depth data type analysis for each column/field.
- Representative sample values and structural insights.
- Descriptions of identified fields/columns.
- Error handling messages for files that could not be fully analyzed (e.g., due to corruption or extreme size).
- Streaming analysis notes for large files processed with memory-efficient methods.

---

## 📦 Large File Handling

### Split File Management

Due to standard Git and GitHub file size limitations, some large datasets (specifically `dataset_2`) are split into multiple smaller files.

#### Reassembling Split Files

To work with the complete dataset, these split files can be reassembled:

```bash
# Recommended: Use the provided reassembly script
./scripts/reassemble_data.sh

# Alternative: Manual reassembly using cat (Linux/macOS)
cat src/data/dataset_game2/playerLogs_game2_playerbasedlines_part_* > src/data/dataset_game2/playerLogs_game2_playerbasedlines.jsonl
```

#### Git LFS Storage

All files exceeding 100MB are managed using Git Large File Storage (LFS):

- `dataset_game2/playerLogs_game2_playerbasedlines_part_*`
- `dataset_game3/bulkmatches.json`
- `dataset_game3/bulkplayers.json`

Ensure Git LFS is installed and configured to properly clone and access these files.

---

## 📋 Data Quality Notes

Specific considerations regarding the quality and nature of the data in each dataset.

### Dataset 2 Considerations

- **Split Files**: Individual split files (`_part_ab` and `_part_ac`) may not begin with a complete JSON record and are marked as "partial" in the analysis.
- **Primary Analysis Source**: For reliable record-by-record processing, it is recommended to start reading from `playerLogs_game2_playerbasedlines_part_aa`, as it contains 27,566 complete initial records.
- **Player Count**: Approximately 27,566 unique players are present in the first split part, with additional players in subsequent parts.

### Dataset 3 Considerations

- **Large JSON Objects**: `bulkmatches.json` and `bulkplayers.json` are very large single JSON objects. The analysis uses streaming methods to avoid memory issues.
- **Numeric String Keys**: Data within these files is indexed by numeric string keys (e.g., `"1"`, `"2"`). Ensure your parsing logic treats these as strings when accessing.
- **Limited Key Sampling**: Due to file size, only the first 5 top-level keys are analyzed in detail. The actual files contain many more entries.
- **Streaming Analysis**: Both files are analyzed using streaming JSON parsers (`ijson`) for memory efficiency, providing complete sample objects for the first few keys.

### Recommended Analysis Approach

1.  **Initial Exploration (Dataset 1)**: Begin by analyzing `dataset_1_game1` to understand basic game patterns and data handling.
2.  **Player Behavior (Dataset 2)**: For detailed player behavior, focus on processing `dataset_game2` using its split parts (`_part_aa`, `_part_ab`, `_part_ac`) to manage memory effectively. Start with `_part_aa` for complete records.
3.  **Match/Player Relationships (Dataset 3)**: When working with `dataset_game3`, employ careful memory management. Use streaming JSON parsers or sample specific keys to analyze match and player relationships without loading entire files.
4.  **Automated Profiling**: Utilize the provided `data_format_summary.py` script for an initial, efficient profiling of all datasets, obtaining vital statistics and structural insights before deeper analysis.

---

### Performance Considerations for Large Datasets

- **Sampling**: Always use sampling for initial exploration and hypothesis generation when dealing with large files to save computational resources and time.
- **Memory Management**: Be mindful of memory consumption, especially when attempting to load full datasets into RAM. The analysis script uses streaming methods for files >100MB.
- **Streaming Parsers**: For very large JSON files, use streaming parsers like `ijson` instead of loading entire files into memory.
- **Split File Processing**: Process split files individually rather than reassembling them to maintain memory efficiency.
- **Appropriate Loading Strategies**: Select the most suitable data loading strategy for each format and file size (e.g., `pd.read_csv` for CSV, iterative line reading for JSONL, streaming parsers for very large JSON objects).
