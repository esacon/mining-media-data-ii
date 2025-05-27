# Data Directory: Project Datasets

This directory houses the datasets integral to the Mining Media Data II project. Each dataset has been meticulously analyzed for its structure, data types, and content, providing a foundational understanding for research and analysis.

## 📊 Dataset Overview

A high-level summary of the datasets included:

| Dataset       | Description                    | Total Size | Number of Files            | Total Records / Entities |
| :------------ | :----------------------------- | :--------- | :------------------------- | :----------------------- |
| **dataset_1** | Game 1 raw interaction data    | ~7.6 MB    | 2                          | 153,929 rows             |
| **dataset_2** | Game 2 detailed player logs    | ~4.5 GB    | 4 (1 reassembled, 3 split) | ~30,009 players          |
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
- `playerLogs_game2_playerbasedlines_part_aa`: 1.50 GB - _Primary analysis source due to complete records._
- `playerLogs_game2_playerbasedlines_part_ab`: 1.50 GB - _Split part._
- `playerLogs_game2_playerbasedlines_part_ac`: 1.47 GB - _Split part._

#### Schema:

Each record in the JSONL files represents player data, typically structured as follows:

| Field     | Type     | Description                                                                |
| :-------- | :------- | :------------------------------------------------------------------------- |
| `uid`     | `string` | Unique player identifier (40-character hexadecimal string).                |
| `records` | `array`  | An array containing a chronological sequence of player events and actions. |

#### Player Event Structure:

Each object within the `records` array corresponds to a specific player event and contains the following fields:

| Field        | Type      | Description                                                           |
| :----------- | :-------- | :-------------------------------------------------------------------- |
| `time`       | `integer` | Unix timestamp of the event in milliseconds.                          |
| `date`       | `string`  | ISO 8601 formatted date and time of the event.                        |
| `clientIp`   | `string`  | Anonymized IP address of the player's client.                         |
| `app`        | `string`  | Application identifier (e.g., "DO").                                  |
| `appVersion` | `string`  | Version string of the application.                                    |
| `device`     | `string`  | Type of device used (e.g., "iPod", "iPhone").                         |
| `event`      | `string`  | Type of event recorded (e.g., "softPurchase", "progress", "install"). |
| `platform`   | `string`  | Operating platform ("iOS", "Android").                                |
| `properties` | `object`  | Event-specific data, varying based on the `event` type.               |
| `uid`        | `string`  | Player identifier (redundant with top-level `uid`).                   |
| `calcDate`   | `string`  | Calculated date for analytical purposes.                              |

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
      "calcDate": "2014-07-25"
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

- Server information
- Match metadata (e.g., duration, time limits)
- Team data and scores
- Individual player statistics within that match

**`bulkplayers.json`**:
Each key represents a **Player ID** (as a numeric string). The corresponding value is an array of player objects, which typically include:

- Authentication status
- Player names and profile information
- Aggregated game statistics and achievements
- Team affiliations

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
- Generates a detailed and structured summary of each dataset.

### Analysis Output

The analysis results are saved to `data_format_summary.json` (located in the project root by default) with the following details:

- File sizes and total record/row counts.
- In-depth data type analysis for each column/field.
- Representative sample values and structural insights.
- Descriptions of identified fields/columns.
- Error handling messages for files that could not be fully analyzed (e.g., due to corruption or extreme size).

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
cat src/data/dataset_2/playerLogs_game2_playerbasedlines_part_* > src/data/dataset_2/playerLogs_game2_playerbasedlines.jsonl
```

#### Git LFS Storage

All files exceeding 100MB are managed using Git Large File Storage (LFS):

- `dataset_2/playerLogs_game2_playerbasedlines_part_*`
- `dataset_3/bulkmatches.json`
- `dataset_3/bulkplayers.json`

Ensure Git LFS is installed and configured to properly clone and access these files.

---

## 📋 Data Quality Notes

Specific considerations regarding the quality and nature of the data in each dataset.

### Dataset 2 Considerations

- **Split Files**: Individual split files (`_part_aa`, etc.) may not begin with a complete JSON record.
- **Primary Analysis Source**: For reliable record-by-record processing, it is recommended to start reading from `playerLogs_game2_playerbasedlines_part_aa`, as it is guaranteed to contain complete initial records.
- **Player Count**: An estimated ~30,009 unique players are present across all parts of this dataset.

### Dataset 3 Considerations

- **Large JSON Objects**: `bulkmatches.json` and `bulkplayers.json` are very large single JSON objects. Direct `json.load()` may consume significant memory, potentially leading to `MemoryError` for systems with limited RAM.
- **Numeric String Keys**: Data within these files is indexed by numeric string keys (e.g., `"1"`, `"2"`). Ensure your parsing logic treats these as strings when accessing.
- **Parsing Limitations**: Due to the file size, deep parsing of the entire file might be slow or memory-intensive. Consider using streaming JSON parsers (e.g., `ijson`) for more efficient access to subsets of data.

### Recommended Analysis Approach

1.  **Initial Exploration (Dataset 1)**: Begin by analyzing `dataset_1` to understand basic game patterns and data handling.
2.  **Player Behavior (Dataset 2)**: For detailed player behavior, focus on processing `dataset_2` using its split parts (`_part_aa`, `_part_ab`, `_part_ac`) to manage memory effectively.
3.  **Match/Player Relationships (Dataset 3)**: When working with `dataset_3`, employ careful memory management. Sample or partially load data to analyze match and player relationships without loading the entire gigabytes-sized files.
4.  **Automated Profiling**: Utilize the provided `data_format_summary.py` script for an initial, efficient profiling of all datasets, obtaining vital statistics and structural insights before deeper analysis.

---

### Performance Considerations for Large Datasets

- **Sampling**: Always use sampling for initial exploration and hypothesis generation when dealing with large files to save computational resources and time.
- **Memory Management**: Be mindful of memory consumption, especially when attempting to load full datasets into RAM. Python libraries like `pandas` and `json` can be memory-intensive with large files.
- **Automated Profiling**: Leverage the `data_format_summary.py` script for efficient data profiling, which helps in understanding the data's characteristics without full loading.
- **Appropriate Loading Strategies**: Select the most suitable data loading strategy for each format and file size (e.g., `pd.read_csv` for CSV, iterative line reading for JSONL, streaming parsers like `ijson` for very large JSON objects).
