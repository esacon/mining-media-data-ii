import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class DataFormatAnalyzer:
    """
    Analyzes the data format of files within specified datasets.

    This class provides methods to inspect CSV, Excel, JSONL, and JSON files
    by sampling their initial records to understand structure, columns,
    data types, and sample values without loading entire files into memory.
    It supports identifying split files and provides a comprehensive summary
    of each dataset.
    """

    def __init__(self, data_dir: str = "src/data"):
        """
        Initializes the DataFormatAnalyzer.

        Args:
            data_dir (str): The path to the root directory containing datasets.
                            Each dataset is expected to be in a subdirectory
                            named 'dataset_*'.
        """
        self.data_dir = Path(data_dir)
        self.summary: Dict[str, Any] = {}

    def analyze_csv_file(self, file_path: Path, sample_size: int = 5) -> Dict[str, Any]:
        """
        Analyzes the format of a CSV file.

        Reads the first few rows to determine columns, data types, and sample values.
        Calculates file size and total row count efficiently.

        Args:
            file_path (Path): The path to the CSV file.
            sample_size (int): The number of rows to sample for analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the CSV file.
                            Includes file type, size, row/column counts, column details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            df_sample = pd.read_csv(file_path, nrows=sample_size)
            file_size = file_path.stat().st_size

            with open(file_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header row

            return {
                "file_type": "CSV",
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "total_rows": total_rows,
                "total_columns": len(df_sample.columns),
                "columns": {
                    col: {
                        "dtype": str(df_sample[col].dtype),
                        "sample_values": df_sample[col].dropna().head(3).tolist(),
                        "null_count_in_sample": df_sample[col].isnull().sum(),
                    }
                    for col in df_sample.columns
                },
                "sample_records": df_sample.head(3).to_dict("records"),
            }
        except Exception as e:
            return {"error": f"Failed to analyze CSV: {str(e)}"}

    def analyze_excel_file(
        self, file_path: Path, sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyzes the format of an Excel file.

        Reads the first few rows to determine columns, data types, and sample values.
        Calculates file size and total row count.

        Args:
            file_path (Path): The path to the Excel file.
            sample_size (int): The number of rows to sample for analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the Excel file.
                            Includes file type, size, row/column counts, column details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            df_sample = pd.read_excel(file_path, nrows=sample_size)
            file_size = file_path.stat().st_size
            # Read only first column to count rows
            df_full = pd.read_excel(file_path, usecols=[0])
            total_rows = len(df_full)

            return {
                "file_type": "Excel",
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "total_rows": total_rows,
                "total_columns": len(df_sample.columns),
                "columns": {
                    col: {
                        "dtype": str(df_sample[col].dtype),
                        "sample_values": df_sample[col].dropna().head(3).tolist(),
                        "null_count_in_sample": df_sample[col].isnull().sum(),
                    }
                    for col in df_sample.columns
                },
                "sample_records": df_sample.head(3).to_dict("records"),
            }
        except Exception as e:
            return {"error": f"Failed to analyze Excel: {str(e)}"}

    def analyze_jsonl_file(
        self, file_path: Path, sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyzes the format of a JSONL file.

        Reads the first few lines to determine record structure, fields, and sample values.
        Estimates total records for very large files, otherwise counts all lines.

        Args:
            file_path (Path): The path to the JSONL file.
            sample_size (int): The number of records to sample for structure analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the JSONL file.
                            Includes file type, size, record count, structure, field details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            file_size = file_path.stat().st_size
            sample_records = []
            total_lines: Any = 0

            # For very large files, estimate total lines; otherwise, count precisely
            if file_size > 100 * 1024 * 1024:  # 100MB threshold for estimation
                with open(file_path, "r", encoding="utf-8") as f:
                    for i in range(sample_size):
                        line = f.readline()
                        if not line:
                            break
                        try:
                            record = json.loads(line.strip())
                            sample_records.append(record)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

                if sample_records:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines_read = 0
                        bytes_read = 0
                        for line in f:
                            lines_read += 1
                            bytes_read += len(line.encode("utf-8"))
                            if lines_read >= 1000:
                                break
                        if lines_read > 0:
                            avg_line_size = bytes_read / lines_read
                            estimated_total = int(file_size / avg_line_size)
                            total_lines = f"~{estimated_total:,} (estimated)"
                        else:
                            total_lines = "Unknown"
                else:
                    total_lines = "Unknown"
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i < sample_size:
                            try:
                                record = json.loads(line.strip())
                                sample_records.append(record)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                        total_lines += 1

            field_analysis = {}
            if sample_records and isinstance(sample_records[0], dict):
                first_record = sample_records[0]
                for key in first_record.keys():
                    values = [
                        record.get(key)
                        for record in sample_records
                        if isinstance(record, dict) and key in record
                    ]
                    if values:
                        field_analysis[key] = {
                            "type": type(values[0]).__name__,
                            "sample_values": values[:3],
                        }

            return {
                "file_type": "JSONL",
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "total_records": total_lines,
                "record_structure": (
                    "dict"
                    if sample_records and isinstance(sample_records[0], dict)
                    else "other"
                ),
                "fields": field_analysis,
                "sample_records": sample_records[:3],
            }
        except Exception as e:
            return {"error": f"Failed to analyze JSONL: {str(e)}"}

    def analyze_json_file(
        self, file_path: Path, sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyzes the format of a JSON file.

        Peeks into the file to determine if it's a JSON array or object.
        For arrays, it attempts to parse the first few elements. For objects,
        it samples top-level keys and their types. Optimized for large files
        by reading chunks rather than the entire file.

        Args:
            file_path (Path): The path to the JSON file.
            sample_size (int): The number of elements/keys to sample for structure analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the JSON file.
                            Includes file type, size, structure type, and sampled data.
                            Returns an 'error' key if analysis fails.
        """
        try:
            file_size = file_path.stat().st_size

            with open(file_path, "r", encoding="utf-8") as f:
                chunk = f.read(50000)  # Read first 50KB to peek at structure
                chunk_stripped = chunk.strip()
                is_array = chunk_stripped.startswith("[")
                is_object = chunk_stripped.startswith("{")

                if is_array:
                    f.seek(0)
                    content = f.read(200000)  # Read more for array parsing
                    objects = []
                    brace_count = 0
                    current_obj_str = ""
                    in_string = False
                    escape_next = False

                    for char in content[1:]:
                        if escape_next:
                            escape_next = False
                            current_obj_str += char
                            continue
                        if char == "\\":
                            escape_next = True
                            current_obj_str += char
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                        current_obj_str += char
                        if brace_count == 0 and current_obj_str.strip().endswith("}"):
                            try:
                                obj = json.loads(current_obj_str.strip().rstrip(","))
                                objects.append(obj)
                                current_obj_str = ""
                                if len(objects) >= sample_size:
                                    break
                            except:
                                current_obj_str = ""

                    if objects:
                        first_obj = objects[0]
                        field_analysis = {}
                        if isinstance(first_obj, dict):
                            for key in first_obj.keys():
                                values = [
                                    obj.get(key)
                                    for obj in objects
                                    if isinstance(obj, dict) and key in obj
                                ]
                                if values:
                                    field_analysis[key] = {
                                        "type": type(values[0]).__name__,
                                        "sample_values": values[:3],
                                    }
                        return {
                            "file_type": "JSON Array",
                            "file_size_mb": round(file_size / (1024 * 1024), 2),
                            "estimated_records": f"Large array (sampled {len(objects)} records)",
                            "record_structure": (
                                "dict"
                                if isinstance(first_obj, dict)
                                else type(first_obj).__name__
                            ),
                            "fields": field_analysis,
                            "sample_records": objects[:3],
                        }
                    else:
                        return {
                            "file_type": "JSON Array (Partial)",
                            "file_size_mb": round(file_size / (1024 * 1024), 2),
                            "note": "Could not parse sample records from array. File may be malformed or too large for deep sampling.",
                            "first_chunk_preview": (
                                content[:200] + "..." if len(content) > 200 else content
                            ),
                        }

                elif is_object:
                    f.seek(0)
                    # Read first 500KB for object parsing
                    content = f.read(500000)
                    keys_found = []
                    sample_structure = {}
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    current_key = None
                    current_value_start = None
                    i = 0

                    while i < len(content) and len(sample_structure) < sample_size:
                        char = content[i]

                        if escape_next:
                            escape_next = False
                            i += 1
                            continue
                        if char == "\\":
                            escape_next = True
                            i += 1
                            continue
                        if char == '"':
                            if not in_string:
                                in_string = True
                                key_start = i + 1
                            else:
                                in_string = False
                                if current_key is None and brace_count == 1:
                                    potential_key = content[key_start:i]
                                    j = i + 1
                                    while j < len(content) and content[j] in " \t\n\r":
                                        j += 1
                                    if j < len(content) and content[j] == ":":
                                        current_key = potential_key
                                        keys_found.append(current_key)
                                        current_value_start = j + 1
                        elif not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                            elif (
                                char == ","
                                and brace_count == 1
                                and current_key is not None
                            ):
                                try:
                                    value_text = content[current_value_start:i].strip()
                                    if value_text.startswith(
                                        '"'
                                    ) and value_text.endswith('"'):
                                        sample_value = json.loads(value_text)
                                        sample_structure[current_key] = {
                                            "type": "str",
                                            "sample_value": (
                                                str(sample_value)[:100] + "..."
                                                if len(str(sample_value)) > 100
                                                else str(sample_value)
                                            ),
                                        }
                                    elif value_text.startswith("["):
                                        bracket_count = 0
                                        array_end = current_value_start
                                        for k in range(
                                            current_value_start,
                                            min(i, current_value_start + 10000),
                                        ):
                                            if content[k] == "[":
                                                bracket_count += 1
                                            elif content[k] == "]":
                                                bracket_count -= 1
                                                if bracket_count == 0:
                                                    array_end = k + 1
                                                    break
                                        array_text = content[
                                            current_value_start:array_end
                                        ]
                                        parsed_array = json.loads(array_text)
                                        sample_structure[current_key] = {
                                            "type": "list",
                                            "sample_value": f"Array with {len(parsed_array)} elements. First element: {str(parsed_array[0])[:100] + '...' if parsed_array else 'Empty array'}",
                                        }
                                    elif value_text.startswith("{"):
                                        sample_structure[current_key] = {
                                            "type": "dict",
                                            "sample_value": "Object (nested structure)",
                                        }
                                    else:
                                        parsed_value = json.loads(value_text)
                                        sample_structure[current_key] = {
                                            "type": type(parsed_value).__name__,
                                            "sample_value": str(parsed_value),
                                        }
                                except Exception as parse_e:
                                    sample_structure[current_key] = {
                                        "type": "unknown",
                                        "sample_value": f"Parse error: {str(parse_e)[:50]}",
                                    }
                                current_key = None
                                current_value_start = None
                        i += 1

                    return {
                        "file_type": "JSON Object",
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "top_level_keys": keys_found[:10],
                        "total_top_level_keys": (
                            len(keys_found) if len(keys_found) < 10 else "10+"
                        ),
                        "structure_sample": sample_structure,
                        "note": f"Large JSON object with {len(keys_found) if len(keys_found) < 10 else '10+'} top-level keys (showing first {len(sample_structure)})",
                    }

                return {
                    "file_type": "JSON (Unknown Structure)",
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "note": "Could not determine JSON structure (array/object) from initial chunk.",
                    "first_chunk_preview": (
                        chunk[:200] + "..." if len(chunk) > 200 else chunk
                    ),
                }

        except Exception as e:
            return {"error": f"Failed to analyze JSON: {str(e)}"}

    def analyze_split_file(
        self, file_path: Path, sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Analyzes a potential split file (e.g., 'part_aa', 'part_ab').

        Attempts to determine if the split file contains JSONL data by
        reading its first line.

        Args:
            file_path (Path): The path to the split file.
            sample_size (int): The number of records to sample if identified as JSONL.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the split file.
                            Identifies if it's a JSONL split part or an unknown binary file.
                            Returns an 'error' key if analysis fails.
        """
        try:
            file_size = file_path.stat().st_size

            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        json.loads(first_line)
                        result = self.analyze_jsonl_file(file_path, sample_size)
                        result["file_type"] = "JSONL Split Part"
                        result["note"] = "This is a split part of a larger JSONL file."
                        return result
                    except json.JSONDecodeError:
                        f.seek(0)
                        chunk = f.read(1000)
                        if "{" in chunk and '"' in chunk:
                            return {
                                "file_type": "JSONL Split Part (Partial)",
                                "file_size_mb": round(file_size / (1024 * 1024), 2),
                                "note": "Split part of JSONL file - may not start with a complete JSON record.",
                            }

            return {
                "file_type": "Split File (Binary/Unknown)",
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "note": "This appears to be a split file part. May need to be reassembled for analysis.",
            }
        except Exception as e:
            return {"error": f"Failed to analyze split file: {str(e)}"}

    def analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Analyzes all relevant files within a single dataset directory.

        Identifies and analyzes CSV, Excel, JSONL, JSON, and split files.
        It prioritizes analyzing split file parts over their reassembled counterparts
        to avoid redundant analysis.

        Args:
            dataset_path (Path): The path to the dataset directory.

        Returns:
            Dict[str, Any]: A dictionary containing the summary for the dataset,
                            including details for each analyzed file.
        """
        dataset_summary: Dict[str, Any] = {
            "dataset_name": dataset_path.name,
            "files": {},
        }

        files_in_dir = list(dataset_path.iterdir())
        split_files = [f for f in files_in_dir if f.is_file() and "part_" in f.name]
        reassembled_files: List[Path] = []

        if split_files:
            base_names = set()
            for split_file in split_files:
                name_parts = split_file.name.split("_part_")
                if len(name_parts) == 2:
                    base_names.add(name_parts[0])

            for f in files_in_dir:
                if f.is_file() and any(
                    f.name.startswith(base_name) and "part_" not in f.name
                    for base_name in base_names
                ):
                    reassembled_files.append(f)

        for file_path in files_in_dir:
            if file_path.is_file() and not file_path.name.startswith("."):
                if file_path in reassembled_files:
                    print(
                        f"Skipping {file_path.name} (split parts available for analysis)"
                    )
                    dataset_summary["files"][file_path.name] = {
                        "file_type": "Reassembled File (Skipped)",
                        "file_size_mb": round(
                            file_path.stat().st_size / (1024 * 1024), 2
                        ),
                        "note": "Skipped analysis - split parts analyzed instead to avoid redundancy.",
                    }
                    continue

                file_ext = file_path.suffix.lower()
                print(f"Analyzing {file_path.name}...")

                if file_ext == ".csv":
                    analysis = self.analyze_csv_file(file_path)
                elif file_ext == ".xlsx":
                    analysis = self.analyze_excel_file(file_path)
                elif file_ext == ".jsonl":
                    analysis = self.analyze_jsonl_file(file_path)
                elif file_ext == ".json":
                    analysis = self.analyze_json_file(file_path)
                elif "part_" in file_path.name:
                    analysis = self.analyze_split_file(file_path)
                else:
                    analysis = {
                        "file_type": f"Unknown ({file_ext})",
                        "file_size_mb": round(
                            file_path.stat().st_size / (1024 * 1024), 2
                        ),
                    }
                dataset_summary["files"][file_path.name] = analysis
        return dataset_summary

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generates a data format summary for all datasets found in the data directory.

        Iterates through subdirectories starting with 'dataset_' and analyzes each.

        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive summary
                            of all analyzed datasets and their files.
        """
        print("Starting data format analysis...")
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                print(f"\nAnalyzing {item.name}...")
                self.summary[item.name] = self.analyze_dataset(item)
        return self.summary

    def save_summary(
        self, output_file: str = "src/data/_analysis/data_format_summary.json"
    ):
        """
        Saves the generated data format summary to a JSON file.

        Args:
            output_file (str): The name of the output JSON file.
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, default=str)
        print(f"\nSummary saved to {output_file}")

    def print_summary(self):
        """
        Prints a formatted summary of the data formats to the console.

        Displays dataset names, file names, types, sizes, row/record counts,
        and sampled column/field information.
        """
        print("\n" + "=" * 80)
        print("DATA FORMAT SUMMARY")
        print("=" * 80)

        for dataset_name, dataset_info in self.summary.items():
            print(f"\n📁 {dataset_name.upper()}")
            print("-" * 50)

            for file_name, file_info in dataset_info["files"].items():
                print(f"\n📄 {file_name}")
                print(f"   Type: {file_info.get('file_type', 'Unknown')}")
                print(f"   Size: {file_info.get('file_size_mb', 'Unknown')} MB")

                if "total_rows" in file_info:
                    print(f"   Rows: {file_info['total_rows']:,}")
                elif "total_records" in file_info:
                    print(f"   Records: {file_info['total_records']}")

                if "total_columns" in file_info:
                    print(f"   Columns: {file_info['total_columns']}")
                elif "fields" in file_info and file_info["fields"]:
                    print(f"   Fields: {len(file_info['fields'])}")

                if "columns" in file_info:
                    print("   Column Info (Sample):")
                    for col, col_info in list(file_info["columns"].items())[:5]:
                        print(f"     • {col}: {col_info['dtype']}")
                elif "fields" in file_info and file_info["fields"]:
                    print("   Field Info (Sample):")
                    for field, field_info in list(file_info["fields"].items())[:5]:
                        print(f"     • {field}: {field_info['type']}")

                if "error" in file_info:
                    print(f"   ⚠️ Error: {file_info['error']}")
                if "note" in file_info:
                    print(f"   ℹ️ Note: {file_info['note']}")


def main():
    """
    Main function to execute the data format analysis.
    """
    analyzer = DataFormatAnalyzer()
    try:
        analyzer.generate_summary()
        analyzer.print_summary()
        analyzer.save_summary()
        print(f"\n✅ Analysis complete! Summary saved to data_format_summary.json")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
