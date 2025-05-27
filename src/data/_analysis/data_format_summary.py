import gc
import json
import mmap
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import ijson
import pandas as pd


class DataFormatAnalyzer:
    """Analyzes the data format of files within specified datasets.

    This class provides methods to inspect CSV, Excel, JSONL, and JSON files
    by sampling their initial records to understand structure, columns,
    data types, and sample values without loading entire files into memory.
    It supports identifying split files and provides a comprehensive summary
    of each dataset.
    """

    def __init__(self, data_dir: str = "src/data"):
        """Initializes the DataFormatAnalyzer.

        Args:
            data_dir (str): The path to the root directory containing datasets.
                            Each dataset is expected to be in a subdirectory
                            named 'dataset_*'.
        """
        self.data_directory = Path(data_dir)
        self.analysis_summary: Dict[str, Any] = {}

    def analyze_csv_file(
        self, file_path: Path, sample_record_count: int = 5
    ) -> Dict[str, Any]:
        """Analyzes the format of a CSV file.

        Reads the first few rows to determine columns, data types, and sample values.
        Calculates file size and total row count efficiently.

        Args:
            file_path (Path): The path to the CSV file.
            sample_record_count (int): The number of rows to sample for analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the CSV file.
                            Includes file type, size, row/column counts, column details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            sample_df = pd.read_csv(file_path, nrows=sample_record_count)
            file_size_bytes = file_path.stat().st_size

            with open(file_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1

            return {
                "file_type": "CSV",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_rows": total_rows,
                "column_count": len(sample_df.columns),
                "column_details": {
                    col: {
                        "data_type": str(sample_df[col].dtype),
                        "sample_values": sample_df[col].dropna().head(3).tolist(),
                        "null_count_in_sample": sample_df[col].isnull().sum(),
                    }
                    for col in sample_df.columns
                },
                "sample_records": sample_df.head(3).to_dict("records"),
            }
        except Exception as e:
            return {"error": f"Failed to analyze CSV: {str(e)}"}

    def analyze_excel_file(
        self, file_path: Path, sample_record_count: int = 5
    ) -> Dict[str, Any]:
        """Analyzes the format of an Excel file.

        Reads the first few rows to determine columns, data types, and sample values.
        Calculates file size and total row count.

        Args:
            file_path (Path): The path to the Excel file.
            sample_record_count (int): The number of rows to sample for analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the Excel file.
                            Includes file type, size, row/column counts, column details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            sample_df = pd.read_excel(file_path, nrows=sample_record_count)
            file_size_bytes = file_path.stat().st_size
            total_rows = pd.read_excel(file_path, usecols=[0]).shape[0]

            return {
                "file_type": "Excel",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_rows": total_rows,
                "column_count": len(sample_df.columns),
                "column_details": {
                    col: {
                        "data_type": str(sample_df[col].dtype),
                        "sample_values": sample_df[col].dropna().head(3).tolist(),
                        "null_count_in_sample": sample_df[col].isnull().sum(),
                    }
                    for col in sample_df.columns
                },
                "sample_records": sample_df.head(3).to_dict("records"),
            }
        except Exception as e:
            return {"error": f"Failed to analyze Excel: {str(e)}"}

    def _read_jsonl_samples_from_mmap(
        self, file_path: Path, sample_count: int, file_size_bytes: int
    ) -> Tuple[List[Dict], Union[int, str]]:
        """Reads sample records and estimates total lines from a large JSONL file using mmap.

        Args:
            file_path (Path): The path to the JSONL file.
            sample_count (int): The number of records to sample.
            file_size_bytes (int): The size of the file in bytes.

        Returns:
            Tuple[List[Dict], Union[int, str]]: A tuple containing a list of sample records
                                                and the estimated total number of lines (or "Unknown").
        """
        sample_records = []
        total_lines = 0
        try:
            with open(file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    for line_num, line in enumerate(iter(mmapped_file.readline, b"")):
                        total_lines += 1
                        if len(sample_records) < sample_count:
                            try:
                                line_str = line.decode("utf-8").strip()
                                if line_str:
                                    record = json.loads(line_str)
                                    sample_records.append(record)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                        if line_num > 10000 and len(sample_records) >= sample_count:
                            avg_line_length = mmapped_file.tell() / (line_num + 1)
                            total_lines = int(file_size_bytes / avg_line_length)
                            break
        except Exception as e:
            print(f"Warning: Error reading large JSONL file for sampling: {e}")
            total_lines = "Error during estimation"
        gc.collect()
        return sample_records, total_lines

    def _read_jsonl_samples_direct(
        self, file_path: Path, sample_count: int
    ) -> Tuple[List[Dict], int]:
        """Reads sample records and counts total lines from a small JSONL file directly.

        Args:
            file_path (Path): The path to the JSONL file.
            sample_count (int): The number of records to sample.

        Returns:
            Tuple[List[Dict], int]: A tuple containing a list of sample records
                                    and the exact total number of lines.
        """
        sample_records = []
        total_lines = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < sample_count:
                    try:
                        record = json.loads(line.strip())
                        sample_records.append(record)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                total_lines += 1
        return sample_records, total_lines

    def _analyze_dict_elements_in_list(self, value: list) -> Dict[str, Any]:
        """Helper to analyze list elements when they are all dictionaries."""
        all_keys = set()
        for item in value[:5]:
            if isinstance(item, dict):
                all_keys.update(item.keys())

        element_field_details = {}
        for key in list(all_keys)[:10]:
            key_values = []
            for item in value[:3]:
                if isinstance(item, dict) and key in item:
                    key_values.append(item[key])

            if key_values:
                element_field_details[key] = self._analyze_value_structure(
                    key_values[0]
                )
                if len(key_values) > 1:
                    element_field_details[key]["sample_values"] = [
                        self._get_display_value(v) for v in key_values[:3]
                    ]

        return {
            "type": "list",
            "length": len(value),
            "element_type": "dict",
            "element_field_details": element_field_details,
            "sample_value_summary": f"Array of {len(value)} dict objects with {len(all_keys)} unique fields",
        }

    def _analyze_mixed_elements_in_list(
        self, value: list, element_types: set, sample_elements_info: list
    ) -> Dict[str, Any]:
        """Helper to analyze list elements when they are of mixed or non-dict types."""
        element_type_summary = (
            list(element_types)[0] if len(element_types) == 1 else "mixed"
        )
        return {
            "type": "list",
            "length": len(value),
            "element_type": element_type_summary,
            "sample_elements_preview": [
                elem["sample_value"] for elem in sample_elements_info[:3]
            ],
            "sample_value_summary": f"Array of {len(value)} {element_type_summary} elements",
        }

    def _analyze_list_structure(self, value: list) -> Dict[str, Any]:
        """Helper to analyze the structure of a list."""
        if not value:
            return {"type": "list", "length": 0, "sample_value_summary": "Empty array"}

        element_types = set()
        sample_elements_info = []
        for item in value[:5]:
            element_analysis = self._analyze_value_structure(item)
            element_types.add(element_analysis["type"])
            sample_elements_info.append(element_analysis)

        if len(element_types) == 1 and "dict" in element_types:
            return self._analyze_dict_elements_in_list(value)

        return self._analyze_mixed_elements_in_list(
            value, element_types, sample_elements_info
        )

    def _analyze_dict_structure(self, value: dict) -> Dict[str, Any]:
        """Helper to analyze the structure of a dictionary."""
        if not value:
            return {"type": "dict", "keys": [], "sample_value_summary": "Empty object"}

        key_type_details = {}
        for key, val in list(value.items())[:10]:
            key_type_details[key] = self._analyze_value_structure(val)

        return {
            "type": "dict",
            "keys_present_in_sample": list(value.keys())[:10],
            "total_keys": len(value),
            "key_type_details": key_type_details,
            "sample_value_summary": f"Object with {len(value)} keys: {', '.join(list(value.keys())[:5])}{'...' if len(value) > 5 else ''}",
        }

    def _analyze_value_structure(self, value: Any) -> Dict[str, Any]:
        """Analyzes the structure of a given value, recursively for complex types.

        Args:
            value (Any): The value to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the value.
        """
        if value is None:
            return {"type": "null", "sample_value": None}
        if isinstance(value, str):
            try:
                parsed_value = json.loads(value)
                return self._analyze_value_structure(parsed_value)
            except (json.JSONDecodeError, TypeError):
                return {
                    "type": "str",
                    "sample_value": value[:100] + "..." if len(value) > 100 else value,
                }
        if isinstance(value, list):
            return self._analyze_list_structure(value)
        if isinstance(value, dict):
            return self._analyze_dict_structure(value)
        return {
            "type": type(value).__name__,
            "sample_value": str(value),
        }

    def _get_display_value(self, value: Any) -> str:
        """Generates a display-friendly string representation of a value."""
        if isinstance(value, str) and len(value) > 50:
            return value[:50] + "..."
        if isinstance(value, (list, dict)):
            return str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        return str(value)

    def _analyze_jsonl_record_fields(
        self, sample_records: List[Dict]
    ) -> Dict[str, Any]:
        """Analyzes the field structure from a list of sample JSONL records."""
        field_details = {}
        if sample_records and isinstance(sample_records[0], dict):
            first_record = sample_records[0]
            for key in first_record.keys():
                values = [
                    record.get(key)
                    for record in sample_records
                    if isinstance(record, dict) and key in record
                ]
                if values:
                    field_info = self._analyze_value_structure(values[0])
                    if len(values) > 1:
                        field_info["sample_values"] = [
                            self._get_display_value(v) for v in values[:3]
                        ]
                    field_details[key] = field_info
        return field_details

    def analyze_jsonl_file(
        self, file_path: Path, sample_record_count: int = 5
    ) -> Dict[str, Any]:
        """Analyzes the format of a JSONL file.

        Reads the first few lines to determine record structure, fields, and sample values.
        Uses streaming approach for large files to avoid memory issues.

        Args:
            file_path (Path): The path to the JSONL file.
            sample_record_count (int): The number of records to sample for structure analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the JSONL file.
                            Includes file type, size, record count, structure, field details,
                            and sample records. Returns an 'error' key if analysis fails.
        """
        try:
            file_size_bytes = file_path.stat().st_size
            megabyte_threshold = 100 * 1024 * 1024

            if file_size_bytes > megabyte_threshold:
                return self._stream_analyze_large_jsonl(file_path, sample_record_count)

            if file_size_bytes > 50 * 1024 * 1024:
                sample_records, total_records = self._read_jsonl_samples_from_mmap(
                    file_path, sample_record_count, file_size_bytes
                )
            else:
                sample_records, total_records = self._read_jsonl_samples_direct(
                    file_path, sample_record_count
                )

            field_details = self._analyze_jsonl_record_fields(sample_records)

            if len(sample_records) > 3:
                sample_records = sample_records[:3]
            gc.collect()

            return {
                "file_type": "JSONL",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_records": total_records,
                "record_structure_type": (
                    "dict"
                    if sample_records and isinstance(sample_records[0], dict)
                    else "other"
                ),
                "field_details": field_details,
                "sample_records": sample_records,
            }
        except Exception as e:
            return {"error": f"Failed to analyze JSONL: {str(e)}"}

    def _analyze_json_array_elements(self, elements: List[Any]) -> Dict[str, Any]:
        """Analyzes the structure of elements within a JSON array."""
        element_details = {}
        if elements and isinstance(elements[0], dict):
            for key in elements[0].keys():
                values = [
                    obj.get(key)
                    for obj in elements
                    if isinstance(obj, dict) and key in obj
                ]
                if values:
                    field_info = self._analyze_value_structure(values[0])
                    if len(values) > 1:
                        field_info["sample_values"] = [
                            self._get_display_value(v) for v in values[:3]
                        ]
                    element_details[key] = field_info
        return element_details

    def analyze_json_file(
        self, file_path: Path, sample_key_count: int = 5
    ) -> Dict[str, Any]:
        """Analyzes the format of a JSON file.

        Determines if the JSON contains an array or object at the root level,
        then analyzes structure accordingly. Uses streaming for large files.

        Args:
            file_path (Path): The path to the JSON file.
            sample_key_count (int): The number of records/keys to sample for structure analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the JSON file.
                            Includes file type, size, structure details, and sample data.
                            Returns an 'error' key if analysis fails.
        """
        try:
            file_size_bytes = file_path.stat().st_size
            megabyte_threshold = 100 * 1024 * 1024

            if file_size_bytes > megabyte_threshold:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_char = f.read(1).strip()
                    if first_char == "[":
                        return self._stream_analyze_large_json_array(
                            file_path, sample_key_count
                        )
                    if first_char == "{":
                        return self._stream_analyze_large_json_object(
                            file_path, sample_key_count
                        )
                    return {
                        "file_type": "JSON (Unknown Format)",
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "error": "File does not start with '[' or '{'",
                    }

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    return {
                        "file_type": "JSON Array",
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "total_records": len(data),
                        "record_structure_type": (
                            "dict" if data and isinstance(data[0], dict) else "other"
                        ),
                        "element_details": self._analyze_json_array_elements(
                            data[:sample_key_count]
                        ),
                        "sample_records": data[:3],
                    }
                elif isinstance(data, dict):
                    top_level_keys = list(data.keys())
                    sampled_structure_details = {
                        key: self._analyze_value_structure(data[key])
                        for key in top_level_keys[:sample_key_count]
                    }
                    return {
                        "file_type": "JSON Object",
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "total_top_level_keys_detected": len(top_level_keys),
                        "top_level_keys_in_sample": top_level_keys[:sample_key_count],
                        "sampled_structure_details": sampled_structure_details,
                    }
                else:
                    return {
                        "file_type": "JSON (Unsupported Root Type)",
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "error": f"Root element is of type {type(data).__name__}, expected list or dict.",
                    }
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to analyze JSON: {str(e)}"}

    def analyze_split_file(
        self, file_path: Path, sample_record_count: int = 5
    ) -> Dict[str, Any]:
        """Analyzes a potential split file (e.g., 'part_aa', 'part_ab').

        Attempts to determine if the split file contains JSONL data by
        reading its first line.

        Args:
            file_path (Path): The path to the split file.
            sample_record_count (int): The number of records to sample if identified as JSONL.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary for the split file.
                            Identifies if it's a JSONL split part or an unknown binary file.
                            Returns an 'error' key if analysis fails.
        """
        try:
            file_size_bytes = file_path.stat().st_size

            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        json.loads(first_line)
                        result = self.analyze_jsonl_file(file_path, sample_record_count)
                        result["file_type"] = "JSONL Split Part"
                        result["note"] = "This is a split part of a larger JSONL file."
                        return result
                    except json.JSONDecodeError:
                        f.seek(0)
                        chunk = f.read(1000)
                        if "{" in chunk and '"' in chunk:
                            return {
                                "file_type": "JSONL Split Part (Partial)",
                                "file_size_mb": round(
                                    file_size_bytes / (1024 * 1024), 2
                                ),
                                "note": "Split part of JSONL file - may not start with a complete JSON record.",
                            }

            return {
                "file_type": "Split File (Binary/Unknown)",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "note": "This appears to be a split file part. May need to be reassembled for analysis.",
            }
        except Exception as e:
            return {"error": f"Failed to analyze split file: {str(e)}"}

    def _identify_split_and_reassembled_files(
        self, files_in_dir: List[Path]
    ) -> Tuple[List[Path], List[Path]]:
        """Identifies split files (e.g., part_aa) and their reassembled counterparts."""
        split_files = [f for f in files_in_dir if f.is_file() and "part_" in f.name]
        reassembled_files_detected: List[Path] = []

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
                    reassembled_files_detected.append(f)
        return split_files, reassembled_files_detected

    def _analyze_file_by_extension(self, file_path: Path) -> Dict[str, Any]:
        """Analyzes a single file based on its extension and type."""
        file_extension = file_path.suffix.lower()
        print(f"Analyzing {file_path.name}...")

        if file_extension == ".csv":
            return self.analyze_csv_file(file_path)
        if file_extension == ".xlsx":
            return self.analyze_excel_file(file_path)
        if file_extension == ".jsonl":
            return self.analyze_jsonl_file(file_path)
        if file_extension == ".json":
            return self.analyze_json_file(file_path)
        if "part_" in file_path.name:
            return self.analyze_split_file(file_path)
        return {
            "file_type": f"Unknown ({file_extension})",
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        }

    def _handle_skipped_reassembled_file(self, file_path: Path) -> Dict[str, Any]:
        """Generates analysis info for reassembled files that are skipped."""
        print(f"Skipping {file_path.name} (split parts available for analysis)")
        return {
            "file_type": "Reassembled File (Skipped)",
            "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            "note": "Skipped analysis - split parts analyzed instead to avoid redundancy.",
        }

    def analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyzes all relevant files within a single dataset directory.

        Identifies and analyzes CSV, Excel, JSONL, JSON, and split files.
        It prioritizes analyzing split file parts over their reassembled counterparts
        to avoid redundant analysis.

        Args:
            dataset_path (Path): The path to the dataset directory.

        Returns:
            Dict[str, Any]: A dictionary containing the summary for the dataset,
                            including details for each analyzed file.
        """
        dataset_analysis: Dict[str, Any] = {
            "dataset_name": dataset_path.name,
            "files": {},
        }

        files_in_dir = list(dataset_path.iterdir())
        split_files, reassembled_files = self._identify_split_and_reassembled_files(
            files_in_dir
        )

        for file_path in files_in_dir:
            if file_path.is_file() and not file_path.name.startswith("."):
                if file_path in reassembled_files:
                    analysis_result = self._handle_skipped_reassembled_file(file_path)
                else:
                    analysis_result = self._analyze_file_by_extension(file_path)

                dataset_analysis["files"][file_path.name] = analysis_result

        return dataset_analysis

    def generate_summary(self) -> Dict[str, Any]:
        """Generates a data format summary for all datasets found in the data directory.

        Iterates through subdirectories starting with 'dataset_' and analyzes each.

        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive summary
                            of all analyzed datasets and their files.
        """
        print("Starting data format analysis...")
        for item in self.data_directory.iterdir():
            if item.is_dir() and item.name.startswith("dataset_"):
                print(f"\nAnalyzing {item.name}...")
                self.analysis_summary[item.name] = self.analyze_dataset(item)
        return self.analysis_summary

    def save_summary(
        self,
        output_file_path: Union[
            str, Path
        ] = "src/data/_analysis/data_format_summary.json",
    ):
        """Saves the generated data format summary to a JSON file.

        Args:
            output_file_path (Union[str, Path]): The path to the output JSON file.
        """
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_summary, f, indent=2, default=str)
        print(f"\nSummary saved to {output_path}")

    def _print_file_basic_info(self, file_name: str, file_info: Dict[str, Any]) -> None:
        """Prints basic file information (name, type, size)."""
        print(f"\n📄 {file_name}")
        print(f"   Type: {file_info.get('file_type', 'Unknown')}")
        print(f"   Size: {file_info.get('file_size_mb', 'Unknown')} MB")

    def _print_file_counts(self, file_info: Dict[str, Any]) -> None:
        """Prints file row/record and column/field counts."""
        if "total_rows" in file_info:
            print(f"   Rows: {file_info['total_rows']:,}")
        elif "total_records" in file_info:
            print(f"   Records: {file_info['total_records']}")

        if "column_count" in file_info:
            print(f"   Columns: {file_info['column_count']}")
        elif "field_details" in file_info and file_info["field_details"]:
            print(f"   Fields: {len(file_info['field_details'])}")

    def _print_file_structure(self, file_info: Dict[str, Any]) -> None:
        """Prints file structure information (columns/fields with types)."""
        if "column_details" in file_info:
            print("   Column Info (Sample):")
            for col, col_info in list(file_info["column_details"].items())[:5]:
                print(f"     • {col}: {col_info['data_type']}")
        elif "field_details" in file_info and file_info["field_details"]:
            print("   Field Info (Sample):")
            for field, field_info in list(file_info["field_details"].items())[:5]:
                field_type = field_info.get("type", "unknown")
                print(f"     • {field}: {field_type}")

        elif (
            "top_level_keys_in_sample" in file_info
            and file_info["top_level_keys_in_sample"]
        ):
            keys_display = ", ".join(
                f'"{k}"' for k in file_info["top_level_keys_in_sample"][:5]
            )
            print(f"   Top-Level Keys: {keys_display}")
            if file_info.get("total_top_level_keys_detected", 0) > 5:
                print(f"   Total Keys: {file_info['total_top_level_keys_detected']}")

    def _print_file_notes_and_errors(self, file_info: Dict[str, Any]) -> None:
        """Prints file error and note information."""
        if "error" in file_info:
            print(f"   ⚠️ Error: {file_info['error']}")
        if "note" in file_info:
            print(f"   ℹ️ Note: {file_info['note']}")

    def print_summary(self) -> None:
        """Prints a formatted summary of the data formats to the console.

        Displays dataset names, file names, types, sizes, row/record counts,
        and sampled column/field information.
        """
        print("\n" + "=" * 80)
        print("DATA FORMAT SUMMARY")
        print("=" * 80)

        for dataset_name, dataset_info in self.analysis_summary.items():
            print(f"\n📁 {dataset_name.upper()}")
            print("-" * 50)

            for file_name, file_info in dataset_info["files"].items():
                self._print_file_basic_info(file_name, file_info)
                self._print_file_counts(file_info)
                self._print_file_structure(file_info)
                self._print_file_notes_and_errors(file_info)

    def _stream_analyze_large_jsonl(
        self, file_path: Path, sample_count: int = 5
    ) -> Dict[str, Any]:
        """Memory-efficient analysis of large JSONL files using streaming."""
        try:
            file_size_bytes = file_path.stat().st_size
            sample_records, total_records = self._read_jsonl_samples_from_mmap(
                file_path, sample_count, file_size_bytes
            )

            field_details = self._analyze_jsonl_record_fields(sample_records)

            return {
                "file_type": "JSONL (Large File - Streamed)",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_records": total_records,
                "record_structure_type": (
                    "dict"
                    if sample_records and isinstance(sample_records[0], dict)
                    else "other"
                ),
                "field_details": field_details,
                "sample_records": sample_records[:3],
                "note": "Analyzed using streaming approach for memory efficiency",
            }
        except Exception as e:
            return {"error": f"Failed to stream analyze JSONL: {str(e)}"}

    def _stream_analyze_large_json_array(
        self, file_path: Path, sample_count: int = 5
    ) -> Dict[str, Any]:
        """Memory-efficient analysis of large JSON array files using ijson streaming."""
        try:
            file_size_bytes = file_path.stat().st_size
            sample_elements_from_stream = []
            total_elements_counted = 0

            with open(file_path, "rb") as f:
                parser = ijson.items(f, "item")
                for item in parser:
                    total_elements_counted += 1
                    if len(sample_elements_from_stream) < sample_count:
                        sample_elements_from_stream.append(item)
                    if (
                        total_elements_counted > 50000
                        and len(sample_elements_from_stream) >= sample_count
                    ):
                        total_elements_counted = "50,000+ (estimated from large file)"
                        break
            gc.collect()

            if sample_elements_from_stream:
                element_details = self._analyze_json_array_elements(
                    sample_elements_from_stream
                )
                return {
                    "file_type": "JSON Array (Large File - Streamed)",
                    "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                    "estimated_record_count": total_elements_counted,
                    "record_structure_type": (
                        "dict"
                        if isinstance(sample_elements_from_stream[0], dict)
                        else type(sample_elements_from_stream[0]).__name__
                    ),
                    "element_details": element_details,
                    "sample_records": sample_elements_from_stream[:3],
                    "note": "Analyzed using streaming approach for memory efficiency",
                }
            return {
                "file_type": "JSON Array (Large File - Empty)",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "note": "No valid items found in array",
            }
        except Exception as e:
            return {"error": f"Failed to stream analyze JSON array: {str(e)}"}

    def _get_json_object_sample_item(self, file_path: Path, key: str) -> Any:
        """Helper to retrieve a complete sample object/value for a given key from a large JSON object."""
        try:
            with open(file_path, "rb") as f:
                items_generator = ijson.items(f, key)
                for item_value in items_generator:
                    return item_value
        except Exception as e:
            return f"Error extracting object: {str(e)}"
        return None  # In case the key is not found or yields no items

    def _stream_analyze_large_json_object(
        self, file_path: Path, sample_key_count: int = 5
    ) -> Dict[str, Any]:
        """Memory-efficient analysis of large JSON object files using ijson streaming.

        This method aims to retrieve complete sample objects/values for a specified
        number of top-level keys to provide a more detailed overview.

        Args:
            file_path (Path): The path to the JSON object file.
            sample_key_count (int): The number of top-level keys for which to
                                    attempt to fetch complete sample objects/values.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis summary, including
                            complete sample objects/values for selected top-level keys.
        """
        try:
            file_size_bytes = file_path.stat().st_size
            final_structure_details = {}
            total_keys_detected = 0

            top_level_keys_to_consider = []
            with open(file_path, "rb") as f:
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    if event == "map_key" and prefix == "":
                        top_level_keys_to_consider.append(value)
                        total_keys_detected += 1
                        if len(top_level_keys_to_consider) >= sample_key_count:
                            break  # We only need to identify up to sample_key_count keys initially

            keys_for_deep_sampling = top_level_keys_to_consider[:sample_key_count]

            for key in keys_for_deep_sampling:
                item_value = self._get_json_object_sample_item(file_path, key)
                if (
                    item_value is not None
                    and not isinstance(item_value, str)
                    and "Error extracting object" not in str(item_value)
                ):
                    if isinstance(item_value, dict):
                        final_structure_details[key] = {
                            "type": "dict",
                            "complete_sample_object": item_value,
                            "keys_present_in_sample": list(item_value.keys()),
                            "total_keys": len(item_value.keys()),
                            "sample_value_summary": f"Complete object with {len(item_value.keys())} keys",
                        }
                    elif isinstance(item_value, list):
                        final_structure_details[key] = {
                            "type": "list",
                            "complete_sample_array": item_value,
                            "length": len(item_value),
                            "sample_value_summary": f"Complete array with {len(item_value)} elements",
                        }
                    else:
                        final_structure_details[key] = {
                            "type": type(item_value).__name__,
                            "complete_sample_value": item_value,
                            "sample_value": str(item_value),
                        }
                else:
                    final_structure_details[key] = {
                        "type": "error",
                        "sample_value": (
                            item_value
                            if item_value
                            else "Key not found or extraction failed"
                        ),
                    }

            gc.collect()

            return {
                "file_type": "JSON Object (Large File - Streamed with Full Samples)",
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "total_top_level_keys_detected": total_keys_detected,
                "sampled_structure_details": final_structure_details,
                "top_level_keys_in_sample": list(final_structure_details.keys()),
                "note": f"Analyzed using streaming for memory efficiency; complete samples retrieved for the first {len(keys_for_deep_sampling)} top-level keys.",
            }
        except Exception as e:
            return {"error": f"Failed to stream analyze JSON object: {str(e)}"}


def main():
    """Main function to execute the data format analysis."""
    analyzer = DataFormatAnalyzer()
    try:
        analyzer.generate_summary()
        analyzer.print_summary()
        analyzer.save_summary()
        print("\n✅ Analysis complete! Summary saved to data_format_summary.json")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
