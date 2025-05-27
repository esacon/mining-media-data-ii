from pathlib import Path
from typing import Any, Dict, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensures that a directory exists, creating it if it doesn't.

    Args:
        path (Union[str, Path]): The directory path to check or create.

    Returns:
        Path: A Path object representing the ensured directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path], unit: str = "MB") -> float:
    """Gets the size of a file in a specified unit.

    Args:
        file_path (Union[str, Path]): Path to the file.
        unit (str): The desired unit for the file size ('B', 'KB', 'MB', 'GB').
                    Defaults to 'MB'.

    Returns:
        float: The file size in the specified unit. Returns 0.0 if the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0

    size_bytes = path.stat().st_size

    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}

    return size_bytes / units.get(unit.upper(), 1)


def count_lines(file_path: Union[str, Path]) -> int:
    """Counts the number of lines in a text file efficiently.

    Args:
        file_path (Union[str, Path]): Path to the file.

    Returns:
        int: The number of lines in the file. Returns 0 if the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Gets comprehensive information about a file.

    Includes existence, size, line count (for text files), extension, name, and parent directory.

    Args:
        file_path (Union[str, Path]): Path to the file.

    Returns:
        Dict[str, Any]: A dictionary containing file information.
                        'lines' will be None for non-text file extensions.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "exists": False,
            "size_mb": 0,
            "lines": 0,
            "extension": None,
            "name": path.name,
            "parent": str(path.parent),
        }

    return {
        "exists": True,
        "size_mb": get_file_size(path, "MB"),
        "lines": (
            count_lines(path)
            if path.suffix in [".txt", ".csv", ".jsonl", ".json"]
            else None
        ),
        "extension": path.suffix,
        "name": path.name,
        "parent": str(path.parent),
    }


def safe_filename(filename: str) -> str:
    """Creates a safe filename by replacing problematic characters with underscores.

    Args:
        filename (str): The original filename.

    Returns:
        str: The safe filename.
    """
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(c if c in safe_chars else "_" for c in filename)
