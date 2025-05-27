import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Iterator
import numpy as np


def load_jsonl_sample(file_path: Union[str, Path], n_samples: int = 5) -> List[Dict]:
    """Loads a sample of records from a JSONL file.

    Args:
        file_path (Union[str, Path]): Path to the JSONL file.
        n_samples (int): Number of samples to load. Defaults to 5.

    Returns:
        List[Dict]: A list of JSON objects representing the sampled records.
    """
    samples = []
    path = Path(file_path)

    if not path.exists():
        return samples

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    return samples


def jsonl_iterator(file_path: Union[str, Path]) -> Iterator[Dict]:
    """Iterates over a JSONL file, yielding one record at a time.

    Args:
        file_path (Union[str, Path]): Path to the JSONL file.

    Yields:
        Dict: A JSON object from the file.
    """
    path = Path(file_path)

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num + 1}: {e}")
                continue


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Saves data as a JSON file.

    Args:
        data (Any): The data to save.
        file_path (Union[str, Path]): The output file path.
        indent (int): The JSON indentation level. Defaults to 2.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Any:
    """Loads data from a JSON file.

    Args:
        file_path (Union[str, Path]): Path to the JSON file.

    Returns:
        Any: The loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], file_path: Union[str, Path]) -> None:
    """Saves a list of dictionaries as a JSONL file.

    Args:
        data (List[Dict]): The list of dictionaries to save.
        file_path (Union[str, Path]): The output file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def _get_player_id_from_record(record: Dict, id_field: Optional[str]) -> Optional[str]:
    """Helper function to extract player ID from a record.

    Args:
        record (Dict): A single JSON record.
        id_field (Optional[str]): The field name for the player ID. If None,
                                   it attempts to auto-detect from common fields.

    Returns:
        Optional[str]: The player ID as a string, or None if not found.
    """
    if id_field:
        return str(record.get(id_field)) if record.get(id_field) is not None else None
    else:
        return str(record.get("device_id") or record.get("uid") or record.get("player_id")) if \
            (record.get("device_id") or record.get("uid") or record.get("player_id")) is not None else None


def get_player_ids(file_path: Union[str, Path], id_field: Optional[str] = None) -> List[str]:
    """Extracts player IDs from a JSONL file.

    Args:
        file_path (Union[str, Path]): Path to the JSONL file.
        id_field (Optional[str]): The field name for the player ID. If None,
                                   it attempts to auto-detect from common fields
                                   like "device_id", "uid", or "player_id".

    Returns:
        List[str]: A list of extracted player IDs.
    """
    player_ids = []

    for record in jsonl_iterator(file_path):
        player_id = _get_player_id_from_record(record, id_field)
        if player_id:
            player_ids.append(player_id)

    return player_ids


def split_jsonl_by_ids(
    input_file: Union[str, Path],
    train_ids: set,
    eval_ids: set,
    train_output: Union[str, Path],
    eval_output: Union[str, Path],
    id_field: Optional[str] = None
) -> Dict[str, int]:
    """Splits a JSONL file into training and evaluation datasets based on player IDs.

    Args:
        input_file (Union[str, Path]): The path to the input JSONL file.
        train_ids (set): A set of player IDs designated for the training set.
        eval_ids (set): A set of player IDs designated for the evaluation set.
        train_output (Union[str, Path]): The path for the training output JSONL file.
        eval_output (Union[str, Path]): The path for the evaluation output JSONL file.
        id_field (Optional[str]): The field name for the player ID. If None,
                                   it attempts to auto-detect from common fields.

    Returns:
        Dict[str, int]: A dictionary containing counts of records in 'train_count',
                        'eval_count', and 'skipped_count'.
    """
    train_count = 0
    eval_count = 0
    skipped_count = 0

    train_path = Path(train_output)
    eval_path = Path(eval_output)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f_train, \
            open(eval_path, 'w', encoding='utf-8') as f_eval:
        for record in jsonl_iterator(input_file):
            player_id = _get_player_id_from_record(record, id_field)

            if not player_id:
                skipped_count += 1
                continue

            if player_id in train_ids:
                f_train.write(json.dumps(record, ensure_ascii=False) + '\n')
                train_count += 1
            elif player_id in eval_ids:
                f_eval.write(json.dumps(record, ensure_ascii=False) + '\n')
                eval_count += 1
            else:
                skipped_count += 1

    return {
        'train_count': train_count,
        'eval_count': eval_count,
        'skipped_count': skipped_count
    }


def calculate_dataset_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Calculates basic statistics for a labeled dataset in JSONL format.

    The statistics include total players, churned/retained counts and rates,
    and descriptive statistics (mean, min, max, std) for 'op_event_count'
    and 'cp_event_count'.

    Args:
        file_path (Union[str, Path]): Path to the labeled dataset file.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated statistics.
    """
    total_count = 0
    churned_count = 0
    op_event_counts = []
    cp_event_counts = []

    for record in jsonl_iterator(file_path):
        total_count += 1

        if record.get("churned", False):
            churned_count += 1

        op_count = record.get("op_event_count", 0)
        cp_count = record.get("cp_event_count", 0)

        op_event_counts.append(op_count)
        cp_event_counts.append(cp_count)

    if total_count == 0:
        return {
            'total_players': 0,
            'churned_players': 0,
            'retained_players': 0,
            'churn_rate': 0.0,
            'retention_rate': 0.0,
            'op_events': {'mean': 0, 'min': 0, 'max': 0, 'std': 0},
            'cp_events': {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        }

    retained_count = total_count - churned_count
    churn_rate = churned_count / total_count
    retention_rate = retained_count / total_count

    return {
        'total_players': total_count,
        'churned_players': churned_count,
        'retained_players': retained_count,
        'churn_rate': churn_rate,
        'retention_rate': retention_rate,
        'op_events': {
            'mean': np.mean(op_event_counts) if op_event_counts else 0,
            'min': min(op_event_counts) if op_event_counts else 0,
            'max': max(op_event_counts) if op_event_counts else 0,
            'std': np.std(op_event_counts) if op_event_counts else 0
        },
        'cp_events': {
            'mean': np.mean(cp_event_counts) if cp_event_counts else 0,
            'min': min(cp_event_counts) if cp_event_counts else 0,
            'max': max(cp_event_counts) if cp_event_counts else 0,
            'std': np.std(cp_event_counts) if cp_event_counts else 0
        }
    }
