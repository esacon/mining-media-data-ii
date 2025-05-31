import json
from pathlib import Path
from typing import Dict, Optional, Union

from src.config import Settings
from src.utils import (
    LoggerMixin,
    _get_player_id_from_record,
    calculate_dataset_stats,
    convert_timestamp,
    ensure_dir,
    get_time_boundaries,
    jsonl_iterator,
)


class DatasetCreator(LoggerMixin):
    """Creates labeled datasets for churn prediction."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = settings.processed_dir
        self.output_dir = ensure_dir(settings.processed_dir)
        self.observation_days = settings.observation_days
        self.churn_period_days = settings.churn_period_days

    def _get_path(self, file_path: Union[str, Path]) -> Path:
        """Get full path for files."""
        return (
            self.data_dir / file_path
            if not Path(file_path).is_absolute()
            else Path(file_path)
        )

    def process_player_records(self, player_data: Dict) -> Optional[Dict]:
        """Processes a single player's raw event records to create a labeled data point.

        Args:
            player_data (Dict): A dictionary containing the player's data.

        Returns:
            Optional[Dict]: A dictionary containing the processed player data.
        """
        records = player_data.get("records", [])
        if not records:
            return None

        records.sort(key=lambda x: x["time"])
        first_time = records[0]["time"]
        boundaries = get_time_boundaries(
            first_time, self.observation_days, self.churn_period_days
        )

        observation_records = []
        churn_period_records = []

        for record in records:
            record_time = record["time"]
            record_dt = convert_timestamp(record_time, "datetime")

            if record_dt <= boundaries["op_end"]:
                observation_records.append(record)
            elif boundaries["cp_start"] < record_dt <= boundaries["cp_end"]:
                churn_period_records.append(record)

        if not observation_records:
            return None

        return {
            "player_id": _get_player_id_from_record(player_data, None),
            "observation_records": observation_records,
            "churn_period_records": churn_period_records,
            "churned": len(churn_period_records) == 0,
            "op_start": boundaries["op_start_iso"],
            "op_end": boundaries["op_end_iso"],
            "cp_start": boundaries["cp_start_iso"],
            "cp_end": boundaries["cp_end_iso"],
            "op_event_count": len(observation_records),
            "cp_event_count": len(churn_period_records),
        }

    def create_dataset(self, input_file: Union[str, Path], output_prefix: str) -> Path:
        """Creates a labeled dataset from an input JSONL file.

        Args:
            input_file (Union[str, Path]): The path to the input JSONL file.
            output_prefix (str): The prefix for the output file name.

        Returns:
            Path: The path to the output file.

        Raises:
            FileNotFoundError: If the input file is not found.
            Exception: If an error occurs during dataset creation.
        """
        self.logger.info(f"Creating dataset from {input_file}...")

        input_path = self._get_path(input_file)
        output_file_name = f"{output_prefix}{self.settings.labeled_suffix}"
        output_path = self.output_dir / output_file_name

        processed_count = 0
        skipped_count = 0

        with open(output_path, "w", encoding="utf-8") as f_out:
            for line_num, player_data in enumerate(jsonl_iterator(input_path)):
                if (line_num + 1) % 1000 == 0:
                    self.logger.info(f"Processed {line_num + 1} players...")

                try:
                    processed_data = self.process_player_records(player_data)
                    if processed_data:
                        f_out.write(
                            json.dumps(processed_data, ensure_ascii=False) + "\n"
                        )
                        processed_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    self.logger.warning(f"Error processing line {line_num + 1}: {e}")
                    skipped_count += 1

        self.logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}")
        return output_path

    def create_all_datasets(self) -> Dict[str, Dict[str, str]]:
        """Creates all necessary labeled datasets for both games.

        Args:
            None

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing the paths to the created datasets.
        """
        self.logger.info("Creating all datasets...")

        datasets_config = {
            "game1": {
                "train": (self.settings.game1_train, "game1_DS1"),
                "eval": (self.settings.game1_eval, "game1_DS2"),
            },
            "game2": {
                "train": (self.settings.game2_train, "game2_DS1"),
                "eval": (self.settings.game2_eval, "game2_DS2"),
            },
        }

        results = {}
        for game, files in datasets_config.items():
            results[game] = {}
            for dataset_type, (input_file, output_prefix) in files.items():
                dataset_path = self.create_dataset(input_file, output_prefix)
                results[game][
                    "DS1" if dataset_type == "train" else "DS2"
                ] = dataset_path.name

        self.logger.info("All datasets created successfully!")
        return results

    def get_dataset_summary(self, dataset_file: Union[str, Path]) -> Dict:
        """Gets summary statistics for a labeled dataset.

        Args:
            dataset_file (Union[str, Path]): The path to the labeled dataset file.

        Returns:
            Dict: A dictionary containing the summary statistics.
        """
        dataset_path = self._get_path(dataset_file)
        if not dataset_path.is_absolute():
            dataset_path = self.output_dir / dataset_file
        return calculate_dataset_stats(dataset_path)
