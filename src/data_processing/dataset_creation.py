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
    """Creates labeled datasets for churn prediction by defining observation and churn periods
    based on player event data.
    """

    def __init__(self, settings: Settings):
        """Initializes the DatasetCreator.

        Args:
            settings (Settings): Configuration settings containing paths, filenames, and parameters.
        """
        self.settings = settings
        self.data_dir = settings.processed_dir
        self.output_dir = ensure_dir(settings.processed_dir)
        self.observation_days = settings.observation_days
        self.churn_period_days = settings.churn_period_days

    def process_player_records(self, player_data: Dict) -> Optional[Dict]:
        """Processes a single player's raw event records to create a labeled data point.

        It identifies observation and churn periods based on the player's first event,
        assigns a churn label, and counts events within each period.

        Args:
            player_data (Dict): A dictionary containing a player's raw data, expected to have
                                "records" (list of event dicts) and an ID field like "device_id" or "uid".

        Returns:
            Optional[Dict]: A dictionary containing the processed and labeled data for the player,
                            including observation records, churn period records, churn label,
                            time boundaries, and event counts. Returns None if the player has no records.
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

        churned = len(churn_period_records) == 0

        player_id = _get_player_id_from_record(player_data, None)

        return {
            "player_id": player_id,
            "observation_records": observation_records,
            "churn_period_records": churn_period_records,
            "churned": churned,
            "op_start": boundaries["op_start_iso"],
            "op_end": boundaries["op_end_iso"],
            "cp_start": boundaries["cp_start_iso"],
            "cp_end": boundaries["cp_end_iso"],
            "op_event_count": len(observation_records),
            "cp_event_count": len(churn_period_records),
        }

    def create_dataset(self, input_file: Union[str, Path], output_prefix: str) -> Path:
        """Creates a labeled dataset from an input JSONL file containing player records.

        This function iterates through each player's data, processes it using
        `process_player_records`, and writes the labeled output to a new JSONL file.
        It also logs processing progress and basic statistics upon completion.

        Args:
            input_file (Union[str, Path]): The name of the input JSONL file (relative to `self.data_dir`).
            output_prefix (str): A prefix for the output labeled JSONL file and its statistics file.
                                 The output file will be named "{output_prefix}{labeled_suffix}".

        Returns:
            Path: The full path to the generated labeled JSONL file.
        """
        self.logger.info(f"Creating dataset from {input_file}...")

        input_path = self.data_dir / input_file
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
                    self.logger.warning(
                        f"Error processing line {line_num + 1} in {input_file}: {e}"
                    )
                    skipped_count += 1

        self.logger.info(
            f"Finished processing {input_file}. Total processed: {processed_count}, Skipped: {skipped_count}"
        )
        return output_path

    def create_all_datasets(self) -> Dict[str, Dict[str, str]]:
        """Creates all necessary labeled datasets for both Game 1 and Game 2.

        This includes creating training (DS1) and evaluation (DS2) datasets for each game.
        The names of the created files are stored and returned in a dictionary.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary mapping game names to a dictionary
                                       of dataset types ("DS1", "DS2") and their corresponding
                                       generated filenames.
        """
        self.logger.info("Creating all datasets (DS1 and DS2 for both games)...")

        datasets_to_process = {
            "game1": {
                "train": self.settings.game1_train,
                "eval": self.settings.game1_eval,
            },
            "game2": {
                "train": self.settings.game2_train,
                "eval": self.settings.game2_eval,
            },
        }

        results = {}

        for game, files in datasets_to_process.items():
            results[game] = {}

            # Create DS1 (training dataset)
            ds1_path = self.create_dataset(files["train"], f"{game}_DS1")
            results[game]["DS1"] = ds1_path.name

            # Create DS2 (evaluation dataset)
            ds2_path = self.create_dataset(files["eval"], f"{game}_DS2")
            results[game]["DS2"] = ds2_path.name

        self.logger.info("All datasets created successfully!")
        return results

    def get_dataset_summary(self, dataset_file: Union[str, Path]) -> Dict:
        """Gets summary statistics for a previously created labeled dataset.

        Args:
            dataset_file (Union[str, Path]): The name or path of the labeled dataset file.
                                            Assumed to be in `self.output_dir` if only a name is given.

        Returns:
            Dict: A dictionary containing comprehensive statistics for the dataset.
        """
        dataset_path = Path(dataset_file)
        if not dataset_path.is_absolute():
            dataset_path = self.output_dir / dataset_file
        return calculate_dataset_stats(dataset_path)
