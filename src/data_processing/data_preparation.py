import polars as pl
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

from src.utils import LoggerMixin, ensure_dir, get_player_ids, split_jsonl_by_ids, save_json


class DataPreparation(LoggerMixin):
    """Handles data preparation tasks for churn prediction, including data conversion
    and splitting datasets into train and evaluation sets.
    """

    def __init__(self, data_dir: Union[str, Path] = "src/data", output_dir: Union[str, Path] = "src/data/processed"):
        """Initializes the DataPreparation class.

        Args:
            data_dir (Union[str, Path]): The directory where raw input data is located.
                                        Defaults to "src/data".
            output_dir (Union[str, Path]): The directory where processed data will be saved.
                                           Defaults to "src/data/processed".
        """
        self.data_dir = Path(data_dir)
        self.output_dir = ensure_dir(output_dir)

    def convert_game1_to_jsonl(
        self,
        input_file: Union[str, Path] = "dataset_1_game1/rawdata_game1.csv",
        output_file: Union[str, Path] = "game1_player_events.jsonl"
    ) -> Path:
        """Converts Game 1 CSV data to JSONL format, with one JSON object per player.

        Each player's data is aggregated, and their events (time, score, event="play")
        are sorted by time.

        Args:
            input_file (Union[str, Path]): The path to the input CSV file relative to `self.data_dir`.
                                           Defaults to "dataset_1_game1/rawdata_game1.csv".
            output_file (Union[str, Path]): The name of the output JSONL file, saved in `self.output_dir`.
                                            Defaults to "game1_player_events.jsonl".

        Returns:
            Path: The full path to the generated JSONL file.

        Raises:
            FileNotFoundError: If the input CSV file does not exist.
        """
        self.logger.info("Converting Game 1 data from CSV to JSONL...")

        input_path = self.data_dir / input_file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pl.read_csv(input_path)

        player_data = df.group_by("device").agg([
            pl.col("time").alias("times"),
            pl.col("score").alias("scores")
        ])

        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in player_data.iter_rows(named=True):
                events = []
                for time, score in zip(row["times"], row["scores"]):
                    events.append({
                        "time": int(time),
                        "score": int(score),
                        "event": "play"
                    })

                events.sort(key=lambda x: x["time"])

                player_json = {
                    "device_id": str(row["device"]),
                    "records": events
                }

                f.write(json.dumps(player_json, ensure_ascii=False) + '\n')

        self.logger.info(
            f"Converted {len(player_data)} players to JSONL format at {output_path}")
        return output_path

    def split_dataset(
        self,
        jsonl_file: Union[str, Path],
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[Path, Path]:
        """Splits a JSONL dataset into training and evaluation sets based on player IDs.

        Args:
            jsonl_file (Union[str, Path]): The name of the input JSONL file. If it starts with "game",
                                           it's assumed to be in `self.output_dir`; otherwise,
                                           it's assumed to be in `self.data_dir`.
            train_ratio (float): The proportion of players to include in the training set.
                                 Defaults to 0.8.
            seed (int): The random seed for shuffling player IDs to ensure reproducibility.
                        Defaults to 42.

        Returns:
            Tuple[Path, Path]: A tuple containing the full paths to the generated
                               training and evaluation JSONL files.

        Raises:
            FileNotFoundError: If the input JSONL file does not exist.
        """
        self.logger.info(f"Splitting {jsonl_file} into train/eval sets...")

        if isinstance(jsonl_file, Path):
            input_path = jsonl_file
        elif str(jsonl_file).startswith("game"):
            input_path = self.output_dir / jsonl_file
        else:
            input_path = self.data_dir / jsonl_file

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input JSONL file not found for splitting: {input_path}")

        player_ids = get_player_ids(input_path)

        random.seed(seed)
        random.shuffle(player_ids)

        split_idx = int(len(player_ids) * train_ratio)
        train_ids = set(player_ids[:split_idx])
        eval_ids = set(player_ids[split_idx:])

        base_name = Path(jsonl_file).stem
        train_file_name = f"{base_name}_train.jsonl"
        eval_file_name = f"{base_name}_eval.jsonl"

        train_path = self.output_dir / train_file_name
        eval_path = self.output_dir / eval_file_name

        # Assuming the original JSONL files for Game 1 and Game 2 use "device_id" or "uid" as player IDs.
        # This parameter would be used if there was a consistent, predefined field.
        # For auto-detection, `id_field=None` is correct based on _get_player_id_from_record logic.
        results = split_jsonl_by_ids(
            input_file=input_path,
            train_ids=train_ids,
            eval_ids=eval_ids,
            train_output=train_path,
            eval_output=eval_path,
            id_field=None  # Retain auto-detection
        )

        self.logger.info(
            f"Split complete: {results['train_count']} train, "
            f"{results['eval_count']} eval players. Skipped: {results['skipped_count']}"
        )

        return train_path, eval_path

    def prepare_all_data(self) -> Dict[str, Dict[str, str]]:
        """Executes the full data preparation pipeline.

        This includes converting Game 1 CSV data to JSONL and then splitting
        both Game 1 and Game 2 data into training and evaluation sets.
        Results (filenames of the split datasets) are saved to a JSON file.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing the names of the
                                       generated train and eval files for each game.
        """
        self.logger.info("Starting data preparation pipeline...")

        game1_jsonl_path = self.convert_game1_to_jsonl()

        game1_train_path, game1_eval_path = self.split_dataset(game1_jsonl_path)

        game2_train_path, game2_eval_path = self.split_dataset(
            self.data_dir / "dataset_2_game2/playerLogs_game2_playerbasedlines.jsonl"
        )

        results = {
            "game1": {
                "train": game1_train_path.name,
                "eval": game1_eval_path.name
            },
            "game2": {
                "train": game2_train_path.name,
                "eval": game2_eval_path.name
            }
        }

        save_json(results, self.output_dir / "preparation_results.json")

        self.logger.info("Data preparation complete!")
        return results
