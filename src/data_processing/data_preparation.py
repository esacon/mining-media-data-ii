import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import polars as pl

from src.config import Settings
from src.utils import (
    LoggerMixin,
    ensure_dir,
    get_player_ids,
    split_jsonl_by_ids,
)


class DataPreparation(LoggerMixin):
    """Handles data preparation tasks for churn prediction, including data conversion
    and splitting datasets into train and evaluation sets.
    """

    def __init__(self, settings: Settings):
        """Initializes the DataPreparation class.

        Args:
            settings (Settings): Configuration settings containing paths and filenames.
        """
        self.settings = settings
        self.data_dir = settings.data_dir
        self.output_dir = ensure_dir(settings.processed_dir)

    def _get_path(self, file_path: Union[str, Path], is_output: bool = False) -> Path:
        """Get full path for input or output files."""
        base_dir = self.output_dir if is_output else self.data_dir
        return base_dir / file_path

    def convert_game1_to_jsonl(
        self, input_file: Union[str, Path] = None, output_file: Union[str, Path] = None
    ) -> Path:
        """Converts Game 1 CSV data to JSONL format, with one JSON object per player.

        Each player's data is aggregated, and their events (time, score, event="play")
        are sorted by time.

        Args:
            input_file (Union[str, Path]): The path to the input CSV file relative to `self.data_dir`.
                                           Defaults to settings.game1_csv.
            output_file (Union[str, Path]): The name of the output JSONL file, saved in `self.output_dir`.
                                            Defaults to settings.game1_converted.

        Returns:
            Path: The full path to the generated JSONL file.

        Raises:
            FileNotFoundError: If the input CSV file does not exist.
        """
        self.logger.info("Converting Game 1 data from CSV to JSONL...")

        input_file = input_file or self.settings.game1_csv
        output_file = output_file or self.settings.game1_converted

        input_path = self._get_path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pl.read_csv(input_path)

        player_data = df.group_by("device").agg(
            [pl.col("time").alias("times"), pl.col("score").alias("scores")]
        )

        output_path = self._get_path(output_file, is_output=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in player_data.iter_rows(named=True):
                events = []
                for time, score in zip(row["times"], row["scores"]):
                    events.append(
                        {"time": int(time), "score": int(score), "event": "play"}
                    )

                events.sort(key=lambda x: x["time"])

                player_json = {"device_id": str(row["device"]), "records": events}

                f.write(json.dumps(player_json, ensure_ascii=False) + "\n")

        self.logger.info(
            f"Converted {len(player_data)} players to JSONL format at {output_path}"
        )
        return output_path

    def split_dataset(
        self, jsonl_file: Union[str, Path], train_ratio: float = None, seed: int = None
    ) -> Tuple[Path, Path]:
        """Splits a JSONL dataset into training and evaluation sets based on player IDs.

        Args:
            jsonl_file (Union[str, Path]): The name of the input JSONL file. If it starts with "game",
                                           it's assumed to be in `self.output_dir`; otherwise,
                                           it's assumed to be in `self.data_dir`.
            train_ratio (float): The proportion of players to include in the training set.
                                 Defaults to settings.train_ratio.
            seed (int): The random seed for shuffling player IDs to ensure reproducibility.
                        Defaults to settings.random_seed.

        Returns:
            Tuple[Path, Path]: A tuple containing the full paths to the generated
                               training and evaluation JSONL files.

        Raises:
            FileNotFoundError: If the input JSONL file does not exist.
        """
        train_ratio = train_ratio or self.settings.train_ratio
        seed = seed or self.settings.random_seed

        self.logger.info(f"Splitting {jsonl_file} into train/eval sets...")

        # Determine input path
        if isinstance(jsonl_file, Path) and jsonl_file.is_absolute():
            input_path = jsonl_file
        else:
            is_processed = str(jsonl_file).startswith("game")
            input_path = self._get_path(jsonl_file, is_output=is_processed)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input JSONL file not found for splitting: {input_path}"
            )

        # Get and shuffle player IDs
        player_ids = get_player_ids(input_path)
        random.seed(seed)
        random.shuffle(player_ids)

        # Split IDs
        split_idx = int(len(player_ids) * train_ratio)
        train_ids = set(player_ids[:split_idx])
        eval_ids = set(player_ids[split_idx:])

        # Generate output paths
        base_name = Path(jsonl_file).stem
        train_file_name = f"{base_name}{self.settings.train_suffix}"
        eval_file_name = f"{base_name}{self.settings.eval_suffix}"

        train_path = self._get_path(train_file_name, is_output=True)
        eval_path = self._get_path(eval_file_name, is_output=True)

        # Split the data
        results = split_jsonl_by_ids(
            input_file=input_path,
            train_ids=train_ids,
            eval_ids=eval_ids,
            train_output=train_path,
            eval_output=eval_path,
            id_field=None,
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

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing the names of the
                                       generated train and eval files for each game.
        """
        self.logger.info("Starting data preparation pipeline...")

        # Convert and split Game 1
        game1_jsonl_path = self.convert_game1_to_jsonl()
        game1_train_path, game1_eval_path = self.split_dataset(game1_jsonl_path)

        # Split Game 2
        game2_train_path, game2_eval_path = self.split_dataset(
            self.settings.game2_jsonl
        )

        results = {
            "game1": {"train": game1_train_path.name, "eval": game1_eval_path.name},
            "game2": {"train": game2_train_path.name, "eval": game2_eval_path.name},
        }

        self.logger.info("Data preparation complete!")
        return results
