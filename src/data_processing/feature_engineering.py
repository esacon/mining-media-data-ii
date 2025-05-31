import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.config import Settings
from src.utils import (
    LoggerMixin,
    convert_timestamp,
    count_lines,
    ensure_dir,
    format_duration,
    jsonl_iterator,
)


class FeatureEngineering(LoggerMixin):
    """
    Extracts behavioral features from labeled player datasets for churn prediction.
    Implements features described in Kim et al. (2017) for telemetry-based churn prediction.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.features_dir = ensure_dir(settings.features_dir)
        self.processed_dir = settings.processed_dir
        self.observation_days = settings.observation_days

    def _get_path(self, file_path: Union[str, Path]) -> Path:
        """Get full path for files."""
        return (
            self.processed_dir / file_path
            if not Path(file_path).is_absolute()
            else Path(file_path)
        )

    def _get_empty_features(self, game_name: str = "game1") -> Dict[str, Any]:
        """Returns empty feature set with appropriate defaults."""
        features = {
            "playCount": 0,
            "bestScore": 0,
            "meanScore": 0.0,
            "worstScore": 0,
            "sdScore": 0.0,
            "bestScoreIndex": 0.0,
            "bestSubMeanCount": 0.0,
            "bestSubMeanRatio": 0.0,
            "activeDuration": 0.0,
            "consecutivePlayRatio": 0.0,
        }
        if game_name == "game2":
            features.update({"purchaseCount": 0, "highestPrice": 0.0})
        return features

    def _extract_game_events(
        self, records: List[Dict], game_name: str
    ) -> Dict[str, Any]:
        """Extract relevant events, scores, and timestamps based on game type."""
        if game_name == "game1":
            play_events = [r for r in records if r.get("event") == "play"]
            scores = [
                r.get("score", 0)
                for r in play_events
                if isinstance(r.get("score"), (int, float))
            ]
            timestamps = [r["time"] for r in play_events if "time" in r]
            return {
                "valid_events": play_events,
                "scores": scores,
                "timestamps": timestamps,
            }
        elif game_name == "game2":
            progress_events = [
                r
                for r in records
                if (
                    r.get("event") == "progress"
                    and r.get("properties", {}).get("action") == "complete"
                    and r.get("properties", {}).get("type") == "race"
                )
            ]
            scores = [
                event["properties"]["reward"]
                for event in progress_events
                if isinstance(event.get("properties", {}).get("reward"), (int, float))
            ]
            timestamps = [r["time"] for r in progress_events if "time" in r]
            return {
                "valid_events": progress_events,
                "scores": scores,
                "timestamps": timestamps,
            }
        else:
            return {"valid_events": [], "scores": [], "timestamps": []}

    def _calculate_score_features(
        self, scores: List[Union[int, float]]
    ) -> Dict[str, Any]:
        """Calculate score-based features from Kim et al. (2017)."""
        if not scores:
            return {
                "bestScore": 0,
                "meanScore": 0.0,
                "worstScore": 0,
                "sdScore": 0.0,
                "bestScoreIndex": 0.0,
                "bestSubMeanCount": 0.0,
                "bestSubMeanRatio": 0.0,
            }

        best_score = max(scores)
        mean_score = statistics.mean(scores)
        worst_score = min(scores)
        sd_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

        best_indices = [i for i, score in enumerate(scores) if score == best_score]
        best_score_index = max(best_indices) / len(scores) if len(scores) > 1 else 0.0

        best_sub_mean = best_score - mean_score
        best_sub_mean_count = best_sub_mean / len(scores)
        best_sub_mean_ratio = best_sub_mean / mean_score if mean_score != 0 else 0.0

        return {
            "bestScore": best_score,
            "meanScore": mean_score,
            "worstScore": worst_score,
            "sdScore": sd_score,
            "bestScoreIndex": best_score_index,
            "bestSubMeanCount": best_sub_mean_count,
            "bestSubMeanRatio": best_sub_mean_ratio,
        }

    def _calculate_temporal_features(self, datetimes: List[datetime]) -> Dict[str, Any]:
        """Calculate temporal features from Kim et al. (2017)."""
        if len(datetimes) < 1:
            return {"activeDuration": 0.0, "consecutivePlayRatio": 0.0}

        # Active duration
        active_duration = 0.0
        if len(datetimes) > 1:
            active_duration = (datetimes[-1] - datetimes[0]).total_seconds() / 3600.0

        # Consecutive play ratio (30-minute threshold)
        consecutive_count = 0
        if len(datetimes) >= 2:
            consecutive_threshold = 30 * 60
            for i in range(1, len(datetimes)):
                time_diff = (datetimes[i] - datetimes[i - 1]).total_seconds()
                if time_diff <= consecutive_threshold:
                    consecutive_count += 1

        consecutive_ratio = (
            consecutive_count / (len(datetimes) - 1) if len(datetimes) > 1 else 0.0
        )

        return {
            "activeDuration": active_duration,
            "consecutivePlayRatio": consecutive_ratio,
        }

    def extract_common_features(
        self, all_records: List[Dict], game_name: str = "game1"
    ) -> Dict[str, Any]:
        """
        Extracts the 10 common behavioral features from Kim et al. (2017).
        Filters records based on the observation period relative to the first play.

        Args:
            all_records (List[Dict]): A list of dictionaries containing game events.
            game_name (str): The name of the game.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted features.
        """
        if not all_records:
            return self._get_empty_features(game_name)

        try:
            sorted_records = sorted(all_records, key=lambda x: x.get("time", 0))
            if not sorted_records:
                return self._get_empty_features(game_name)

            first_play_timestamp = sorted_records[0].get("time")
            if first_play_timestamp is None:
                self.logger.warning("First play timestamp not found in records.")
                return self._get_empty_features(game_name)

            first_play_datetime = convert_timestamp(first_play_timestamp, "datetime")
            observation_end_datetime = first_play_datetime + timedelta(
                days=self.observation_days
            )

            observation_records = [
                r
                for r in sorted_records
                if convert_timestamp(r.get("time", 0), "datetime")
                < observation_end_datetime
            ]

            if not observation_records:
                return self._get_empty_features(game_name)

            events_data = self._extract_game_events(observation_records, game_name)
            if not events_data["valid_events"]:
                return self._get_empty_features(game_name)

            scores = events_data["scores"]
            timestamps = events_data["timestamps"]
            datetimes = sorted([convert_timestamp(ts, "datetime") for ts in timestamps])

            play_count = len(timestamps)

            features = {
                "playCount": play_count,
                **self._calculate_score_features(scores),
                **self._calculate_temporal_features(datetimes),
            }
            return features
        except Exception as e:
            self.logger.warning(f"Error extracting common features: {e}")
            return self._get_empty_features(game_name)

    def extract_game2_specific_features(
        self, observation_records: List[Dict]
    ) -> Dict[str, Any]:
        """
        Extracts Game 2 specific features: purchaseCount and highestPrice.

        Args:
            observation_records (List[Dict]): A list of dictionaries containing game events.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted features.
        """
        if not observation_records:
            return {"purchaseCount": 0, "highestPrice": 0.0}

        try:
            purchase_events = [
                r
                for r in observation_records
                if r.get("event") in ["softPurchase", "hardPurchase", "purchase"]
            ]
            purchase_count = len(purchase_events)

            if purchase_count == 0:
                return {"purchaseCount": 0, "highestPrice": 0.0}

            purchase_prices = []
            for event in purchase_events:
                price = event.get("properties", {}).get("price", 0)
                if isinstance(price, (int, float)) and price > 0:
                    purchase_prices.append(price)

            highest_price = max(purchase_prices) if purchase_prices else 0.0

            return {
                "purchaseCount": purchase_count,
                "highestPrice": highest_price,
            }
        except Exception as e:
            self.logger.warning(f"Error extracting Game 2 features: {e}")
            return {"purchaseCount": 0, "highestPrice": 0.0}

    def calculate_features(
        self, player_data: Dict, game_name: str = None
    ) -> Union[Dict[str, Any], None]:
        """Calculate churn prediction features from processed player records.
        Enhanced to handle both regular dataset format and pipeline format.

        Args:
            player_data (Dict): A dictionary containing the player's data.
            game_name (str, optional): The name of the game. If None, will auto-detect.

        Returns:
            Union[Dict[str, Any], None]: A dictionary containing the extracted features or None if an error occurs.
        """
        try:
            player_id = player_data.get("player_id")
            churned = player_data.get("churned", False)
            all_records = player_data.get("observation_records", [])

            if not player_id or not isinstance(all_records, list):
                return None

            if game_name is None:
                game_name = "game1"
                for record in all_records[:10]:
                    if (
                        record.get("event") == "progress"
                        and record.get("properties", {}).get("type") == "race"
                    ):
                        game_name = "game2"
                        break

            feature_row = {
                "player_id": player_id,
                "churned": int(churned),
                **self.extract_common_features(all_records, game_name),
            }

            if game_name == "game2":
                feature_row.update(self.extract_game2_specific_features(all_records))

            return feature_row
        except Exception as e:
            self.logger.warning(f"Error processing player: {e}")
            return None

    def compute_feature_stats(self, features_df: pd.DataFrame) -> None:
        """Compute and log feature statistics using approach adapted for DataFrames.

        Args:
            features_df: DataFrame containing extracted features
        """
        if features_df.empty:
            self.logger.info("Feature stats - No features found to summarize.")
            return

        active_durations = features_df.get(
            "activeDuration", pd.Series(dtype=float)
        ).dropna()
        play_counts = features_df.get("playCount", pd.Series(dtype=int)).dropna()
        mean_scores = features_df.get("meanScore", pd.Series(dtype=float)).dropna()

        if len(active_durations) > 0 and len(play_counts) > 0:
            self.logger.info(
                "Feature stats - activeDuration: avg=%.2fh, playCount: avg=%.1f, meanScore: avg=%.1f",
                active_durations.mean(),
                play_counts.mean(),
                mean_scores.mean() if len(mean_scores) > 0 else 0,
            )
        else:
            self.logger.info("Feature stats - No features found to summarize.")

    def extract_features_from_dataset(
        self, dataset_file: Union[str, Path], game_name: str
    ) -> pd.DataFrame:
        """
        Extracts features from a labeled dataset file using Kim et al. (2017) methodology.

        Args:
            dataset_file (Union[str, Path]): The path to the dataset file.
            game_name (str): The name of the game.

        Returns:
            pd.DataFrame: A pandas dataframe containing the extracted features.
        """
        dataset_path = self._get_path(dataset_file)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        total_lines = count_lines(dataset_path)
        self.logger.info(
            f"Extracting features from {dataset_file} for {game_name} ({total_lines:,} players)"
        )

        features_list = []
        errors_count = 0

        for line_num, player_data in enumerate(jsonl_iterator(dataset_path)):
            if (line_num + 1) % self.settings.progress_interval == 0:
                progress = (line_num + 1) / total_lines * 100 if total_lines > 0 else 0
                self.logger.info(
                    f"Processed {line_num + 1:,} players ({progress:.1f}%)..."
                )

            feature_row = self.calculate_features(player_data, game_name)

            if feature_row is None:
                errors_count += 1
                if errors_count > total_lines * 0.1:
                    self.logger.error("Too many errors, stopping extraction.")
                    break
            else:
                features_list.append(feature_row)

        if errors_count > 0:
            self.logger.warning(
                f"Feature extraction completed with {errors_count} errors"
            )

        self.logger.info(
            f"Successfully extracted features for {len(features_list):,} players from {game_name}"
        )

        df = pd.DataFrame(features_list)
        if df.empty:
            raise ValueError("No features extracted - check input data format")

        return self._validate_and_clean_features(df, game_name)

    def run_feature_extraction(self) -> None:
        """
        Run feature extraction for all datasets.

        Raises:
            FileNotFoundError: If the input file is not found.
            ValueError: If the input data format is invalid.
            Exception: If an error occurs during feature extraction.
        """
        self.logger.info("Starting feature extraction for all datasets...")
        extraction_start_time = time.time()

        datasets = [
            (
                self.settings.game1_ds1,
                "game1",
                self.settings.game1_ds1_features,
                "game1_DS1",
            ),
            (
                self.settings.game1_ds2,
                "game1",
                self.settings.game1_ds2_features,
                "game1_DS2",
            ),
            (
                self.settings.game2_ds1,
                "game2",
                self.settings.game2_ds1_features,
                "game2_DS1",
            ),
            (
                self.settings.game2_ds2,
                "game2",
                self.settings.game2_ds2_features,
                "game2_DS2",
            ),
        ]

        successful_extractions = 0
        failed_extractions = 0
        all_features_dfs = []

        for input_file, game_name, output_file, dataset_name in datasets:
            dataset_start_time = time.time()
            input_path = self.processed_dir / input_file

            if not input_path.exists():
                self.logger.warning(
                    f"Input file not found: {input_file}, skipping {dataset_name}"
                )
                failed_extractions += 1
                continue

            try:
                features_df = self.extract_features_from_dataset(input_file, game_name)
                output_path = self.features_dir / output_file
                features_df.to_csv(output_path, index=False)
                dataset_time = time.time() - dataset_start_time

                total_players = len(features_df)

                self.logger.info(
                    "Processed %s: %d players, skipped %d, %s",
                    dataset_name,
                    total_players,
                    0,
                    format_duration(dataset_time),
                )
                successful_extractions += 1
                all_features_dfs.append(features_df)
            except Exception as e:
                self.logger.error(f"✗ Error processing {dataset_name}: {e}")
                failed_extractions += 1

        total_extraction_time = time.time() - extraction_start_time

        self.logger.info(
            "Feature extraction completed in %s",
            format_duration(total_extraction_time),
        )

        if all_features_dfs:
            combined_df = pd.concat(all_features_dfs, ignore_index=True)
            self.compute_feature_stats(combined_df)

        if successful_extractions == len(datasets):
            self.logger.info(
                f"🎉 Feature extraction completed successfully! "
                f"{successful_extractions}/{len(datasets)} datasets processed "
                f"in {format_duration(total_extraction_time)}"
            )
        else:
            self.logger.warning(
                f"⚠️ Feature extraction completed with issues: "
                f"{successful_extractions}/{len(datasets)} datasets successful, "
                f"{failed_extractions} failed in {format_duration(total_extraction_time)}"
            )

    def _validate_and_clean_features(
        self, df: pd.DataFrame, game_name: str
    ) -> pd.DataFrame:
        """
        Validates extracted features for data quality, checking for missing columns
        and infinite/NaN values.

        Args:
            df (pd.DataFrame): A pandas dataframe containing the features.
            game_name (str): The name of the game.

        Returns:
            pd.DataFrame: A pandas dataframe containing the validated features.
        """
        required_cols = ["player_id", "churned", "playCount"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = df[numeric_cols].isin([np.inf, -np.inf]).sum()
        nan_counts = df[numeric_cols].isna().sum()

        if inf_counts.sum() > 0:
            self.logger.warning(
                f"Found infinite values in columns: {inf_counts[inf_counts > 0].to_dict()}"
            )
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

        if nan_counts.sum() > 0:
            self.logger.warning(
                f"Found NaN values in columns: {nan_counts[nan_counts > 0].to_dict()}"
            )
            df[numeric_cols] = df[numeric_cols].fillna(0)

        churn_rate = df["churned"].mean()
        feature_count = len(df.columns) - 2
        expected_features = 10 + (2 if game_name == "game2" else 0)

        self.logger.info(
            f"Validated: {len(df)} players, {feature_count} features "
            f"(expected: {expected_features}), {churn_rate:.1%} churn rate"
        )
        return df
