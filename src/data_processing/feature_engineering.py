"""Feature engineering module for mining-media-data-ii.

Provides functions for loading previous creation steps and extracting features from datasets.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

def format_duration(seconds):
    """Format duration in seconds to H:MM:SS string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

class FeatureEngineering:
    def __init__(self, output_dir, settings, logger):
        self.output_dir = output_dir
        self.settings = settings
        self.logger = logger

    def load_previous_creation_step(self):
        self.logger.info(
            "No dataset_creation results found in memory, "
            "attempting to load from previous output files "
            "(DS1 only for game1 and game2)."
        )
        results_path = (
            self.output_dir /
            self.settings.pipeline_results
        )
        if results_path.exists():
            with open(
                results_path, "r", encoding="utf-8"
            ) as f:
                previous_results = json.load(f)
                creation_step = previous_results.get(
                    "dataset_creation"
                )
                if creation_step is None:
                    raise RuntimeError(
                        "Could not find 'dataset_creation' "
                        "in previous pipeline results."
                    )
                # Only keep DS1 files for game1 and game2
                filtered_results = {}
                for game in ["game1", "game2"]:
                    if game in creation_step.get("results", {}):
                        ds1_file = creation_step["results"][game].get("DS1")
                        if ds1_file:
                            file_path = self.output_dir / ds1_file
                            if file_path.exists():
                                with open(
                                    file_path, "r", encoding="utf-8"
                                ) as data_f:
                                    record_count = sum(1 for _ in data_f)
                                self.logger.info(
                                    "Loaded %s for %s DS1: %d records",
                                    ds1_file, game, record_count
                                )
                            else:
                                self.logger.warning(
                                    "Expected dataset file not found: %s",
                                    file_path
                                )
                            filtered_results[game] = {"DS1": ds1_file}
                        else:
                            self.logger.warning(
                                "DS1 file not found for %s in previous results.", game
                            )
                if not filtered_results:
                    raise RuntimeError(
                        "No DS1 files found for game1 or game2 in previous pipeline results."
                    )
                # Return a creation_step-like dict with only DS1 for game1 and game2
                return {"results": filtered_results}
        else:
            raise RuntimeError(
                f"Could not find previous pipeline results at {results_path}"
            )

    def calculate_features(self, player_data: Dict) -> Optional[Dict]:
        """Calculate churn prediction features from processed player records.

        Args:
            player_data: Dictionary containing processed player data from
                dataset creation step.

        Returns:
            Dictionary with original player data plus calculated features,
            or None if invalid
        """
        def get_play_times(records):
            times = []
            for r in records:
                if isinstance(r["time"], str):
                    times.append(
                        int(datetime.fromisoformat(r["time"]).timestamp() * 1000)
                    )
                else:
                    times.append(r["time"])
            return times

        def get_scores(records):
            scores = []
            for r in records:
                if (
                    "properties" in r
                    and r["properties"]
                    and "score" in r["properties"]
                ):
                    scores.append(r["properties"]["score"])
                elif "score" in r:
                    scores.append(r["score"])
            return scores

        observation_records = player_data.get("observation_records", [])
        if not observation_records:
            return None

        play_times = get_play_times(observation_records)
        active_duration = (max(play_times) - min(play_times)) / (1000 * 3600)
        play_count = len(observation_records)

        consecutive_threshold = 10 * 60 * 1000  # 10 minutes in ms
        sorted_play_times = sorted(play_times)
        consecutive_plays = sum(
            1
            for i in range(len(sorted_play_times) - 1)
            if (sorted_play_times[i + 1] - sorted_play_times[i]) <= consecutive_threshold
        )
        consecutive_play_ratio = (
            consecutive_plays / (play_count - 1) if play_count > 1 else 0
        )

        scores = get_scores(observation_records)
        best_score = max(scores) if scores else 0
        worst_score = min(scores) if scores else 0
        mean_score = sum(scores) / len(scores) if scores else 0

        def get_purchase_info(records):
            purchase_count = 0
            max_purchase = 0
            for r in records:
                if (
                    r.get("event") == "softPurchase"
                    and "properties" in r
                    and r["properties"]
                    and "price" in r["properties"]
                ):
                    purchase_count += 1
                    price = r["properties"]["price"]
                    if price > max_purchase:
                        max_purchase = price
            return purchase_count, max_purchase

        purchase_count, max_purchase = get_purchase_info(observation_records)

        features = {
            "player_id": player_data.get("player_id"),
            "active_duration": active_duration,
            "play_count": play_count,
            "consecutive_play_ratio": consecutive_play_ratio,
            "best_score": best_score,
            "worst_score": worst_score,
            "mean_score": mean_score,
            "purchase_count": purchase_count,
            "max_purchase": max_purchase,
            "churned": player_data.get("churned"),
            "op_start": player_data.get("op_start"),
            "op_end": player_data.get("op_end"),
            "cp_start": player_data.get("cp_start"),
            "cp_end": player_data.get("cp_end"),
            "op_event_count": player_data.get("op_event_count"),
        }
        
        return features

    def run_feature_extraction(
        self, creation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the feature extraction step on created datasets.

        Args:
            creation_results: Output from run_dataset_creation(),
                containing paths to datasets

        Returns:
            Dictionary containing paths to feature-extracted datasets
            and execution time
        """
        
        if creation_results is None:
            creation_results = self.load_previous_creation_step()
            
        self.logger.info("\n%s", "=" * 50)
        self.logger.info("STEP 3: FEATURE EXTRACTION")
        self.logger.info("=" * 50)

        def process_dataset(input_path, output_path):
            processed_count = 0
            skipped_count = 0
            with open(output_path, "w", encoding="utf-8") as f_out:
                with open(input_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        player_data = json.loads(line)
                        features = self.calculate_features(player_data)
                        if features:
                            f_out.write(json.dumps(features, ensure_ascii=False) + "\n")
                            processed_count += 1
                        else:
                            skipped_count += 1
            return processed_count, skipped_count

        start_time = time.time()
        feature_results = {}

        for game in ["game1", "game2"]:
            feature_results[game] = {}
            ds_type = "DS1"
            input_path = self.output_dir / creation_results["results"][game][ds_type]
            output_file = f"{game}_{ds_type}_features.jsonl"
            output_path = self.output_dir / output_file

            processed_count, skipped_count = process_dataset(input_path, output_path)

            feature_results[game][ds_type] = output_file
            self.logger.info(
                "Processed %s %s: %d players, skipped %d",
                game,
                ds_type,
                processed_count,
                skipped_count,
            )

        execution_time = time.time() - start_time
        self.logger.info(
            "Feature extraction completed in %s",
            format_duration(execution_time),
        )
        
        self.logger.info(
            "\nFEATURE EXTRACTION SUMMARY:\n"
            "  - Features files: %s\n"
            "  - Execution time: %s",
            feature_results,
            format_duration(execution_time)
        )
        
        # self.compute_feature_stats(feature_results)

        return {
            "results": feature_results,
            "execution_time": execution_time,
        }
        
    def compute_feature_stats(self, feature_results):
        """Compute and log average active_duration and play_count from feature files."""
        active_durations = []
        play_counts = []
        for game, datasets in feature_results.items():
            for ds_type, feature_file in datasets.items():
                feature_path = self.output_dir / feature_file
                if not feature_path.exists():
                    self.logger.warning(f"Feature file not found: {feature_path}")
                    continue
                with open(feature_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            feat = json.loads(line)
                            ad = feat.get("active_duration", 0)
                            pc = feat.get("play_count", 0)
                            if isinstance(ad, (int, float)):
                                active_durations.append(ad)
                            if isinstance(pc, (int, float)):
                                play_counts.append(pc)
                        except Exception as e:
                            self.logger.warning(f"Error parsing line in {feature_file}: {e}")
                            continue

        if active_durations and play_counts:
            self.logger.info(
                "Feature stats - active_duration: avg=%.2fh, play_count: avg=%.1f",
                np.mean(active_durations),
                np.mean(play_counts),
            )
        else:
            self.logger.info("Feature stats - No features found to summarize.")