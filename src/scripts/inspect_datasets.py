import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from src.config import get_settings
from src.data_processing.dataset_creation import DatasetCreator
from src.utils import format_timestamp, load_jsonl_sample

sys.path.append(str(Path(__file__).parent.parent))


def print_player_details(player_data: Dict[str, Any], detailed: bool = False) -> None:
    """Prints detailed information about a single player's labeled data.

    Args:
        player_data (Dict[str, Any]): A dictionary containing a single player's
                                     labeled data, as produced by DatasetCreator.
        detailed (bool): If True, also prints the first few observation events.
    """
    print(f"\nPlayer ID: {player_data['player_id']}")
    print(f"Churned: {'YES' if player_data['churned'] else 'NO'}")
    print(f"Observation Period: {player_data['op_start']} to {player_data['op_end']}")
    print(f"Events in OP: {player_data['op_event_count']}")
    print(f"Events in CP: {player_data['cp_event_count']}")

    if detailed and player_data["observation_records"]:
        print("\nFirst 3 observation events:")
        for i, event in enumerate(player_data["observation_records"][:3]):
            event_type = event.get("event", "play")
            score = event.get("score", "N/A")
            time = event["time"]

            time_str = format_timestamp(time)
            print(f"  {i+1}. {time_str} - {event_type} (score: {score})")


def analyze_dataset(
    dataset_path: Path, dataset_creator: DatasetCreator, detailed: bool = False
) -> None:
    """Analyzes and prints statistics about a specific labeled dataset.

    It loads overall statistics and displays sample player records for inspection.

    Args:
        dataset_path (Path): The full path to the labeled JSONL dataset file.
        dataset_creator (DatasetCreator): An instance of DatasetCreator to get dataset summary.
        detailed (bool): If True, enables detailed event printing for sample players.
    """
    print(f"\nAnalyzing: {dataset_path.name}")
    print("-" * 50)

    summary = dataset_creator.get_dataset_summary(dataset_path)

    print(f"Total Players: {summary.get('total_players', 0):,}")
    print(
        f"Churned: {summary.get('churned_players', 0):,} ({summary.get('churn_rate', 0.0):.1%})"
    )
    print(
        f"Retained: {summary.get('retained_players', 0):,} ({summary.get('retention_rate', 0.0):.1%})"
    )
    print(f"Avg events in OP: {summary.get('op_events', {}).get('mean', 0.0):.1f}")
    print(f"Avg events in CP: {summary.get('cp_events', {}).get('mean', 0.0):.1f}")
    print(f"Skipped records (during creation): {summary.get('skipped_count', 0):,}")

    print("\nSample Players:")
    samples = load_jsonl_sample(dataset_path, n_samples=3)
    for sample in samples:
        print_player_details(sample, detailed=detailed)


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect churn prediction datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory where processed datasets are located (overrides config)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed player information (e.g., first few observation events)",
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["game1", "game2"],
        help="Inspect datasets only for a specific game (e.g., 'game1' or 'game2')",
    )
    return parser


def _filter_and_organize_files(labeled_files: list, game_filter: str = None) -> tuple:
    """Filter labeled files by game and organize them into game1 and game2 lists."""
    if game_filter:
        labeled_files = [f for f in labeled_files if game_filter in f.name]

    game1_files = sorted([f for f in labeled_files if "game1" in f.name])
    game2_files = sorted([f for f in labeled_files if "game2" in f.name])

    return game1_files, game2_files


def _analyze_game_files(
    game_files: list, game_name: str, dataset_creator, detailed: bool
) -> None:
    """Analyze and print results for files from a specific game."""
    if game_files:
        print("\n" + "=" * 30 + " " + game_name + " " + "=" * 30)
        for file in game_files:
            analyze_dataset(file, dataset_creator, detailed=detailed)


def main() -> None:
    """Main function to parse arguments and orchestrate dataset inspection."""
    parser = _setup_argument_parser()
    args = parser.parse_args()

    settings = get_settings()
    if args.data_dir:
        settings.processed_dir = Path(args.data_dir)

    data_dir = settings.processed_dir
    labeled_files = list(data_dir.glob("*_labeled.jsonl"))

    if not labeled_files:
        print(
            "No labeled datasets found. Please ensure the data pipeline has been run."
        )
        return

    game1_files, game2_files = _filter_and_organize_files(labeled_files, args.game)
    dataset_creator = DatasetCreator(settings)

    print("=" * 70)
    print("CHURN PREDICTION DATASET INSPECTION")
    print("=" * 70)

    _analyze_game_files(game1_files, "GAME 1", dataset_creator, args.detailed)
    _analyze_game_files(game2_files, "GAME 2", dataset_creator, args.detailed)


if __name__ == "__main__":
    main()
