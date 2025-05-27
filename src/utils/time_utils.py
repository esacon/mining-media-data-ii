from datetime import datetime, timedelta
from typing import Union


def detect_time_format(timestamp: Union[int, float]) -> str:
    """Detects if a Unix timestamp is in seconds or milliseconds.

    This detection is based on the magnitude of the timestamp. Timestamps
    greater than 1e10 (which roughly corresponds to dates after early 2001 in milliseconds)
    are considered milliseconds.

    Args:
        timestamp (Union[int, float]): The Unix timestamp.

    Returns:
        str: 'seconds' if the timestamp is likely in seconds, 'milliseconds' otherwise.
    """
    if timestamp > 1e10:  # Roughly timestamps after early 2001 are in milliseconds
        return "milliseconds"
    else:
        return "seconds"


def convert_timestamp(
    timestamp: Union[int, float], to_format: str = "datetime"
) -> Union[datetime, int, float]:
    """Converts a Unix timestamp to different formats.

    Args:
        timestamp (Union[int, float]): The Unix timestamp to convert.
        to_format (str): The target format ('datetime', 'seconds', 'milliseconds').
                         Defaults to 'datetime'.

    Returns:
        Union[datetime, int, float]: The converted timestamp in the specified format.

    Raises:
        ValueError: If an unsupported target format is provided.
    """
    current_format = detect_time_format(timestamp)

    if current_format == "milliseconds":
        dt = datetime.fromtimestamp(timestamp / 1000)
    else:
        dt = datetime.fromtimestamp(timestamp)

    if to_format == "datetime":
        return dt
    elif to_format == "seconds":
        return int(dt.timestamp())
    elif to_format == "milliseconds":
        return int(dt.timestamp() * 1000)
    else:
        raise ValueError(f"Unsupported format: {to_format}")


def format_timestamp(
    timestamp: Union[int, float], format_string: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Formats a Unix timestamp into a human-readable string.

    Args:
        timestamp (Union[int, float]): The Unix timestamp.
        format_string (str): The `strftime` format string for the output.
                             Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
        str: The formatted timestamp string.
    """
    dt = convert_timestamp(timestamp, "datetime")
    return dt.strftime(format_string)


def get_time_boundaries(
    start_time: Union[int, float], observation_days: int, churn_days: int
) -> dict:
    """Calculates time boundaries for observation and churn periods.

    Given a start time and durations for observation and churn periods, this
    function returns the start and end datetimes for both periods, along with
    their ISO formatted string representations.

    Args:
        start_time (Union[int, float]): The starting Unix timestamp of the observation period.
        observation_days (int): The length of the observation period in days.
        churn_days (int): The length of the churn period in days.

    Returns:
        dict: A dictionary containing the following keys:
              'op_start' (datetime): Observation period start.
              'op_end' (datetime): Observation period end.
              'cp_start' (datetime): Churn period start.
              'cp_end' (datetime): Churn period end.
              'op_start_iso' (str): Observation period start in ISO format.
              'op_end_iso' (str): Observation period end in ISO format.
              'cp_start_iso' (str): Churn period start in ISO format.
              'cp_end_iso' (str): Churn period end in ISO format.
    """
    start_dt = convert_timestamp(start_time, "datetime")
    op_end = start_dt + timedelta(days=observation_days)
    cp_start = op_end
    cp_end = cp_start + timedelta(days=churn_days)

    return {
        "op_start": start_dt,
        "op_end": op_end,
        "cp_start": cp_start,
        "cp_end": cp_end,
        "op_start_iso": start_dt.isoformat(),
        "op_end_iso": op_end.isoformat(),
        "cp_start_iso": cp_start.isoformat(),
        "cp_end_iso": cp_end.isoformat(),
    }


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: A formatted string representing the duration (e.g., "60.0 seconds",
             "1.5 minutes", "2.3 hours").
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def is_within_period(
    timestamp: Union[int, float],
    start_time: Union[int, float],
    end_time: Union[int, float],
) -> bool:
    """Checks if a given timestamp falls within a specified time period.

    Args:
        timestamp (Union[int, float]): The timestamp to check.
        start_time (Union[int, float]): The start time of the period.
        end_time (Union[int, float]): The end time of the period.

    Returns:
        bool: True if the timestamp is within the period (inclusive), False otherwise.
    """
    ts_dt = convert_timestamp(timestamp, "datetime")
    start_dt = convert_timestamp(start_time, "datetime")
    end_dt = convert_timestamp(end_time, "datetime")

    return start_dt <= ts_dt <= end_dt
