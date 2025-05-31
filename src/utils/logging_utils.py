import logging
import sys
from pathlib import Path
from typing import Optional, Union  # Added Union

# Default format string can be defined as a constant
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str,
    level_str: str = "INFO",  # Changed to string for user-friendliness
    log_file: Optional[Union[str, Path]] = None,  # Accepts str or Path
    format_string: Optional[str] = None,
    log_to_console: bool = True,  # Added this parameter
) -> logging.Logger:
    """Sets up a logger with consistent formatting, optional file output,
    and controllable console output.

    Args:
        name (str): The name of the logger.
        level_str (str): The logging level string (e.g., "INFO", "DEBUG").
                         Defaults to "INFO".
        log_file (Optional[Union[str, Path]]): Optional path to a log file.
        format_string (Optional[str]): Optional custom format string for log messages.
        log_to_console (bool): If True, logs to console. Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    # Convert string level to logging level integer
    log_level_int = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(log_level_int)

    # Clear direct handlers to avoid duplicate logs if called multiple times on the same logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    current_format_string = (
        format_string if format_string is not None else DEFAULT_LOG_FORMAT
    )
    formatter = logging.Formatter(current_format_string)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setLevel(log_level_int) # Handler level inherits from logger if not set
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file_path = Path(log_file)
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_file_path, mode="a", encoding="utf-8"
            )
            # file_handler.setLevel(log_level_int) # Handler level inherits from logger
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console if file logging setup fails
            print(
                f"Warning: Could not set up file logging for {name} at {log_file_path}. Error: {e}",
                file=sys.stderr,
            )
            if not log_to_console:  # If console wasn't already on, add it now
                console_handler_fallback = logging.StreamHandler(sys.stdout)
                console_handler_fallback.setFormatter(formatter)
                logger.addHandler(console_handler_fallback)

    # Prevent log messages from being passed to the root logger if this logger has handlers
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieves an existing logger by name. If it has no handlers,
    it sets it up with default console logging.

    This is useful for library modules that want to use logging without
    forcing a configuration if the main application hasn't set one up.
    However, for application entry points, calling setup_logger directly
    with desired configuration is usually preferred.

    Args:
        name (str): The name of the logger to retrieve or create.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    # If the logger has no handlers, it means it hasn't been configured by setup_logger yet.
    # So, set it up with some defaults (e.g., INFO level, console output).
    if not logger.handlers:
        # This will set it up with default: level="INFO", log_to_console=True, no file logging
        return setup_logger(name)
    return logger


class LoggerMixin:
    """A mixin class that provides a convenient `logger` property.

    Classes inheriting from LoggerMixin will have a `self.logger` attribute
    that returns a `logging.Logger` instance named after the class.
    The logger is configured via `get_logger` on first access.
    """

    @property
    def logger(self) -> logging.Logger:
        """Gets the logger instance for the current class.

        The logger is created and cached on the first access using get_logger.
        """
        # The class name is used as the logger name.
        class_name = self.__class__.__name__

        # Check if a logger specific to this instance with this name already exists and is configured
        # This avoids re-configuring if get_logger is called multiple times.
        # However, get_logger itself handles the "configure if no handlers" logic.

        # Using a private attribute to cache the logger for the instance.
        if not hasattr(self, "_logger_instance_cache"):
            self._logger_instance_cache = get_logger(class_name)
        return self._logger_instance_cache
