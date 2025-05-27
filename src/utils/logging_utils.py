import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Sets up a logger with consistent formatting and optional file output.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
                     Defaults to logging.INFO.
        log_file (Optional[str]): Optional path to a log file where messages will also be written.
                                   If None, logs only to console.
        format_string (Optional[str]): Optional custom format string for log messages.
                                       If None, a default format is used.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieves an existing logger by name or sets up a new one with default configuration.

    Args:
        name (str): The name of the logger to retrieve or create.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerMixin:
    """A mixin class that provides a convenient `logger` property to any class.

    Classes inheriting from LoggerMixin will have a `self.logger` attribute
    that returns a `logging.Logger` instance named after the class.
    """

    @property
    def logger(self) -> logging.Logger:
        """Gets the logger instance for the current class.

        The logger is created and cached on the first access.

        Returns:
            logging.Logger: The logger instance for the class.
        """
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
