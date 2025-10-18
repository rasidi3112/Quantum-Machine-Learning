import logging
from pathlib import Path
from typing import Optional

from rich.console import Console # type: ignore
from rich.logging import RichHandler # type: ignore

LOGGER_NAME = "qml_app"
_console = Console()


def init_logger(log_level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    rich_handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        log_time_format="[%X]",
    )

    formatter = logging.Formatter(fmt="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logger initialized.")
    return logger