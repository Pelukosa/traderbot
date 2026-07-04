from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_file: str | Path = "traderbot.log") -> None:
    """Configure loguru: colorised console + rotating file audit trail."""
    logger.remove()  # Remove default stderr handler

    # Console — warning+ only in production, info+ in dev
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:{line} — {message}",
        level="INFO",
        colorize=True,
    )

    # File — full audit trail, rotated daily
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )

    logger.info("Logger initialised — writing to {}", log_path.resolve())
