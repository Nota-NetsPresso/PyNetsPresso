import sys
from pathlib import Path

from loguru import logger

# Configure logger format FIRST, before importing other modules
# This ensures the custom format is applied before any module uses logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

from .netspresso import NPQAI, NetsPresso  # noqa: E402

__all__ = ["NetsPresso", "NPQAI"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
