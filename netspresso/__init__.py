from pathlib import Path

from .netspresso import NetsPresso

__all__ = ["NetsPresso"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
