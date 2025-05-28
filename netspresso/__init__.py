from pathlib import Path

from .netspresso import NPQAI, NetsPresso

__all__ = ["NetsPresso", "NPQAI"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
