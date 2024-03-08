from pathlib import Path

from .netspresso import TAO, NetsPresso

__all__ = ["NetsPresso", "TAO"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
