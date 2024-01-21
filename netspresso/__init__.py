from pathlib import Path

from netspresso.trainer import Trainer
from netspresso.compressor import Compressor
from netspresso.launcher import Converter, Benchmarker


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
