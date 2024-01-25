from pathlib import Path

from netspresso.trainer import Trainer
from netspresso.compressor import Compressor
from netspresso.converter import Converter
from netspresso.benchmarker import Benchmarker


__all__ = ["Trainer", "Compressor", "Converter", "Benchmarker"]


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
