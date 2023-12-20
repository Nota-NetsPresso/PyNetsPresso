from pathlib import Path

from netspresso.trainer import ModelTrainer
from netspresso.compressor import ModelCompressor
from netspresso.launcher import ModelConverter, ModelBenchmarker


version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
