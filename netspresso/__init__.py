from pathlib import Path

from .trainer import ModelTrainerV1, ModelTrainerV2
from .compressor import ModelCompressor
from .launcher import ModelConverter, ModelBenchmarker

version = (Path(__file__).parent / "VERSION").read_text().strip()

__version__ = version
