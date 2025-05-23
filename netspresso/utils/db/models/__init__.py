from netspresso.utils.db.models.benchmark import BenchmarkResult, BenchmarkTask
from netspresso.utils.db.models.compression import CompressionModelResult, CompressionTask
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.evaluation import EvaluationTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.session import Base, engine

Base.metadata.create_all(engine)


__all__ = [
    "Model",
    "Project",
    "TrainingTask",
    "ConversionTask",
    "BenchmarkTask",
    "BenchmarkResult",
    "EvaluationTask",
    "CompressionTask",
    "CompressionModelResult",
]
