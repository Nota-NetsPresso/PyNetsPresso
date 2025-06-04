from enum import Enum

from aenum import NamedConstant


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"


class LauncherTask(str, Enum):
    CONVERT = "convert"
    BENCHMARK = "benchmark"
    QUANTIZE = "quantize"


class TaskStatusForDisplay(NamedConstant):
    # task status for display - launchx
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"
    TIMEOUT = "TIMEOUT"


class TaskType(str, Enum):
    COMPRESS = "compress"
    RETRAIN = "retrain"
    CONVERT = "convert"
    BENCHMARK = "benchmark"
    EVALUATE = "evaluate"
