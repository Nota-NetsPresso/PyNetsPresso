from aenum import NamedConstant

from netspresso.enums.base import StrEnum


class Task(StrEnum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"


class LauncherTask(StrEnum):
    CONVERT = "convert"
    BENCHMARK = "benchmark"
    QUANTIZE = "quantize"
    GRAPH_OPTIMIZE = "graph-optimize"
    SIMULATE = "simulate"


class TaskStatusForDisplay(NamedConstant):
    # task status for display - launchx
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"
    TIMEOUT = "TIMEOUT"
