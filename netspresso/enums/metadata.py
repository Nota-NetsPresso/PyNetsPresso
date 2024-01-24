from enum import Enum


class TaskType(str, Enum):
    TRAIN = "train"
    COMPRESS = "compress"
    CONVERT = "convert"
    BENCHMARK = "benchmark"


class Status(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
