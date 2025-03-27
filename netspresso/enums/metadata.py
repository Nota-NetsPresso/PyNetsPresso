from netspresso.enums.base import StrEnum


class TaskType(StrEnum):
    TRAIN = "train"
    COMPRESS = "compress"
    CONVERT = "convert"
    QUANTIZE = "quantize"
    BENCHMARK = "benchmark"


class Status(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
