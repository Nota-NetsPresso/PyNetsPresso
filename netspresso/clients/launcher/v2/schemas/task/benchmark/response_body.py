import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List

from netspresso.clients.launcher.v2.enums import (
    TaskStatus,
)
from netspresso.clients.launcher.v2.schemas.task.common import Device, TaskStatusInfo
from netspresso.clients.launcher.v2.schemas import ResponseItem, ResponseItems
from netspresso.enums import DataType


@dataclass(init=False)
class BenchmarkResult:
    processor: str
    ram_size: float
    latency: float
    power_consumption: float
    memory_footprint_cpu: float
    memory_footprint_gpu: float


@dataclass(init=False)
class BenchmarkEnvironment:
    model_framework: str
    library: list
    cpu: str = ""
    gpu: str = ""


@dataclass(init=False)
class BenchmarkTask:
    benchmark_task_id: str
    input_model_id: str
    target_device_name: str
    display_brand_name: str
    display_device_name: str
    data_type: DataType
    status: TaskStatus
    hardware_type: Optional[str] = None
    software_version: Optional[str] = ""
    display_software_version: Optional[str] = ""
    benchmark_result: Optional[BenchmarkResult] = None
    benchmark_environment: Optional[BenchmarkEnvironment] = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseBenchmarkTaskItem(ResponseItem):
    data: Optional[BenchmarkTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = BenchmarkTask(**self.data)


@dataclass(init=False)
class BenchmarkOption:
    option_id: str
    option_name: str
    framework: str
    device: Device
    sw_version: Optional[str] = ""

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseBenchmarkOptionItems(ResponseItems):
    data: List[Optional[BenchmarkOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [BenchmarkOption(**item) for item in self.data]


@dataclass
class ResponseBenchmarkStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)
