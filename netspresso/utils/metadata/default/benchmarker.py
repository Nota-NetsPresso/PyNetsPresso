import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from netspresso.enums.metadata import Status, TaskType

from .common import TargetDevice


@dataclass
class InputShape:
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class BenchmarkInfo:
    target_device: str = ""
    filename: str = ""
    data_type: str = ""
    processor: str = ""
    software_version: str = ""
    hardware_type: str = ""
    input_model_uuid: str = ""
    benchmark_task_uuid: str = ""
    devicefarm_benchmark_task_uuid: str = ""
    devicefarm_model_uuid: str = ""


@dataclass
class Result:
    memory_footprint_gpu: int = 0
    memory_footprint_cpu: int = 0
    latency: int = 0
    ram_size: int = 0
    power_consumption: int = 0
    file_size: int = 0


@dataclass
class BenchmarkerMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.BENCHMARK
    benchmark_info: BenchmarkInfo = field(default_factory=BenchmarkInfo)
    result: Result = field(default_factory=Result)

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_status(self, status: Status) -> None:
        self.status = status

    def update_benchmark_info(
        self,
        target_device: str,
        file_name: str,
        data_type: str,
        processor: str,
        software_version: str,
        hardware_type: str,
        input_model_uuid: str,
        benchmark_task_uuid: str,
        devicefarm_benchmark_task_uuid: str,
        devicefarm_model_uuid: str,
    ) -> None:
        self.benchmark_info.target_device = target_device
        self.benchmark_info.filename = file_name
        self.benchmark_info.data_type = data_type
        self.benchmark_info.processor = processor
        self.benchmark_info.software_version = software_version
        self.benchmark_info.hardware_type = hardware_type
        self.benchmark_info.input_model_uuid = input_model_uuid
        self.benchmark_info.benchmark_task_uuid = benchmark_task_uuid
        self.benchmark_info.devicefarm_benchmark_task_uuid = devicefarm_benchmark_task_uuid
        self.benchmark_info.devicefarm_model_uuid = devicefarm_model_uuid

    def update_result(
        self,
        memory_footprint_gpu: int,
        memory_footprint_cpu: int,
        latency: int,
        ram_size: int,
        power_consumption: int,
        file_size: int,
    ) -> None:
        self.result.memory_footprint_gpu = memory_footprint_gpu
        self.result.memory_footprint_cpu = memory_footprint_cpu
        self.result.latency = latency
        self.result.ram_size = ram_size
        self.result.power_consumption = power_consumption
        self.result.file_size = file_size
