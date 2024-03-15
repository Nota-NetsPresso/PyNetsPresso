from dataclasses import dataclass, field
from typing import List

from netspresso_v2.metadata.common import InputModelInfo, DeviceInfo
from netspresso_v2.metadata.enums import (
    DataType,
    SoftwareVersion,
    DeviceName,
    HardwareType,
    Status,
    TaskType,
)


@dataclass
class BenchmarkTaskInfo:
    benchmark_task_id: str = ""
    data_type: DataType = ""
    target_device_name: DeviceName = ""
    software_version: SoftwareVersion = ""
    hardware_type: HardwareType = ""


@dataclass
class BenchmarkResult:
    memory_footprint_gpu: int = None
    memory_footprint_cpu: int = None
    power_consumption: int = None
    ram_size: int = None
    latency: int = None


@dataclass
class BenchmarkEnvironment:
    model_framework: str = ""
    system: str = ""
    machine: str = ""
    cpu: str = ""
    gpu: str = ""
    library: List[str] = field(default_factory=list)


@dataclass
class BenchmarkerMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.BENCHMARK
    benchmark_task_info: BenchmarkTaskInfo = field(default_factory=BenchmarkTaskInfo)
    input_model_info: InputModelInfo = field(default_factory=InputModelInfo)
    device_info: DeviceInfo = field(default_factory=DeviceInfo)
    benchmark_result: BenchmarkResult = field(default_factory=BenchmarkResult)
    benchmark_environment: BenchmarkEnvironment = field(
        default_factory=BenchmarkEnvironment
    )
