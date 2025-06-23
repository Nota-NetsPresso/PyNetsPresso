from dataclasses import dataclass, field
from typing import List

from netspresso.enums import DataType, DeviceName, Framework, HardwareType, SoftwareVersion, TaskType
from netspresso.metadata.common import BaseMetadata


@dataclass
class BenchmarkTaskInfo:
    benchmark_task_uuid: str = ""
    framework: Framework = ""
    display_framework: str = ""
    device_name: DeviceName = ""
    display_device_name: str = ""
    display_brand_name: str = ""
    software_version: SoftwareVersion = ""
    display_software_version: str = ""
    data_type: DataType = ""
    hardware_type: HardwareType = ""


@dataclass
class BenchmarkResult:
    memory_footprint: int = None
    memory_footprint_gpu: int = None
    memory_footprint_cpu: int = None
    power_consumption: int = None
    ram_size: int = None
    latency: int = None
    file_size: int = None


@dataclass
class BenchmarkEnvironment:
    model_framework: str = ""
    system: str = ""
    machine: str = ""
    cpu: str = ""
    gpu: str = ""
    library: List[str] = field(default_factory=list)


@dataclass
class ProfilerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.PROFILE
    input_model_path: str = ""
    profile_task_info: BenchmarkTaskInfo = field(default_factory=BenchmarkTaskInfo)
    profile_result: BenchmarkResult = field(default_factory=BenchmarkResult)
