from dataclasses import dataclass, field
from typing import List

from netspresso_v2.metadata.enums import (
    Status,
    TaskType,
    DataType,
    Framework,
    DeviceName,
    SoftwareVersion,
    HardwareType,
)


@dataclass
class InputLayer:
    name: str = None
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class ConvertTaskInfo:
    convert_task_id: str = ""
    data_type: DataType = ""
    target_framework: Framework = ""
    target_device_name: DeviceName = ""
    software_version: SoftwareVersion = ""


@dataclass
class ModelInfo:
    model_fime_name: str = ""
    model_file_path: str = ""
    data_type: DataType = ""
    framework: Framework = ""
    input_layer: InputLayer = field(default_factory=InputLayer)


class InputModelInfo(ModelInfo):
    pass


class OutputModelInfo(ModelInfo):
    pass


@dataclass()
class DeviceInfo:
    device_name: DeviceName = ""
    software_version: SoftwareVersion = ""
    hardware_type: HardwareType = ""


@dataclass
class ConverterMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.CONVERT
    convert_task_info: ConvertTaskInfo = field(default_factory=ConvertTaskInfo)
    input_model_info: InputModelInfo = field(default_factory=InputModelInfo)
    output_model_info: OutputModelInfo = field(default_factory=OutputModelInfo)
    available_options: List[DeviceInfo] = field(default_factory=lambda: [DeviceInfo()])
