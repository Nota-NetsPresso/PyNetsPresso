from dataclasses import dataclass, field
from typing import List

from netspresso_v2.metadata.common import InputModelInfo, OutputModelInfo, DeviceInfo
from netspresso_v2.metadata.enums import (
    Status,
    TaskType,
    DataType,
    Framework,
    DeviceName,
    SoftwareVersion
)


@dataclass
class ConvertTaskInfo:
    convert_task_id: str = ""
    data_type: DataType = ""
    target_framework: Framework = ""
    target_device_name: DeviceName = ""
    software_version: SoftwareVersion = ""


@dataclass
class ConverterMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.CONVERT
    convert_task_info: ConvertTaskInfo = field(default_factory=ConvertTaskInfo)
    input_model_info: InputModelInfo = field(default_factory=InputModelInfo)
    output_model_info: OutputModelInfo = field(default_factory=OutputModelInfo)
    available_options: List[DeviceInfo] = field(default_factory=lambda: [DeviceInfo()])
