from dataclasses import dataclass, field
from typing import List, Dict

from netspresso_v2.metadata.enums import Status, TaskType


@dataclass
class InputShape:
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class ModelInfo:
    data_type: str = ""
    framework: str = ""
    input_shapes: List[InputShape] = field(default_factory=lambda: [InputShape()])


@dataclass
class ConvertInfo:
    target_framework: str = ""
    target_device_name: str = ""
    data_type: str = ""
    software_version: str = ""
    model_file_name: str = ""
    convert_task_uuid: str = ""
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class ConverterMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.CONVERT
    converted_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    convert_info: ConvertInfo = field(default_factory=ConvertInfo)
    available_devices: List[Dict] = field(default_factory=list)
