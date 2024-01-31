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

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_status(self, status: Status) -> None:
        self.status = status

    def update_model_info(self, data_type: str, framework: str, input_shape: Any) -> None:
        self.model_info.data_type = data_type
        self.model_info.framework = framework
        self.model_info.input_shapes = [
            InputShape(
                batch=input_shape.batch,
                channel=input_shape.channel,
                dimension=[int(x.strip()) for x in input_shape.input_size.split(",")],
            )
        ]

    def update_convert_info(
        self,
        target_framework: str,
        target_device_name: str,
        data_type: str,
        software_version: str,
        model_file_name: str,
        convert_task_uuid: str,
        input_model_uuid: str,
        output_model_uuid: str,
    ) -> None:
        self.convert_info.target_framework = target_framework
        self.convert_info.target_device_name = target_device_name
        self.convert_info.data_type = data_type
        self.convert_info.software_version = software_version
        self.convert_info.model_file_name = model_file_name
        self.convert_info.convert_task_uuid = convert_task_uuid
        self.convert_info.input_model_uuid = input_model_uuid
        self.convert_info.output_model_uuid = output_model_uuid

    def update_converted_model_path(self, converted_model_path):
        self.converted_model_path = converted_model_path

    def update_available_devices(self, available_devices: List[Dict]) -> None:
        self.available_devices = [
            TargetDevice(
                display_name=device.display_name,
                display_brand_name=device.display_brand_name,
                device_name=device.device_name,
                software_version=device.software_version,
                software_version_display_name=device.software_version_display_name,
                hardware_type=device.hardware_type,
            )
            for device in available_devices
        ]
