from dataclasses import dataclass, field
from typing import List

from netspresso_v2.metadata.enums import (
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
