from dataclasses import dataclass
from typing import Optional

from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.enums import DeviceName, SoftwareVersion, HardwareType


@dataclass
class RequestBenchmark:
    input_model_id: str
    target_device_name: DeviceName
    software_version: Optional[SoftwareVersion] = ""
    hardware_type: Optional[HardwareType] = None
    input_layer: Optional[InputLayer] = None
