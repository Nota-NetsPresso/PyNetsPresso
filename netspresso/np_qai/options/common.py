from dataclasses import dataclass
from typing import List, Optional

from qai_hub.client import Device

from netspresso.enums.base import StrEnum


class ComputeUnit(StrEnum):
    ALL = "all"
    NPU = "npu"
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class CommonOptions:
    """
    Common options for all tasks.

    Args:
        compute_unit: Specifies the target compute unit(s)

    Note:
        For details, see [CommonOptions in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/api.html#common-options).
    """

    compute_unit: Optional[List[ComputeUnit]] = None

    def normalize_compute_units(self):
        if self.compute_unit is None:
            return None
        sorted_units = sorted(self.compute_unit)
        return ",".join(sorted_units)


def normalize_device_name(device: Optional[List[Device]] = None):
    if device is None:
        return None

    if not isinstance(device, list):
        return device.name

    sorted_devices = sorted([d.name for d in device])
    return ",".join(sorted_devices)
