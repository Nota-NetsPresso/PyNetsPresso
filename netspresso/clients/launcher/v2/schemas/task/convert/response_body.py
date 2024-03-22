import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List

from netspresso.clients.launcher.v2.enums import TaskStatus
from netspresso.clients.launcher.v2.schemas.task.common import Device, TaskStatusInfo
from netspresso.clients.launcher.v2.schemas import (
    InputLayer,
    ResponseItem,
    ResponseItems,
)
from netspresso.enums import DataType


@dataclass
class ConvertTask:
    convert_task_id: str
    input_model_id: str
    output_model_id: str
    target_framework: str
    target_device_name: str
    display_brand_name: str
    display_device_name: str
    data_type: DataType
    input_layer: InputLayer
    status: TaskStatus
    software_version: str = None
    display_software_version: str = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseConvertTaskItem(ResponseItem):
    data: Optional[ConvertTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ConvertTask(**self.data)


@dataclass(init=False)
class ConvertOption:
    option_id: str
    option_name: str
    framework: str
    device: Device
    sw_version: Optional[str] = ""

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class ResponseConvertOptionItems(ResponseItems):
    data: List[Optional[ConvertOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ConvertOption(**item) for item in self.data]


@dataclass
class ResponseConvertStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)
