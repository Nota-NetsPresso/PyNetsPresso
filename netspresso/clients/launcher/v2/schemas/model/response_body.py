from dataclasses import dataclass, field
from typing import Optional, List

from netspresso.clients.launcher.v2.schemas.common import (
    ResponseItem,
    ResponsePaginationItems,
)
from netspresso.clients.launcher.v2.schemas.model import ModelBase, ModelStatus
from netspresso.enums import DataType


@dataclass
class DeviceInfo:
    device_name: str
    display_brand_name: str
    display_device_name: str


@dataclass
class ConvertOption:
    option_name: str
    framework: str
    target_framework: str
    data_type: DataType
    device: DeviceInfo = field(default_factory=DeviceInfo)
    software_version: Optional[str] = None
    display_software_version: Optional[str] = None


@dataclass
class BenchmarkOption:
    option_name: str
    framework: str
    device: DeviceInfo = field(default_factory=DeviceInfo)
    hardware_type: Optional[str] = None
    software_version: Optional[str] = None
    display_software_version: Optional[str] = None


@dataclass
class ModelUploadUrlData:
    """ """

    ai_model_id: str
    presigned_upload_url: str


@dataclass
class ResponseModelUploadUrl(ResponseItem):
    """ """

    data: Optional[ModelUploadUrlData] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ModelUploadUrlData(**self.data)


@dataclass
class ResponseModelStatus(ResponseItem):
    """ """

    data: ModelStatus

    def __post_init__(self):
        self.data = ModelStatus(**self.data)


@dataclass
class ResponseModelItem(ResponseItem):
    """ """

    data: ModelBase = field(default_factory=ModelBase)

    def __post_init__(self):
        self.data = ModelBase(**self.data)


@dataclass
class ResponseModelItems(ResponsePaginationItems):
    """ """

    data: Optional[List[ModelBase]] = field(default_factory=list)
