from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from netspresso.clients.utils.system import ENV_STR
from netspresso.enums import DataType, DisplaySoftwareVersion, Framework, HardwareType, SoftwareVersion
from netspresso.enums.base import StrEnum
from netspresso.metadata import common
from netspresso.metadata.common import AvailableOption, SoftwareVersions

version = (Path(__file__).parent.parent.parent.parent.parent / "VERSION").read_text().strip()


class Order(StrEnum):
    """ """

    DESC = "desc"
    ASC = "asc"


@dataclass
class AuthorizationHeader:
    Authorization: str

    def __init__(self, access_token):
        self.Authorization = f"Bearer {access_token}"

    def to_dict(self):
        return {"Authorization": self.Authorization, "User-Agent": f"NetsPresso Python Package v{version} ({ENV_STR})"}


@dataclass
class UploadFile:
    files: List

    def __init__(self, file_name, file_content):
        self.files = [("file", (file_name, file_content))]


@dataclass
class UploadDataset:
    files: List

    def __init__(self, file_name, file_content):
        self.files = [("dataset", (file_name, file_content))]


@dataclass
class RequestPagination:
    """ """

    start: int = 0
    size: int = 10
    order: Order = Order.DESC.value
    paging: bool = True


@dataclass
class ResponseItem:
    """ """

    data: Optional[object] = field(default_factory=dict)


@dataclass
class ResponseItems:
    """ """

    data: List[Optional[object]] = field(default_factory=list)


@dataclass
class ResponsePaginationItems:
    """ """

    result_count: int
    total_count: int
    data: List[Optional[object]] = field(default_factory=list)


@dataclass
class SoftwareVersionInfo:
    """ """

    software_version: Optional[Union[None, SoftwareVersion]] = None
    display_software_version: Optional[Union[None, DisplaySoftwareVersion]] = None

    def to(self) -> SoftwareVersions:
        software_version = SoftwareVersions()

        software_version.software_version = self.software_version
        software_version.display_software_versions = self.display_software_version

        return software_version


@dataclass
class TaskInfo:
    """ """

    device_name: str
    display_brand_name: str
    display_device_name: str
    software_version: Optional[SoftwareVersion]
    display_software_version: Optional[DisplaySoftwareVersion]
    data_type: DataType
    hardware_type: Optional[HardwareType]


@dataclass
class DeviceInfo:
    """ """

    device_name: str
    display_device_name: str
    display_brand_name: str
    software_versions: Optional[List[SoftwareVersionInfo]] = field(default_factory=list)
    data_types: Optional[List[DataType]] = field(default_factory=list)
    hardware_types: Optional[List[HardwareType]] = field(default_factory=list)

    def __post_init__(self):
        self.software_versions = [SoftwareVersionInfo(**item) for item in self.software_versions]

    def to(self) -> common.DeviceInfo:
        device_info = common.DeviceInfo()

        device_info.device_name = self.device_name
        device_info.display_device_name = self.display_device_name
        device_info.display_brand_name = self.display_brand_name
        device_info.data_types = self.data_types
        device_info.hardware_types = self.hardware_types

        device_info.software_versions = [item.to() for item in self.software_versions]

        return device_info


@dataclass
class TaskOption:
    """ """

    framework: Optional[Framework] = ""
    display_framework: Optional[str] = ""
    target_device: Optional[TaskInfo] = field(default_factory=dict)

    def __post_init__(self):
        self.target_device = TaskInfo(**self.target_device)


@dataclass
class ModelOption:
    """ """

    framework: Optional[Framework] = ""
    display_framework: Optional[str] = ""
    devices: List[DeviceInfo] = field(default_factory=list)

    def __post_init__(self):
        self.devices = [DeviceInfo(**item) for item in self.devices]

    def to(self) -> AvailableOption:
        available_options = AvailableOption()
        available_options.framework = self.framework
        available_options.display_framework = self.display_framework
        for device in self.devices:
            available_options.devices.append(device.to())

        return available_options
