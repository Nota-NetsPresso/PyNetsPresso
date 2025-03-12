from dataclasses import dataclass, field
from typing import List, Optional, Union

from netspresso.enums.base import StrEnum
from netspresso.enums.device import DisplaySoftwareVersion, HardwareType, SoftwareVersion
from netspresso.enums.model import DataType, Framework
from netspresso.metadata import common
from netspresso.metadata.common import AvailableOption


class Order(StrEnum):
    """ """

    DESC = "desc"
    ASC = "asc"


@dataclass
class AuthorizationHeader:
    Authorization: str

    def __init__(self, access_token):
        self.Authorization = f"Bearer {access_token}"


@dataclass
class UploadFile:
    files: List

    def __init__(self, file_name, file_content):
        self.files = [("file", (file_name, file_content))]


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

        device_info.software_versions.software_version = self.software_versions[0].software_version
        device_info.software_versions.display_software_versions = self.software_versions[0].display_software_version

        return device_info


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
