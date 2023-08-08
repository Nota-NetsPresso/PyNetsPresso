from typing import List
from netspresso.launcher.schemas import DeviceName
from netspresso.launcher.schemas.model import TargetDevice

def filter_devices_with_device_name(name: DeviceName, devices: List[TargetDevice]) -> List[TargetDevice]:
    """Filter the devices with given device name.

        Args:
            name (DeviceName): The device name.
            devices (List[TargetDevice]): The list of devices
        Returns:
            List[TargetDevice]: filtered list of devices.
    """
    filtered_devices: List = []
    for device in devices:
        if device.device_name == name:
            filtered_devices.append(device)
    return filtered_devices

def filter_devices_with_device_software_version(software_version: str, devices: List[TargetDevice]) -> List[TargetDevice]:
    """Filter the devices with given software version.

        Args:
            software_version (DeviceName): The software version of device.
            devices (List[TargetDevice]): The list of devices
        Returns:
            List[TargetDevice]: filtered list of devices.
    """
    filtered_devices: List = []
    for device in devices:
        if device.device_name == software_version:
            filtered_devices.append(device)
    return filtered_devices