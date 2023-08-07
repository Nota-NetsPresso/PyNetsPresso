from netspresso.launcher.schemas import DeviceName
from netspresso.launcher.schemas.model import TargetDevice

def filter_devices_with_device_name(name: DeviceName, devices: list[TargetDevice]) -> list[TargetDevice]:
    """Filter the devices with given device name.

        Args:
            name (DeviceName): The device name.
            devices (list[TargetDevice]): The list of devices
        Returns:
            list[TargetDevice]: filtered list of devices.
    """
    filtered_devices = (device for device in devices if device.device_name is name)
    return list(filtered_devices)

def filter_devices_with_device_software_version(software_version: str, devices: list[TargetDevice]) -> list[TargetDevice]:
    """Filter the devices with given software version.

        Args:
            software_version (DeviceName): The software version of device.
            devices (list[TargetDevice]): The list of devices
        Returns:
            list[TargetDevice]: filtered list of devices.
    """
    filtered_devices = (device for device in devices if device.software_version is software_version)
    return list(filtered_devices)