from netspresso.launchx.schemas import DeviceName
from netspresso.launchx.schemas.model import TargetDevice

def filter_devices_with_device_name(name: DeviceName, devices: list[TargetDevice]) -> list[TargetDevice]:
    filtered_devices = (device for device in devices if device.device_name is name)
    return list(filtered_devices)

def filter_devices_with_device_software_version(software_version: str, devices: list[TargetDevice]) -> list[TargetDevice]:
    filtered_devices = (device for device in devices if device.software_version is software_version)
    return list(filtered_devices)