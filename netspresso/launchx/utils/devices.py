from netspresso.launchx.schemas import ModelFramework, DeviceName
from netspresso.launchx.schemas.model import TargetDevice

def filter_devices_with_device_name(name: DeviceName, devices: list[TargetDevice]) -> list[TargetDevice]:
    filtered_devices = (device for device in devices if device.device_name is name)
    return list(filtered_devices)