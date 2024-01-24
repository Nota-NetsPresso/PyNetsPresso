from dataclasses import dataclass


@dataclass
class TargetDevice:
    display_name: str = ""
    display_brand_name: str = ""
    device_name: str = ""
    software_version: str = ""
    software_version_display_name: str = ""
    hardware_type: str = ""
