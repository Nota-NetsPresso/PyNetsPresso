from dataclasses import dataclass

from netspresso.clients.launcher.v2.enums import TaskStatus


@dataclass
class Device:
    device_brand: str
    device_name: str


@dataclass
class TaskStatusInfo:
    status: TaskStatus
