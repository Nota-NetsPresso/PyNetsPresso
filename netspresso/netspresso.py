from pathlib import Path
from typing import Optional, Union

from netspresso.trainer import Trainer
from netspresso.compressor import Compressor
from netspresso.converter import Converter
from netspresso.benchmarker import Benchmarker
from .enums import Task

from netspresso.clients.auth import AuthClient


class NetsPresso:
    def __init__(self, email: str, password: str) -> None:
        self.auth = AuthClient()
        self.login(email, password)

    def login(self, email: str, password: str) -> None:
        self.auth.login(email, password)

    def trainer(self, task: Optional[Task] = None, yaml_path: Optional[Union[Path, str]] = None):
        return Trainer(task=task, yaml_path=yaml_path)

    def compressor(self):
        return Compressor(self.auth)

    def converter(self):
        return Converter(self.auth)

    def benchmarker(self):
        return Benchmarker(self.auth)
