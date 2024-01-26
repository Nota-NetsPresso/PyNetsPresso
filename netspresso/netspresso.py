from pathlib import Path
from typing import Optional, Union

from netspresso.trainer import Trainer
from netspresso.compressor import Compressor
from netspresso.converter import Converter
from netspresso.benchmarker import Benchmarker
from .enums import Task

from netspresso.clients.auth import auth_client


class NetsPresso:
    def __init__(self, email: str, password: str) -> None:
        self.tokens = self.login(email, password)
        self.user_info = self.get_user()

    def login(self, email: str, password: str) -> None:
        tokens = auth_client.login(email, password)
        return tokens

    def get_user(self):
        user_info = auth_client.get_user_info(self.tokens.access_token)
        return user_info

    def trainer(self, task: Optional[Task] = None, yaml_path: Optional[Union[Path, str]] = None):
        return Trainer(task=task, yaml_path=yaml_path)

    def compressor(self):
        return Compressor(tokens=self.tokens)

    def converter(self):
        return Converter(tokens=self.tokens, user_info=self.user_info)

    def benchmarker(self):
        return Benchmarker(tokens=self.tokens, user_info=self.user_info)
