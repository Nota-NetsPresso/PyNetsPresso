from pathlib import Path
from typing import Optional, Union

from netspresso.trainer import Trainer
from netspresso.compressor import Compressor
from netspresso.converter import Converter
from netspresso.benchmarker import Benchmarker
from netspresso.enums import Task
from netspresso.clients.auth import auth_client, TokenHandler
from netspresso.clients.auth.schemas import UserInfo, Tokens


class NetsPresso:
    def __init__(self, email: str, password: str) -> None:
        """Initialize NetsPresso instance and perform user authentication.

        Args:
            email (str): User's email for authentication.
            password (str): User's password for authentication.
        """
        tokens = self.login(email, password)
        self.token_handler = TokenHandler(tokens=tokens)
        self.user_info = self.get_user()

    def login(self, email: str, password: str) -> Tokens:
        """Perform user login and retrieve authentication tokens.

        Args:
            email (str): User's email for login.
            password (str): User's password for login.

        Returns:
            Tokens: Authentication tokens containing access and refresh tokens.
        """
        tokens = auth_client.login(email, password)
        return tokens

    def get_user(self) -> UserInfo:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(self.token_handler.tokens.access_token)
        return user_info

    def trainer(self, task: Optional[Union[str, Task]] = None, yaml_path: Optional[Union[Path, str]] = None) -> Trainer:
        """Initialize and return a Trainer instance.

        Args:
            task (Optional[Union[str, Task]]): Type of task (classification, detection, segmentation).
            yaml_path (Optional[Union[Path, str]]): Path to the YAML configuration file.

        Returns:
            Trainer: Initialized Trainer instance.
        """
        return Trainer(task=task, yaml_path=yaml_path)

    def compressor(self) -> Compressor:
        """Initialize and return a Compressor instance.

        Returns:
            Compressor: Initialized Compressor instance.
        """
        return Compressor(token_handler=self.token_handler)

    def converter(self) -> Converter:
        """Initialize and return a Converter instance.

        Returns:
            Converter: Initialized Converter instance.
        """
        return Converter(token_handler=self.token_handler, user_info=self.user_info)

    def benchmarker(self) -> Benchmarker:
        """Initialize and return a Benchmarker instance.

        Returns:
            Benchmarker: Initialized Benchmarker instance.
        """
        return Benchmarker(token_handler=self.token_handler, user_info=self.user_info)
