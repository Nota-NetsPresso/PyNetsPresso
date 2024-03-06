from typing import Optional, Union


from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.enums import Task
from netspresso.modules.benchmarker import Benchmarker
from netspresso.modules.compressor import Compressor
from netspresso.modules.converter import Converter
from netspresso.modules.tao import TAOTrainer
from netspresso.modules.trainer import Trainer


class NetsPresso:
    def __init__(self, email: str, password: str, verify_ssl: bool = True) -> None:
        """Initialize NetsPresso instance and perform user authentication.

        Args:
            email (str): User's email for authentication.
            password (str): User's password for authentication.
            verify_ssl (bool): Flag to indicate whether SSL certificates should be verified. Defaults to True.
        """
        self.token_handler = TokenHandler(email=email, password=password, verify_ssl=verify_ssl)
        self.user_info = self.get_user()

    def get_user(self) -> UserResponse:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(self.token_handler.tokens.access_token, self.token_handler.verify_ssl)
        return user_info

    def trainer(self, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None) -> Trainer:
        """Initialize and return a Trainer instance.

        Args:
            task (Union[str, Task], optional): Type of task (classification, detection, segmentation).
            yaml_path (str, optional): Path to the YAML configuration file.

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

    def tao_trainer(self, ngc_api_key: str) -> TAOTrainer:
        """Initialize and return a TAO instance.

        Args:
            ngc_api_key (str): API key for TAO authentication.

        Returns:
            TAO: Initialized TAO instance.
        """
        return TAOTrainer(ngc_api_key=ngc_api_key)
