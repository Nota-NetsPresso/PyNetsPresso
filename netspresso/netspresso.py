import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import pkg_resources
import requests
from loguru import logger
from packaging import version

from netspresso.benchmarker import BenchmarkerV2
from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.tao import TAOTokenHandler
from netspresso.compressor import CompressorV2
from netspresso.converter import ConverterV2
from netspresso.enums import Task
from netspresso.exceptions.common import FailedFetchPackageException
from netspresso.inferencer.inferencer import CustomInferencer, NPInferencer
from netspresso.np_qai.benchmarker import NPQAIBenchmarker
from netspresso.np_qai.converter import NPQAIConverter
from netspresso.np_qai.quantizer import NPQAIQuantizer
from netspresso.quantizer import Quantizer
from netspresso.tao import TAOTrainer
from netspresso.trainer import Trainer
from netspresso.utils.file import FileHandler


class NetsPresso:
    def __init__(self, email: str, password: str, verify_ssl: bool = True, dev_mode: bool = False) -> None:
        """Initialize NetsPresso instance and perform user authentication.

        Args:
            email (str): User's email for authentication.
            password (str): User's password for authentication.
            verify_ssl (bool): Flag to indicate whether SSL certificates should be verified. Defaults to True.
            dev_mode (bool): If True, skip version check for development. Defaults to False.

        Raises:
            SystemExit: If the installed version is not the latest version and dev_mode is False.
        """
        self.dev_mode = dev_mode
        if not self._check_version():
            sys.exit(1)
        self.token_handler = TokenHandler(email=email, password=password, verify_ssl=verify_ssl)
        self.user_info = self.get_user()

    def _check_version(self) -> bool:
        """Check if the installed version of PyNetsPresso is the latest available version.

        Returns:
            bool: True if version check passes or dev_mode is True, False if newer version is available
        """
        if self.dev_mode:
            logger.info("Development mode enabled, skipping version check")
            return True

        try:
            # Get installed version
            current_version = pkg_resources.get_distribution('netspresso').version

            # Get latest version from PyPI
            latest_version = self.get_latest_version('netspresso')

            # Compare versions
            if version.parse(current_version) < version.parse(latest_version):
                logger.warning(
                    f"Your current version is {current_version}. The latest version {latest_version} is released."
                    f"Please upgrade via 'pip install --upgrade netspresso'"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to check PyNetsPresso version: {str(e)}")
            return False

    def get_latest_version(self, package_name):
        """Get the latest version of a package from PyPI.

        Args:
            package_name (str): Name of the package to check.

        Returns:
            str: Latest version number.

        Raises:
            FailedFetchPackageException: If unable to fetch package information.
        """
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["info"]["version"]
        else:
            raise FailedFetchPackageException(package_name=package_name, error_log=response.text)

    def get_user(self) -> UserResponse:
        """Get user information using the access token.

        Returns:
            UserInfo: User information.
        """
        user_info = auth_client.get_user_info(self.token_handler.tokens.access_token, self.token_handler.verify_ssl)
        return user_info

    def create_project(self, project_name: str, project_path: str = "./"):
        # Create the main project folder
        project_folder_path = Path(project_path) / project_name

        # Check if the project folder already exists
        if project_folder_path.exists():
            logger.info(f"Project '{project_name}' already exists at {project_folder_path.resolve()}.")
        else:
            project_folder_path.mkdir(parents=True, exist_ok=True)

            # Subfolder names
            subfolders = ["Trainer models", "Compressed models", "Pretrained models"]

            # Create subfolders
            for folder in subfolders:
                (project_folder_path / folder).mkdir(parents=True, exist_ok=True)

            # Create a metadata.json file
            metadata_file_path = project_folder_path / "metadata.json"
            metadata = {"is_project_folder": True}

            # Write metadata to the json file
            FileHandler.save_json(data=metadata, file_path=metadata_file_path)

            logger.info(f"Project '{project_name}' created at {project_folder_path.resolve()}.")

    def trainer(self, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None) -> Trainer:
        """Initialize and return a Trainer instance.

        Args:
            task (Union[str, Task], optional): Type of task (classification, detection, segmentation).
            yaml_path (str, optional): Path to the YAML configuration file.

        Returns:
            Trainer: Initialized Trainer instance.
        """
        return Trainer(token_handler=self.token_handler, task=task, yaml_path=yaml_path)

    def compressor_v2(self) -> CompressorV2:
        """Initialize and return a Compressor instance.

        Returns:
            Compressor: Initialized Compressor instance.
        """
        return CompressorV2(token_handler=self.token_handler)

    def converter_v2(self) -> ConverterV2:
        """Initialize and return a Converter instance.

        Returns:
            Converter: Initialized Converter instance.
        """
        return ConverterV2(token_handler=self.token_handler, user_info=self.user_info)

    def quantizer(self) -> Quantizer:
        """Initialize and return a Quantizer instance.

        Returns:
            Quantizer: Initialized Quantizer instance.
        """
        return Quantizer(token_handler=self.token_handler, user_info=self.user_info)

    def benchmarker_v2(self) -> BenchmarkerV2:
        """Initialize and return a Benchmarker instance.

        Returns:
            Benchmarker: Initialized Benchmarker instance.
        """
        return BenchmarkerV2(token_handler=self.token_handler, user_info=self.user_info)

    def np_inferencer(self, config_path: str, input_model_path: str) -> NPInferencer:
        """Initialize and return a Inferencer instance.

        Returns:
            Inferencer: Initialized Inferencer instance.
        """

        return NPInferencer(config_path=config_path, input_model_path=input_model_path)

    def custom_inferencer(self, input_model_path: str) -> CustomInferencer:
        """Initialize and return a Inferencer instance.

        Returns:
            Inferencer: Initialized Inferencer instance.
        """
        return CustomInferencer(input_model_path=input_model_path)


class TAO:
    def __init__(self, ngc_api_key: str) -> None:
        """Initialize TAO instance and perform user authentication.

        Args:
            ngc_api_key (str): API key for TAO authentication.
        """
        self.ngc_api_key = ngc_api_key
        self.token_handler = TAOTokenHandler(ngc_api_key=ngc_api_key)

    def trainer(self) -> TAOTrainer:
        """Initialize and return a Trainer instance.

        Returns:
            TAO: Initialized Trainer instance.
        """
        return TAOTrainer(token_handler=self.token_handler)


class NPQAI:
    def __init__(self, api_token: str) -> None:
        # Define the command and arguments
        command = "qai-hub"
        args = ["configure", "--api_token", f"{api_token}"]

        # Execute the command
        result = subprocess.run([command] + args, capture_output=True, text=True)
        logger.info(result)

    def converter(self) -> NPQAIConverter:
        """Initialize and return a Converter instance.

        Returns:
            NPQAIConverter: Initialized Converter instance.
        """
        return NPQAIConverter()

    def benchmarker(self) -> NPQAIBenchmarker:
        """Initialize and return a Benchmarker instance.

        Returns:
            NPQAIBenchmarker: Initialized Benchmarker instance.
        """
        return NPQAIBenchmarker()

    def quantizer(self) -> NPQAIQuantizer:
        """Initialize and return a Quantizer instance.

        Returns:
            NPQAIQuantizer: Initialized Quantizer instance.
        """
        return NPQAIQuantizer()
