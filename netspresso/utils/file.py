import json
import sys
from pathlib import Path
from typing import Tuple, Union
from urllib import request

FRAMEWORK_EXTENSION_MAP = {
    "tensorflow_keras": ".h5",
    "pytorch": ".pt",
    "onnx": ".onnx",
    "tensorflow_lite": ".tflite",
    "drpai": ".zip",
    "openvino": ".zip",
    "tensorrt": ".trt",
}


class FileManager:
    """Utility class for file-related operations."""

    @staticmethod
    def check_exists(folder_path: str) -> bool:
        """Check if the file or folder exists.

        Args:
            folder_path (str): The path to the folder.

        Returns:
            bool: True if the file or folder exists, False otherwise
        """
        return Path(folder_path).exists()

    @staticmethod
    def create_folder(
        folder_path: str, parents: bool = True, exist_ok: bool = True, is_folder_check: bool = True,
    ) -> None:
        """Create a folder.

        Args:
            folder_path (str): The path to the folder to be created.
            parents (bool, optional): If True, also create parent directories if they don't exist.
            exist_ok (bool, optional): If False, raise an error if the folder already exists.
        
        Returns:
            None
        """
        if is_folder_check and not FileManager.check_exists(folder_path=folder_path):
            Path(folder_path).mkdir(parents=parents, exist_ok=exist_ok)
        elif is_folder_check:
            sys.exit(f"This folder already exists. Local Path: {Path(folder_path)}")

    @staticmethod
    def create_file_path(
        folder_path: str, name: str, extension: str
    ) -> Union[str, Path]:
        """Create a file path.

        Args:
            folder_path (str): The path to the folder where the file will be located.
            name (str): The name of the file.
            extension (str): The file extension.

        Returns:
            Union[str, Path]: The full path to the created file.
        """
        return Path(folder_path) / (name + extension)

    @staticmethod
    def download_file(url: str, save_path: Union[str, Path]) -> None:
        """Download a file from the given URL and save it to the specified path.

        Args:
            url (str): The URL of the file to be downloaded.
            save_path (Union[str, Path]): The path where the downloaded file will be saved.
        
        Returns:
            None
        """
        request.urlretrieve(url, save_path)

    @staticmethod
    def get_extension_by_framework(framework: str) -> str:
        """Get the file extension based on the given framework.

        Args:
            framework (str): The framework name.

        Raises:
            KeyError: If the framework is not found in the extension map.

        Returns:
            str: The file extension corresponding to the framework.
        """
        extension = FRAMEWORK_EXTENSION_MAP.get(framework)
        if extension is None:
            available_frameworks = [key for key in FRAMEWORK_EXTENSION_MAP.keys()]
            raise KeyError(
                f"The framework supports {available_frameworks}. The entered framework is {framework}."
            )
        return extension

    @staticmethod
    def prepare_model_path(
        folder_path: str, framework: str, is_folder_check: bool = True
    ) -> Tuple[Path, str]:
        """Prepare the model path by creating folders and generating a default model path.

        Args:
            folder_path (str): The base folder path.
            framework (str): The framework name.
            is_folder_check (bool, optional): If True, check if the folder exists before creating.

        Returns:
            Tuple[Path, str]: A tuple containing the default model path (Path) and the file extension (str).
        """
        default_model_path = Path(folder_path) / f"{Path(folder_path).name}.ext"
        extension = FileManager.get_extension_by_framework(framework=framework)

        FileManager.create_folder(folder_path=folder_path, is_folder_check=is_folder_check)

        return default_model_path, extension

    @staticmethod
    def load_json(file_path: str):
        with open(file_path, "r") as json_data:
            data = json.load(json_data)
        return data
