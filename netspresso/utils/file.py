import json
import shutil
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


class FileHandler:
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
    def check_input_model_path(input_model_path: str):
        """Check if the input model path is a file.

        Args:
            input_model_path (str): The path to the input model file.

        Raises:
            FileNotFoundError: If the input model path is not a file.
        """

        if not Path(input_model_path).is_file():
            raise FileNotFoundError(
                "The input_model_path should be a file and cannot be a directory. Ex) ./model/sample_model.pt"
            )

    @staticmethod
    def create_folder(
        folder_path: str,
        parents: bool = True,
        exist_ok: bool = True,
        is_folder_check: bool = True,
    ) -> None:
        """Create a folder.

        Args:
            folder_path (str): The path to the folder to be created.
            parents (bool, optional): If True, also create parent directories if they don't exist.
            exist_ok (bool, optional): If False, raise an error if the folder already exists.
            is_folder_check (bool, optional): If True, check if the folder already exists.

        Raises:
            SystemExit: If the folder already exists and `exist_ok` is False.
        """
        if is_folder_check and not FileHandler.check_exists(folder_path=folder_path):
            Path(folder_path).mkdir(parents=parents, exist_ok=exist_ok)
        elif is_folder_check:
            sys.exit(f"This folder already exists. Local Path: {Path(folder_path)}")

    @staticmethod
    def create_unique_folder(folder_path: str) -> str:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        else:
            count = 1
            while True:
                new_folder_path = folder_path.with_name(f"{folder_path.name} ({count})")
                if not new_folder_path.exists():
                    new_folder_path.mkdir(parents=True)
                    folder_path = new_folder_path
                    break
                count += 1

        return str(folder_path)

    @staticmethod
    def create_file_path(folder_path: str, name: str, extension: str) -> Union[str, Path]:
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
            available_frameworks = list(FRAMEWORK_EXTENSION_MAP.keys())
            raise KeyError(f"The framework supports {available_frameworks}. The entered framework is {framework}.")
        return extension

    @staticmethod
    def get_path_and_extension(folder_path: str, framework: str) -> Tuple[Path, str]:
        """Prepare the model path by creating folders and generating a default model path.

        Args:
            folder_path (str): The base folder path.
            framework (str): The framework name.
            is_folder_check (bool, optional): If True, check if the folder exists before creating.

        Returns:
            Tuple[Path, str]: A tuple containing the default model path (Path) and the file extension (str).
        """
        default_model_path = Path(folder_path) / f"{Path(folder_path).name}.ext"
        extension = FileHandler.get_extension_by_framework(framework=framework)

        return default_model_path, extension

    @staticmethod
    def load_json(file_path: str):
        """Load JSON data from a file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Loaded JSON data.
        """

        with open(file_path, "r") as json_data:
            data = json.load(json_data)
        return data

    @staticmethod
    def move_and_cleanup_folders(source_folder: str, destination_folder: str):
        """Move files from the source folder to the destination folder and remove the source folder.

        Args:
            source_folder (str): The path to the source folder.
            destination_folder (str): The path to the destination folder.
        """

        source_folder = Path(source_folder)
        destination_folder = Path(destination_folder)

        for file_path in source_folder.iterdir():
            destination_path = destination_folder / file_path.name
            shutil.move(file_path, destination_path)

        source_folder.rmdir()

    @staticmethod
    def remove_folder(folder_path: str) -> None:
        """Remove a folder and its contents.

        Args:
            folder_path (str): Path to the folder.
        """

        folder_path = Path(folder_path)
        shutil.rmtree(folder_path, ignore_errors=True)
