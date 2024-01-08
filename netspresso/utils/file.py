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
    @staticmethod
    def check_exists(folder_path: str) -> bool:
        return Path(folder_path).exists()

    @staticmethod
    def create_folder(
        folder_path: str, parents: bool = True, exist_ok: bool = True
    ) -> None:
        Path(folder_path).mkdir(parents=parents, exist_ok=exist_ok)

    @staticmethod
    def create_file_path(
        folder_path: str, name: str, extension: str
    ) -> Union[str, Path]:
        return Path(folder_path) / (name + extension)

    @staticmethod
    def download_file(url: str, save_path: Union[str, Path]) -> None:
        request.urlretrieve(url, save_path)

    @staticmethod
    def get_extension_by_framework(framework: str) -> str:
        extension = FRAMEWORK_EXTENSION_MAP.get(framework)
        if extension is None:
            available_framework = [key for key in FRAMEWORK_EXTENSION_MAP.keys()]
            raise KeyError(
                f"The framework supports {available_framework}. The entered framework is {framework}."
            )
        return extension

    @staticmethod
    def prepare_model_path(
        folder_path: str, framework: str, is_folder_check: bool = True
    ) -> Tuple[Path, str]:
        default_model_path = Path(folder_path) / f"{Path(folder_path).name}.ext"
        extension = FileManager.get_extension_by_framework(framework=framework)

        if is_folder_check and not FileManager.check_exists(folder_path=folder_path):
            FileManager.create_folder(folder_path=folder_path)
        elif is_folder_check:
            sys.exit(f"This folder already exists. Local Path: {Path(folder_path)}")

        return default_model_path, extension
