from pathlib import Path
from typing import Union
from urllib import request


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
