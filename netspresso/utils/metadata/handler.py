import json
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from netspresso.metadata.common import BaseMetadata
from netspresso.metadata.profiler import ProfilerMetadata


class MetadataHandler:
    @staticmethod
    def save_metadata(data: BaseMetadata, folder_path: str, file_name: str = "metadata") -> None:
        """Save metadata to a JSON file.

        Args:
            data (BaseMetadata): The metadata object to be saved. It should be an instance of BaseMetadata or its subclass.
            folder_path (str): The directory path where the JSON file will be saved.
            file_name (str): The name of the JSON file to be created (without extension). Defaults to "metadata".

        Returns:
            None: This function does not return any value.

        Notes:
            The data is converted to a dictionary using the `asdict` method before saving.
        """
        MetadataHandler.save_json(data.asdict(), folder_path, file_name)

    @staticmethod
    def save_benchmark_result(data: List[ProfilerMetadata], folder_path: str, file_name: str = "profile") -> None:
        """Save a list of benchmark metadata objects to a JSON file.

        Args:
            data (List[ProfilerMetadata]): A list of ProfilerMetadata objects to be saved. Each object is converted to a dictionary using the `asdict` method.
            folder_path (str): The directory path where the JSON file will be saved.
            file_name (str): The name of the JSON file to be created (without extension). Defaults to "benchmark".

        Returns:
            None: This function does not return any value.

        Note
            Each item in the `data` list is checked to ensure it is an instance of ProfilerMetadata. If it is, it is converted to a dictionary before saving.
        """
        data = [_data.asdict() if isinstance(_data, ProfilerMetadata) else _data for _data in data]
        MetadataHandler.save_json(data, folder_path, file_name)

    def save_json(data: Union[Dict, List[Dict]], folder_path: str, file_name: str) -> None:
        """Save data to a JSON file.

        Args:
            data (Union[Dict, List[Dict]]): The data to be saved. This can be a dictionary or a list of dictionaries.
            folder_path (str): The directory path where the JSON file will be saved.
            file_name (str): The name of the JSON file to be created (without extension).

        Returns:
            None: This function does not return any value.

        Notes:
            The data is written to a JSON file with indentation for readability. The file is saved in the specified directory with the given name.
        """
        file_path = Path(folder_path) / f"{file_name}.json"

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"JSON file saved at {file_path.resolve()}")

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON data from a file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: Loaded dictionary data.
        """
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data
