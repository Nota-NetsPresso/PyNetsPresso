import json
from pathlib import Path
from typing import Any, Dict

from netspresso.enums import TaskType

from .default import BenchmarkerMetadata, CompressorMetadata, ConverterMetadata, TrainerMetadata


class MetadataHandler:
    @staticmethod
    def save_json(data: dict, folder_path: str, file_name: str = "metadata") -> None:
        """Save dictionary data to a JSON file.

        Args:
            data (dict): The dictionary data to be saved.
            file_path (str): The path to the JSON file.

        Returns:
            None
        """
        file_path = Path(folder_path) / f"{file_name}.json"

        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

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

    @staticmethod
    def get_default_metadata(task_type: TaskType) -> Any:
        """Get the default metadata object based on the task type.

        Args:
            task_type (TaskType): The type of the task.

        Returns:
            Any: Default metadata object.
        """
        if task_type == TaskType.TRAIN:
            _metadata = TrainerMetadata()
        elif task_type == TaskType.COMPRESS:
            _metadata = CompressorMetadata()
        elif task_type == TaskType.CONVERT:
            _metadata = ConverterMetadata()
        elif task_type == TaskType.BENCHMARK:
            _metadata = BenchmarkerMetadata()
        return _metadata

    @staticmethod
    def init_metadata(folder_path: str, task_type: TaskType, file_name: str = "metadata") -> Any:
        """Initialize metadata by saving default metadata to a JSON file.

        Args:
            folder_path (str): The path to the folder where metadata will be stored.
            task_type (TaskType): The type of the task.

        Returns:
            Any: Default metadata object.
        """
        default_metadata = MetadataHandler.get_default_metadata(task_type)
        MetadataHandler.save_json(default_metadata.asdict(), folder_path, file_name=file_name)

        return default_metadata
