import tempfile
from dataclasses import is_dataclass
from pathlib import Path
from typing import Union

from netspresso_trainer.cfg import (
    AugmentationConfig,
    DatasetConfig,
    EnvironmentConfig,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
)
from omegaconf import OmegaConf


class TrainerConfigs:
    def __init__(
        self,
        data: Union[DatasetConfig, Path],
        augmentation: Union[AugmentationConfig, Path],
        model: Union[ModelConfig, Path],
        training: Union[ScheduleConfig, Path],
        logging: Union[LoggingConfig, Path],
        environment: Union[EnvironmentConfig, Path],
        prefix: str = "temp_np_trainer_configs_",
    ) -> None:
        """Initialize TrainerConfigs with specified configurations.

        Args:
            data (Union[DatasetConfig, Path]): Dataset configuration or path to a YAML file.
            augmentation (Union[AugmentationConfig, Path]): Augmentation configuration or path to a YAML file.
            model (Union[ModelConfig, Path]): Model configuration or path to a YAML file.
            training (Union[ScheduleConfig, Path]): Training schedule configuration or path to a YAML file.
            logging (Union[LoggingConfig, Path]): Logging configuration or path to a YAML file.
            environment (Union[EnvironmentConfig, Path]): Environment configuration or path to a YAML file.
            prefix (str, optional): Prefix for the temporary folder. Defaults to "temp_np_trainer_configs_".
        """

        self.prefix = prefix
        self.temp_folder = self.create_temp_folder()
        self.data = self.handle_config(data, "data")
        self.augmentation = self.handle_config(augmentation, "augmentation")
        self.model = self.handle_config(model, "model")
        self.training = self.handle_config(training, "training")
        self.logging = self.handle_config(logging, "logging")
        self.environment = self.handle_config(environment, "environment")

    def create_temp_folder(self) -> str:
        """Create a temporary folder and return its path.

        Returns:
            str: Path to the created temporary folder.
        """

        return tempfile.mkdtemp(prefix=self.prefix)

    def _save_config_as_yaml(self, config, type: str) -> Path:
        """Save the given configuration as a YAML file in the temporary folder.

        Args:
            config: Configuration to be saved.
            type (str): Type of the configuration.

        Returns:
            Path: Path to the saved YAML file.
        """

        config = OmegaConf.create({type: config})
        yaml_path = Path(self.temp_folder) / f"{type}.yaml"
        OmegaConf.save(config=config, f=yaml_path)

        return yaml_path

    def handle_config(self, config, type: str) -> Path:
        """Handle the input configuration, either saving it as a YAML file or using YAML file path directly.

        Args:
            config: Input configuration.
            type (str): Type of the configuration.

        Returns:
            Path: Path to the saved YAML file if the input is a dataclass; otherwise, the YAML file path itself.
        """

        if is_dataclass(config):
            return self._save_config_as_yaml(config, type)
        else:
            return config
