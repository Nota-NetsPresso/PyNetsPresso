from pathlib import Path
import tempfile
from typing import Union
from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import OmegaConf
from netspresso_trainer.cfg import (
    ModelConfig,
    AugmentationConfig,
    ScheduleConfig,
    LoggingConfig,
    EnvironmentConfig,
    DatasetConfig,
)


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
    ):
        self.prefix = prefix
        self.create_temp_folder()
        self.data = self.handle_config(data, "data")
        self.augmentation = self.handle_config(augmentation, "augmentation")
        self.model = self.handle_config(model, "model")
        self.training = self.handle_config(training, "training")
        self.logging = self.handle_config(logging, "logging")
        self.environment = self.handle_config(environment, "environment")

    def create_temp_folder(self):
        self.temp_folder = tempfile.mkdtemp(prefix=self.prefix)

    def _save_config_as_yaml(self, config, type):
        config = OmegaConf.create({type: config})
        yaml_path = Path(self.temp_folder) / f"{type}.yaml"
        OmegaConf.save(config=config, f=yaml_path)

        return yaml_path

    def handle_config(self, config, type):
        if is_dataclass(config):
            return self._save_config_as_yaml(config, type)
        else:
            return config
