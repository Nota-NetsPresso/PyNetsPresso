import shutil
import tempfile
import os
from typing import List, Union, Dict, Optional
from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import OmegaConf
from loguru import logger
from netspresso_trainer import train_with_yaml
from netspresso_trainer.cfg import (
    ModelConfig,
    AugmentationConfig,
    ScheduleConfig,
    LoggingConfig,
    EnvironmentConfig,
    DatasetConfig,
    LocalClassificationDatasetConfig,
    LocalDetectionDatasetConfig,
    LocalSegmentationDatasetConfig,
    ClassificationAugmentationConfig,
    DetectionAugmentationConfig,
    SegmentationAugmentationConfig,
    ClassificationScheduleConfig,
    DetectionScheduleConfig,
    SegmentationScheduleConfig,
)
from netspresso_trainer.cfg.data import PathConfig, ImageLabelPathConfig
from netspresso_trainer.cfg.augmentation import *

from .enums import Backbone, Head, Format, Task, SUPPORTED_MODELS


_DATA_CONFIG_TYPE_DICT = {
    "classification": LocalClassificationDatasetConfig,
    "detection": LocalDetectionDatasetConfig,
    "segmentation": LocalSegmentationDatasetConfig,
}

_AUGMENTATION_CONFIG_TYPE_DICT = {
    "classification": ClassificationAugmentationConfig,
    "detection": DetectionAugmentationConfig,
    "segmentation": SegmentationAugmentationConfig,
}

_TRAINING_CONFIG_TYPE_DICT = {
    "classification": ClassificationScheduleConfig,
    "detection": DetectionScheduleConfig,
    "segmentation": SegmentationScheduleConfig,
}


class TrainerConfigs:
    def __init__(
        self,
        data: Union[DatasetConfig, Path],
        augmentation: Union[AugmentationConfig, Path],
        model: Union[ModelConfig, Path],
        training: Union[ScheduleConfig, Path],
        logging: Union[LoggingConfig, Path],
        environment: Union[EnvironmentConfig, Path],
        prefix: str="temp_configs_",
    ):
        self.prefix = prefix
        self.create_temp_folder()
        self.data = self.process_config(data, "data")
        self.augmentation = self.process_config(augmentation, "augmentation")
        self.model = self.process_config(model, "model")
        self.training = self.process_config(training, "training")
        self.logging = self.process_config(logging, "logging")
        self.environment = self.process_config(environment, "environment")

    def create_temp_folder(self):
        self.temp_folder = tempfile.mkdtemp(prefix=self.prefix)

    def process_config(self, config, type):
        if is_dataclass(config):
            return self._save_config_as_yaml(config, type)
        else:
            return config

    def _save_config_as_yaml(self, config, type):
        config = OmegaConf.create({type: config})
        yaml_path = os.path.join(self.temp_folder, f"{type}.yaml")
        with open(yaml_path, 'w') as file:
            OmegaConf.save(config=config, f=yaml_path)

        return yaml_path


class ModelTrainer:
    def __init__(self, task: Task) -> None:
        self.task = task

    def set_dataset_config(
        self,
        name: str,
        root_path: str,
        train_image: str = "images/train",
        train_label: str = "labels/train",
        valid_image: str = "images/val",
        valid_label: str = "labels/val",
        id_mapping: Optional[Union[List[str], Dict[str, str]]] = None,
    ):
        common_config = {
            "name": name,
            "path": PathConfig(
                root=root_path,
                train=ImageLabelPathConfig(image=train_image, label=train_label),
                valid=ImageLabelPathConfig(image=valid_image, label=valid_label),
            ),
            "id_mapping": id_mapping,
        }
        data = _DATA_CONFIG_TYPE_DICT[self.task.value](**common_config)

        return data

    def set_model_config(self, backbone: Backbone, head: Head):
        config_key = (backbone, head)
        model = SUPPORTED_MODELS.get(config_key)
        available_heads = [key[1].name for key in SUPPORTED_MODELS.keys() if key[0] == backbone]

        if model is None:
            raise Exception(f"Unsupported head. Available heads are {available_heads}")

        return model

    def set_training_config(
        self,
        seed: int = 1,
        opt: str = "adamw",
        lr: float = 6e-5,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        sched: str = "cosine",
        min_lr: float = 1e-6,
        warmup_bias_lr: float = 1e-5,
        warmup_epochs: int = 5,
        iters_per_phase: int = 30,
        sched_power: float = 1.0,
        epochs: int = 3,
        batch_size: int = 8,
    ):
        schedule_config = ScheduleConfig(
            seed,
            opt,
            lr,
            momentum,
            weight_decay,
            sched,
            min_lr,
            warmup_bias_lr,
            warmup_epochs,
            iters_per_phase,
            sched_power,
            epochs,
            batch_size,
        )

        return schedule_config

    def set_augmentation_config(self, img_size, transforms, mix_transforms):
        augmentation_config = AugmentationConfig(
            img_size=img_size,
            transforms=transforms,
            mix_transforms=mix_transforms,
        )

        return augmentation_config

    def train(
        self,
        gpus: str,
        data: Union[DatasetConfig, Path],
        model: Union[ModelConfig, Path],
        training: Union[ScheduleConfig, Path] = None,
        augmentation: Union[AugmentationConfig, Path] = None,
        logging: Union[LoggingConfig, Path] = LoggingConfig(),
        environment: Union[EnvironmentConfig, Path] = EnvironmentConfig(),
    ):
        if training is None:
            training = _TRAINING_CONFIG_TYPE_DICT[self.task.value]()
        if augmentation is None:
            augmentation = _AUGMENTATION_CONFIG_TYPE_DICT[self.task.value]()
        
        configs = TrainerConfigs(data, augmentation, model, training, logging, environment)

        train_with_yaml(
            gpus=gpus,
            data=configs.data,
            augmentation=configs.augmentation,
            model=configs.model,
            training=configs.training,
            logging=configs.logging,
            environment=configs.environment,
        )

        # Remove temp config folder
        shutil.rmtree(configs.temp_folder, ignore_errors=True)
