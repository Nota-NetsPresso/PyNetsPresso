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
    PathConfig,
    ImageLabelPathConfig,
    LocalClassificationDatasetConfig,
    LocalDetectionDatasetConfig,
    LocalSegmentationDatasetConfig,
)

from .enums.model import Task, Backbone, Head, SUPPORTED_MODELS
from .enums.data import Format


class TrainerConfigs:
    def __init__(
        self,
        data: Union[DatasetConfig, Path, str],
        augmentation: Union[AugmentationConfig, Path, str],
        model: Union[ModelConfig, Path, str],
        training: Union[ScheduleConfig, Path, str],
        logging: Union[LoggingConfig, Path, str],
        environment: Union[EnvironmentConfig, Path, str],
    ) -> None:
        self.data = self._export_config_as_yaml(data, "data.yaml")
        self.augmentation = self._export_config_as_yaml(augmentation, "augmentation.yaml")
        self.model = self._export_config_as_yaml(model, "model.yaml")
        self.training = self._export_config_as_yaml(training, "training.yaml")
        self.logging = self._export_config_as_yaml(logging, "logging.yaml")
        self.environment = self._export_config_as_yaml(environment, "environment.yaml")

    def _export_config_as_yaml(self, conf, yaml_path):
        if is_dataclass(conf):
            conf = OmegaConf.create(conf)
            yaml_path = Path.cwd() / "temp_config" / yaml_path
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Save in {yaml_path}")
            OmegaConf.save(config=conf, f=yaml_path)
            return str(yaml_path)
        else:
            return str(conf)


class ModelTrainer:
    def __init__(self) -> None:
        pass

    def get_dataset_config(
        self,
        task: Task,
        format: Format,
        name: str,
        root_path: str,
        id_mapping: Optional[Union[List[str], Dict[str, str]]],
    ):
        common_config = {
            "name": name,
            "path": PathConfig(
                root=root_path,
                train=ImageLabelPathConfig(image="images/train", label="labels/train"),
                valid=ImageLabelPathConfig(image="images/val", label="labels/val"),
            ),
            "id_mapping": id_mapping,
        }

        if task == Task.Image_Classification:
            return LocalClassificationDatasetConfig(**common_config)
        elif task == Task.Object_Detection:
            return LocalDetectionDatasetConfig(**common_config)
        elif task == Task.Semantic_Segmentation:
            return LocalSegmentationDatasetConfig(**common_config)

    def _filter_heads_by_backbone(self, backbone: Backbone):
        return [key[1].name for key in SUPPORTED_MODELS.keys() if key[0] == backbone]

    def get_model_config(self, backbone: Backbone, head: Head):
        config_key = (backbone, head)
        model = SUPPORTED_MODELS.get(config_key)
        available_heads = self._filter_heads_by_backbone(backbone=backbone)

        if model is None:
            raise Exception(f"Unsupported head. Available heads are {available_heads}")

        return model

    def get_training_config(
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

    def get_augmentation_config(self, img_size, transforms, mix_transforms):
        augmentation_config = AugmentationConfig(
            img_size=img_size,
            transforms=transforms,
            mix_transforms=mix_transforms,
        )

        return augmentation_config

    def train(
        self,
        gpus: str,
        data: Union[DatasetConfig, Path, str],
        model: Union[ModelConfig, Path, str],
        augmentation: Union[AugmentationConfig, Path, str] = None,
        training: Union[ScheduleConfig, Path, str] = None,
        logging: Union[LoggingConfig, Path, str] = LoggingConfig(),
        environment: Union[EnvironmentConfig, Path, str] = EnvironmentConfig(),
    ):
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
