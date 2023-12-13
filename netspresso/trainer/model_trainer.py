from pathlib import Path
from typing import List, Union, Dict, Optional
import shutil

from loguru import logger
from netspresso_trainer import train_with_yaml
from netspresso_trainer.cfg import (
    AugmentationConfig,
    ScheduleConfig,
    LoggingConfig,
    EnvironmentConfig,
)
from netspresso_trainer.cfg.data import PathConfig, ImageLabelPathConfig
from netspresso_trainer.cfg.augmentation import *

from .trainer_configs import TrainerConfigs
from .registries import (
    DATA_CONFIG_TYPE,
    TRAINING_CONFIG_TYPE,
    AUGMENTATION_CONFIG_TYPE,
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
)


class ModelTrainer:
    def __init__(self, task) -> None:
        self.task = self._validate_task(task)
        self.available_models = list(self._get_available_models().keys())
        self.data = None
        self.model = None
        self.training = TRAINING_CONFIG_TYPE[self.task]()
        self.augmentation = AUGMENTATION_CONFIG_TYPE[self.task]()
        self.logging = LoggingConfig()
        self.environment = EnvironmentConfig()

    def _validate_task(self, task):
        if task not in ["classification", "detection", "segmentation"]:
            raise ValueError(
                f"The task supports classification, detection, and segmentation. The entered task is {task}."
            )
        return task

    def _validate_config(self):
        if self.data is None:
            raise Exception(
                "The dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration."
            )
        if self.model is None:
            raise Exception(
                "The model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration."
            )

    def _get_available_models(self):
        available_models = {
            "classification": CLASSIFICATION_MODELS,
            "detection": DETECTION_MODELS,
            "segmentation": SEGMENTATION_MODELS,
        }[self.task]

        return available_models

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
        self.data = DATA_CONFIG_TYPE[self.task](**common_config)

    def set_dataset_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.data = yaml_path

    def set_model(self, model_name):
        self.model = self._get_available_models().get(model_name)()

        if self.model is None:
            raise ValueError(
                f"The '{model_name}' model is not supported for the '{self.task}' task. The available models are {self.available_models}."
            )

    def set_model_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.model = yaml_path

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
        self.training = ScheduleConfig(
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

    def set_training_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.training = yaml_path

    def set_augmentation_config(self, img_size, transforms, mix_transforms=None):
        self.augmentation = AugmentationConfig(
            img_size=img_size,
            transforms=transforms,
            mix_transforms=mix_transforms,
        )

    def set_augmentation_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.augmentation = yaml_path

    def set_logging_config(
        self,
        project_id: Optional[str] = None,
        output_dir: Union[Path, str] = "./outputs",
        tensorboard: bool = True,
        csv: bool = False,
        image: bool = True,
        stdout: bool = True,
        save_optimizer_state: bool = True,
        validation_epoch: int = 10,
        save_checkpoint_epoch: Optional[int] = None,
    ):
        self.logging = LoggingConfig(
            project_id=project_id,
            output_dir=output_dir,
            tensorboard=tensorboard,
            csv=csv,
            image=image,
            stdout=stdout,
            save_optimizer_state=save_optimizer_state,
            validation_epoch=validation_epoch,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )

    def set_logging_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.logging = yaml_path

    def set_environment_config(self, num_workers=4):
        self.environment = EnvironmentConfig(num_workers=num_workers)

    def set_environment_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.environment = yaml_path

    def train(self, gpus: str):
        self._validate_config()

        configs = TrainerConfigs(
            self.data, self.augmentation, self.model, self.training, self.logging, self.environment
        )

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
        logger.info(f"Remove {configs.temp_folder} folder.")
        shutil.rmtree(configs.temp_folder, ignore_errors=True)
