import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from netspresso_trainer import train_with_yaml
from netspresso_trainer.cfg import (
    AugmentationConfig,
    EnvironmentConfig,
    LoggingConfig,
    ScheduleConfig,
)
from netspresso_trainer.cfg.augmentation import *
from netspresso_trainer.cfg.data import ImageLabelPathConfig, PathConfig
from netspresso_trainer.cfg.model import CheckpointConfig
from netspresso.enums import TaskType, Status

from .enums import Task
from .registries import (
    AUGMENTATION_CONFIG_TYPE,
    CLASSIFICATION_MODELS,
    DATA_CONFIG_TYPE,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    TRAINING_CONFIG_TYPE,
)
from .trainer_configs import TrainerConfigs
from ..utils import FileHandler
from ..utils.metadata import MetadataHandler
from ..utils.metadata.default.trainer import InputShape


class Trainer:
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
        available_tasks = [task.value for task in Task]
        if task not in available_tasks:
            raise ValueError(
                f"The task supports {available_tasks}. The entered task is {task}."
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

    def set_model_config(
        self,
        model_name: str,
        img_size: int,
        use_pretrained: bool = True,
        load_head: bool = False,
        path: Optional[Union[Path, str]] = None,
        fx_model_path: Optional[Union[Path, str]] = None,
        optimizer_path: Optional[Union[Path, str]] = None,
    ):
        model = self._get_available_models().get(model_name)
        self.img_size = img_size

        if model is None:
            raise ValueError(
                f"The '{model_name}' model is not supported for the '{self.task}' task. The available models are {self.available_models}."
            )

        self.model = model(
            checkpoint=CheckpointConfig(
                use_pretrained=use_pretrained,
                load_head=load_head,
                path=path,
                fx_model_path=fx_model_path,
                optimizer_path=optimizer_path,
            )
        )

    def set_model_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.model = yaml_path

    def set_training_config(
        self,
        optimizer,
        scheduler,
        epochs: int = 3,
        batch_size: int = 8,
    ):
        self.training = ScheduleConfig(
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer.asdict(),
            scheduler=scheduler.asdict(),
        )

    def set_training_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.training = yaml_path

    def set_augmentation_config(
        self,
        train_transforms: Optional[List] = None,
        train_mix_transforms: Optional[List] = None,
        inference_transforms: Optional[List] = None,
    ):        
        self.augmentation = AugmentationConfig(
            train=Train(
                transforms=train_transforms,
                mix_transforms=train_mix_transforms,
            ),
            inference=Inference(
                transforms=inference_transforms,
            ),
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

    def set_environment_config(self, seed: int = 1, num_workers: int = 4):
        self.environment = EnvironmentConfig(seed=seed, num_workers=num_workers)

    def set_environment_config_with_yaml(self, yaml_path: Union[Path, str]):
        self.environment = yaml_path

    def _change_transforms(self, transforms):
        field_name_to_check = "size"

        if transforms is None:
            return transforms

        for transform in transforms:
            field_type = transform.__annotations__.get(field_name_to_check)

            if field_type == List:
                transform.size = [self.img_size, self.img_size]
            elif field_type == int:
                transform.size = self.img_size

        return transforms

    def _apply_img_size(self):
        self.augmentation.img_size = self.img_size
        self.augmentation.train.transforms = self._change_transforms(self.augmentation.train.transforms)
        self.augmentation.train.mix_transforms = self._change_transforms(self.augmentation.train.mix_transforms)
        self.augmentation.inference.transforms = self._change_transforms(self.augmentation.inference.transforms)

    def move_and_cleanup_folders(self, source_folder: str, destination_folder: str):
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(source_path, destination_path)

        os.rmdir(source_folder)

    def train(self, gpus: str, project_id: str) -> Dict:
        try:
            self.logging.project_id = project_id
            destination_folder = f"{self.logging.output_dir}/{self.logging.project_id}"
            os.makedirs(destination_folder, exist_ok=True)
            metadata = MetadataHandler.init_metadata(folder_path=destination_folder, task_type=TaskType.TRAIN)

            self._validate_config()
            self._apply_img_size()

            configs = TrainerConfigs(
                self.data,
                self.augmentation,
                self.model,
                self.training,
                self.logging,
                self.environment,
            )

            logging_dir = train_with_yaml(
                gpus=gpus,
                data=configs.data,
                augmentation=configs.augmentation,
                model=configs.model,
                training=configs.training,
                logging=configs.logging,
                environment=configs.environment,
            )
            training_summary_path = f"{logging_dir}/training_summary.json"
            training_summary = FileHandler.load_json(file_path=training_summary_path)

            metadata.update_model_info(
                task=self.task,
                model=self.model.name,
                dataset=self.data.name,
                input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size])],
            )
            metadata.update_training_info(epoch=self.training.epochs, batch_size=self.training.batch_size)
            metadata.update_training_result(training_summary=training_summary)
            metadata.update_logging_dir(logging_dir=destination_folder)
            metadata.update_status(status=Status.COMPLETED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=logging_dir)

            # Remove temp config folder
            logger.info(f"Remove {configs.temp_folder} folder.")
            shutil.rmtree(configs.temp_folder, ignore_errors=True)

            destination_folder = f"{self.logging.output_dir}/{self.logging.project_id}"
            self.move_and_cleanup_folders(source_folder=logging_dir, destination_folder=destination_folder)

            return metadata.asdict()

        except Exception as e:
            logger.error(f"Training failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=logging_dir)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=logging_dir)
