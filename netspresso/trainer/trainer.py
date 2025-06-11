import warnings
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.dataforge.schemas.response_body import DatasetPayload, DatasetVersionInfo
from netspresso.clients.launcher import launcher_client_v2
from netspresso.enums import Framework, Status, Task
from netspresso.enums.project import SubFolder
from netspresso.enums.train import StorageLocation
from netspresso.exceptions.trainer import (
    BaseDirectoryNotFoundException,
    DirectoryNotFoundException,
    FailedTrainingException,
    FileNotFoundErrorException,
    NotSetDatasetException,
    NotSetModelException,
    NotSupportedModelException,
    NotSupportedTaskException,
    RetrainingFunctionException,
    TaskOrYamlPathException,
)
from netspresso.inferencer.preprocessors.base import Preprocessor
from netspresso.metadata.common import InputShape
from netspresso.trainer.augmentations import AUGMENTATION_CONFIG_TYPE, AugmentationConfig, Transform
from netspresso.trainer.data import DATA_CONFIG_TYPE, ImageLabelPathConfig, PathConfig
from netspresso.trainer.models import (
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    CheckpointConfig,
    ModelConfig,
)
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.trainer.storage import DatasetManager
from netspresso.trainer.storage.dataforge import Split
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.trainer.training import TRAINING_CONFIG_TYPE, EnvironmentConfig, LoggingConfig, ScheduleConfig
from netspresso.trainer.training.logging import Metrics, ModelSaveOptions
from netspresso.utils import FileHandler
from netspresso.utils.db.models.evaluation import EvaluationDataset
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.models.training import (
    Augmentation,
    Dataset,
    Environment,
    Hyperparameter,
    Performance,
    TrainingTask,
)
from netspresso.utils.db.repositories.evaluation import evaluation_dataset_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"


class Trainer(NetsPressoBase):
    """
    NetsPresso Trainer Class: Base class for training models.
    """

    def __init__(
        self, token_handler: TokenHandler, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None
    ) -> None:
        """Initialize the Trainer.

        Args:
            task (Union[str, Task]], optional): The type of task (classification, detection, segmentation). Either 'task' or 'yaml_path' must be provided, but not both.
            yaml_path (str, optional): Path to the YAML configuration file. Either 'task' or 'yaml_path' must be provided, but not both.
        """
        super().__init__(token_handler=token_handler)

        self.token_handler = token_handler
        self.deprecated_names = {
            "efficientformer": "efficientformer_l1",
            "mobilenetv3_small": "mobilenet_v3_small",
            "mobilenetv3_large": "mobilenet_v3_large",
            "vit_tiny": "vit_tiny",
            "mixnet_small": "mixnet_s",
            "mixnet_medium": "mixnet_m",
            "mixnet_large": "mixnet_l",
            "pidnet": "pidnet_s",
        }

        if (task is not None) == (yaml_path is not None):
            raise TaskOrYamlPathException()

        if task is not None:
            self._initialize_from_task(task)
        elif yaml_path is not None:
            self._initialize_from_yaml(yaml_path)

        self.dataset_manager = DatasetManager(token_handler=token_handler)
        self.is_dataforge = False
        self.test_dataset = None
        self.test_dataset_id = None

    def _initialize_from_task(self, task: Union[str, Task]) -> None:
        """Initialize the Trainer object based on the provided task.

        Args:
            task (Union[str, Task]): The task for which the Trainer is initialized.
        """

        self.task = self._validate_task(task)
        self.available_models = list(self._get_available_models().keys())
        self.data = None
        self.model = None
        self.training = TRAINING_CONFIG_TYPE[self.task]()
        self.augmentation = AUGMENTATION_CONFIG_TYPE[self.task]()
        self.logging = LoggingConfig()
        self.environment = EnvironmentConfig()

    def _initialize_from_yaml(self, yaml_path: str) -> None:
        """Initialize the Trainer object based on the configuration provided in a YAML file.

        Args:
            yaml_path (str): The path to the YAML file containing the configuration.
        """

        hparams = OmegaConf.load(yaml_path)
        hparams["model"].pop("single_task_model")

        metadata_path = Path(yaml_path).parent / "metadata.json"
        metadata = FileHandler.load_json(metadata_path)
        self.model_name = metadata["model_info"]["model"]

        self.img_size = hparams["augmentation"]["img_size"]
        self.task = hparams["data"]["task"]
        self.available_models = list(self._get_available_models().keys())

        self.data = DATA_CONFIG_TYPE[self.task](**hparams["data"])
        self.model = ModelConfig(**hparams["model"])
        self.training = ScheduleConfig(**hparams["training"])
        self.augmentation = AugmentationConfig(**hparams["augmentation"])
        self.logging = LoggingConfig(**hparams["logging"])
        self.environment = EnvironmentConfig(**hparams["environment"])

    def _validate_task(self, task: Union[str, Task]):
        """Validate the provided task.

        Args:
            task (Union[str, Task]): The task to be validated.

        Raises:
            ValueError: If the provided task is not supported.

        Returns:
            Task: The validated task.
        """

        available_tasks = [task.value for task in Task]
        if task not in available_tasks:
            raise NotSupportedTaskException(available_tasks, task)
        return task

    def _validate_config(self):
        """Validate the configuration setup.

        Raises:
            ValueError: Raised if the dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration.
            ValueError: Raised if the model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration.
        """

        if self.data is None:
            raise NotSetDatasetException()
        if self.model is None:
            raise NotSetModelException()

    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models based on the current task.

        Returns:
            Dict[str, Any]: A dictionary mapping model types to available models.
        """

        available_models = {
            "classification": CLASSIFICATION_MODELS,
            "detection": DETECTION_MODELS,
            "segmentation": SEGMENTATION_MODELS,
        }[self.task]

        # Filter out deprecated names
        filtered_models = {
            name: config for name, config in available_models.items() if name not in self.deprecated_names
        }

        return filtered_models

    def _get_available_models_w_deprecated_names(self) -> Dict[str, Any]:
        """Get available models based on the current task.

        Returns:
            Dict[str, Any]: A dictionary mapping model types to available models.
        """

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
        valid_image: str = "images/valid",
        valid_label: str = "labels/valid",
        test_image: str = "images/valid",
        test_label: str = "labels/valid",
        id_mapping: Optional[Union[List[str], Dict[str, str], str]] = None,
    ):
        """Set the dataset configuration for the Trainer.

        Args:
            name (str): The name of dataset.
            root_path (str): Root directory of dataset.
            train_image (str, optional): The directory for training images. Should be relative path to root directory. Defaults to "images/train".
            train_label (str, optional): The directory for training labels. Should be relative path to root directory. Defaults to "labels/train".
            valid_image (str, optional): The directory for validation images. Should be relative path to root directory. Defaults to "images/val".
            valid_label (str, optional): The directory for validation labels. Should be relative path to root directory. Defaults to "labels/val".
            id_mapping (Union[List[str], Dict[str, str]], optional): ID mapping for the dataset. Defaults to None.
        """

        common_config = {
            "name": name,
            "path": PathConfig(
                root=root_path,
                train=ImageLabelPathConfig(image=train_image, label=train_label),
                valid=ImageLabelPathConfig(image=valid_image, label=valid_label),
                test=ImageLabelPathConfig(image=test_image, label=test_label),
            ),
            "id_mapping": id_mapping,
        }
        self.data = DATA_CONFIG_TYPE[self.task](**common_config)

    def check_paths_exist(self, base_path):
        paths = [
            "images/train",
            "images/valid",
            "id_mapping.json",
        ]

        # Check for the existence of required directories and files
        logger.info(f"Checking paths exist: {base_path}")
        for relative_path in paths:
            path = Path(base_path) / relative_path
            if not path.exists():
                if path.suffix:  # It's a file
                    raise FileNotFoundErrorException(path.as_posix())
                else:  # It's a directory
                    raise DirectoryNotFoundException(path.as_posix())

    def check_test_paths_exist(self, base_path):
        paths = [
            "images/test",
            "id_mapping.json",
        ]

        # Check for the existence of required directories and files
        for relative_path in paths:
            path = Path(base_path) / relative_path
            if not path.exists():
                if path.suffix:  # It's a file
                    raise FileNotFoundErrorException(relative_path)
                else:  # It's a directory
                    raise DirectoryNotFoundException(relative_path)

    def find_paths(self, base_path: str, search_dir, split: str) -> List[str]:
        base_dir = Path(base_path)

        if not base_dir.exists():
            raise BaseDirectoryNotFoundException(base_dir)

        result_paths = []

        dir_path = base_dir / search_dir
        if dir_path.exists() and dir_path.is_dir():
            for item in dir_path.iterdir():
                if (item.is_dir() or item.is_file()) and split in item.name:
                    result_paths.append(item.as_posix())

        return result_paths[0]

    def set_dataset(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        self.check_paths_exist(root_path)
        images_train = self.find_paths(root_path, "images", "train")
        images_valid = self.find_paths(root_path, "images", "valid")
        labels_train = self.find_paths(root_path, "labels", "train")
        labels_valid = self.find_paths(root_path, "labels", "valid")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")
        self.set_dataset_config(
            name=dataset_name,
            root_path=dataset_root_path,
            train_image=images_train,
            train_label=labels_train,
            valid_image=images_valid,
            valid_label=labels_valid,
            id_mapping=id_mapping,
        )

        train_image_path = Path(images_train)
        valid_image_path = Path(images_valid)
        train_image_count = len(list(train_image_path.glob("*.*"))) if train_image_path.is_dir() else 1
        valid_image_count = len(list(valid_image_path.glob("*.*"))) if valid_image_path.is_dir() else 1
        total_image_count = train_image_count + valid_image_count

        self.train_dataset = Dataset(
            name=dataset_name,
            path=root_path,
            id_mapping=self.data.id_mapping,
            palette=self.data.pallete,
            task_type=self.task,
            class_count=len(self.data.id_mapping),
            count=total_image_count,
            storage_location=StorageLocation.STORAGE if self.is_dataforge else StorageLocation.LOCAL,
            storage_info={"dataset_id": Path(root_path).name}
        )

    def _save_evaluation_dataset(self, evaluation_dataset):
        with get_db_session() as db:
            evaluation_dataset = evaluation_dataset_repository.save(db=db, model=evaluation_dataset)

            return evaluation_dataset

    def set_test_dataset(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        # self.check_test_paths_exist(root_path)
        images_test = self.find_paths(root_path, "images", "test")
        labels_test = self.find_paths(root_path, "labels", "test")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")

        if self.data is not None:
            self.data.path.test.image = images_test
            self.data.path.test.label = labels_test
        else:
            self.set_dataset_config(
                name=dataset_name,
                root_path=dataset_root_path,
                test_image=images_test,
                test_label=labels_test,
                id_mapping=id_mapping,
            )

        test_image_path = Path(images_test)
        test_image_count = len(list(test_image_path.glob("*.*"))) if test_image_path.is_dir() else 1

        storage_location = StorageLocation.STORAGE if self.is_dataforge else StorageLocation.LOCAL
        storage_info = {"dataset_id": Path(root_path).name} if self.is_dataforge else None

        self.test_dataset = EvaluationDataset(
            name=dataset_name,
            path=root_path,
            id_mapping=self.data.id_mapping,
            palette=self.data.pallete,
            task_type=self.task,
            class_count=len(self.data.id_mapping),
            count=test_image_count,
            storage_location=storage_location,
            storage_info=storage_info
        )
        self.test_dataset = self._save_evaluation_dataset(self.test_dataset)

    def set_test_dataset_no_create(self, dataset_root_path: str, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = Path(dataset_root_path).name
        root_path = Path(dataset_root_path).resolve().as_posix()

        # self.check_test_paths_exist(root_path)
        images_test = self.find_paths(root_path, "images", "test")
        labels_test = self.find_paths(root_path, "labels", "test")
        id_mapping = FileHandler.load_json(f"{root_path}/id_mapping.json")

        if self.data is not None:
            self.data.path.test.image = images_test
            self.data.path.test.label = labels_test
        else:
            self.set_dataset_config(
                name=dataset_name,
                root_path=dataset_root_path,
                test_image=images_test,
                test_label=labels_test,
                id_mapping=id_mapping,
            )

    def set_model_config(
        self,
        model_name: str,
        img_size: int,
        use_pretrained: bool = True,
        load_head: bool = False,
        path: Optional[str] = None,
        fx_model_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
    ):
        """Set the model configuration for the Trainer.

        Args:
            model_name (str): Name of the model.
            img_size (int): Image size for the model.
            use_pretrained (bool, optional): Whether to use a pre-trained model. Defaults to True.
            load_head (bool, optional): Whether to load the model head. Defaults to False.
            path (str, optional): Path to the model. Defaults to None.
            fx_model_path (str, optional): Path to the FX model. Defaults to None.
            optimizer_path (str, optional): Path to the optimizer. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported for the current task.
        """

        if model_name in self.deprecated_names:
            warnings.filterwarnings("default", category=DeprecationWarning)
            warnings.warn(
                f"The model name '{model_name}' is deprecated and will be removed in a future version. "
                f"Please use '{self.deprecated_names[model_name]}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            warnings.filterwarnings("ignore", category=DeprecationWarning)

        self.model_name = model_name
        model = self._get_available_models_w_deprecated_names().get(model_name)
        self.img_size = img_size
        self.logging.model_save_options.sample_input_size = [img_size, img_size]

        if model is None:
            raise NotSupportedModelException(
                available_models=self._get_available_models_w_deprecated_names(),
                model_name=model_name,
                task=self.task,
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

    def set_fx_model(self, fx_model_path: str):
        """Set the FX model path for retraining.

        Args:
            fx_model_path (str): The path to the FX model.

        Raises:
            ValueError: If the model is not set. Please use 'set_model_config' for model setup.
        """

        if not self.model:
            raise RetrainingFunctionException()

        self.model.checkpoint.path = None
        self.model.checkpoint.fx_model_path = fx_model_path

    def set_training_config(
        self,
        optimizer,
        scheduler,
        epochs: int = 3,
        batch_size: int = 8,
    ):
        """Set the training configuration.

        Args:
            optimizer: The configuration of optimizer.
            scheduler: The configuration of learning rate scheduler.
            epochs (int, optional): The total number of epoch for training the model. Defaults to 3.
            batch_size (int, optional): The number of samples in single batch input. Defaults to 8.
        """

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training = ScheduleConfig(
            epochs=epochs,
            optimizer=optimizer.asdict(),
            scheduler=scheduler.asdict(),
        )
        self.environment.batch_size = batch_size

    def set_augmentation_config(
        self,
        train_transforms: Optional[List] = None,
        inference_transforms: Optional[List] = None,
    ):
        """Set the augmentation configuration for training.

        Args:
            train_transforms (List, optional): List of transforms for training. Defaults to None.
            inference_transforms (List, optional): List of transforms for inference. Defaults to None.
        """

        self.augmentation = AugmentationConfig(
            train=train_transforms,
            inference=inference_transforms,
        )

    def set_logging_config(
        self,
        project_id: Optional[str] = None,
        output_dir: str = "./outputs",
        tensorboard: bool = True,
        csv: bool = False,
        image: bool = True,
        stdout: bool = True,
        save_optimizer_state: bool = True,
        validation_epoch: int = 10,
        save_checkpoint_epoch: Optional[int] = None,
    ):
        """Set the logging configuration.

        Args:
            project_id (str, optional): Project name to save the experiment. If None, it is set as {task}_{model} (e.g. segmentation_segformer).
            output_dir (str, optional): Root directory for saving the experiment. Defaults to "./outputs".
            tensorboard (bool, optional): Whether to use the tensorboard. Defaults to True.
            csv (bool, optional): Whether to save the result in csv format. Defaults to False.
            image (bool, optional): Whether to save the validation results. It is ignored if the task is classification. Defaults to True.
            stdout (bool, optional): Whether to log the standard output. Defaults to True.
            save_optimizer_state (bool, optional): Whether to save optimizer state with model checkpoint to resume training. Defaults to True.
            validation_epoch (int, optional): Validation frequency in total training process. Defaults to 10.
            save_checkpoint_epoch (int, optional): Checkpoint saving frequency in total training process. Defaults to None.
        """
        model_save_options = ModelSaveOptions(
            save_optimizer_state=save_optimizer_state,
            sample_input_size=[self.img_size, self.img_size],
            validation_epoch=validation_epoch,
            save_checkpoint_epoch=save_checkpoint_epoch,
        )
        metrics = Metrics(
            classwise_analysis=False,
            metric_names=None,
        )

        self.logging = LoggingConfig(
            project_id=project_id,
            output_dir=output_dir,
            tensorboard=tensorboard,
            csv=csv,
            image=image,
            stdout=stdout,
            model_save_options=model_save_options,
            metrics=metrics,
        )

    def set_environment_config(self, seed: int = 1, num_workers: int = 4):
        """Set the environment configuration.

        Args:
            seed (int, optional): Random seed. Defaults to 1.
            num_workers (int, optional): The number of multi-processing workers to be used by the data loader. Defaults to 4.
        """

        self.environment = EnvironmentConfig(seed=seed, num_workers=num_workers)

    def _change_transforms(self, transforms: Transform):
        """Update the 'size' attribute in the given list of transforms with the specified image size.

        Args:
            transforms (List[Transform]): The list of transforms to be updated.

        Returns:
            List[Transform]: The list of transforms with the 'size' attribute updated to the specified image size.
        """

        field_name_to_check = "size"

        if transforms is None:
            return transforms

        for transform in transforms:
            field_type = transform.__annotations__.get(field_name_to_check)

            if field_type == List:
                transform.size = [self.img_size, self.img_size]
            elif isinstance(field_type, int):
                transform.size = self.img_size

        return transforms

    def _apply_img_size(self):
        """Apply the specified image size to the augmentation configurations.

        This method updates the 'img_size' attribute in the augmentation configurations, including
        'train.transforms', 'train.mix_transforms', and 'inference.transforms'.
        """

        self.augmentation.img_size = self.img_size
        self.augmentation.train = self._change_transforms(self.augmentation.train)
        self.augmentation.inference = self._change_transforms(self.augmentation.inference)

    def _get_available_options(self):
        self.token_handler.validate_token()
        options_response = launcher_client_v2.converter.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=Framework.ONNX,
        )

        available_options = options_response.data

        # TODO: Will be removed when we support DLC in the future
        available_options = [
            available_option for available_option in available_options if available_option.framework != "dlc"
        ]

        return available_options

    def _get_status_by_training_summary(self, status):
        status_mapping = {
            "success": Status.COMPLETED,
            "stop": Status.STOPPED,
            "error": Status.ERROR,
            "": Status.IN_PROGRESS,
        }
        return status_mapping.get(status, Status.IN_PROGRESS)

    def find_best_model_paths(self, destination_folder: Path):
        best_fx_paths_set = set()

        for pattern in ["*best_fx.pt", "*best.pt"]:
            best_fx_paths_set.update(destination_folder.glob(pattern))

        best_fx_paths = list(best_fx_paths_set)
        best_onnx_paths = list(destination_folder.glob("*best.onnx"))

        return best_fx_paths, best_onnx_paths

    def create_runtime_config(self, yaml_path):
        hparams = OmegaConf.load(yaml_path)

        preprocess = hparams.augmentation.inference
        for _preprocess in preprocess:
            if hasattr(_preprocess, "size") and _preprocess.size:
                _preprocess.size = _preprocess.size[0]
            if _preprocess.name == "resize":
                _preprocess.resize_criteria = "long"

        if hparams.model.task == Task.IMAGE_CLASSIFICATION:
            visualize = {"params": {"class_map": hparams.data.id_mapping, "pallete": None}}
        elif hparams.model.task == Task.OBJECT_DETECTION:
            visualize = {
                "params": {"class_map": hparams.data.id_mapping, "normalized": False, "brightness_factor": 1.5}
            }
        elif hparams.model.task == Task.SEMANTIC_SEGMENTATION:
            visualize = {
                "params": {
                    "class_map": hparams.data.id_mapping,
                    "pallete": None,
                    "normalized": False,
                    "brightness_factor": 1.5,
                }
            }

        _config = {
            "task": hparams.model.task,
            "preprocess": preprocess,
            "postprocess": hparams.model.postprocessor,
            "visualize": visualize,
        }
        name = "runtime"
        config = OmegaConf.create({name: _config})
        save_path = Path(yaml_path).parent / f"{name}.yaml"
        OmegaConf.save(config=config, f=save_path)

        return save_path

    def _save_train_task(self, train_task):
        with get_db_session() as db:
            train_task = training_task_repository.save(db=db, model=train_task)

            return train_task

    def update_task_status(self, task_id: str, status: Status, error_message: Optional[str] = None) -> None:
        """
        Update the task status.

        Args:
            task_id: The ID of the task to update.
            status: The new status (Status enum value).
            error_message: The error message (optional).
        """
        logger.info(f"Updating task status: {task_id} -> {status}")
        if error_message:
            logger.error(f"Error message: {error_message}")

        with get_db_session() as session:
            training_task_repository.update_status(
                db=session,
                task_id=task_id,
                status=status,
                error_message=error_message,
            )

    def _save_model(self, model) -> Model:
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)

            return model

    def save_trained_model(self, model_name, project_id, user_id) -> Model:
        model = Model(
            name=model_name,
            type=SubFolder.TRAINED_MODELS,
            is_retrainable=True,
            project_id=project_id,
            user_id=user_id,
        )
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)
            return model

    def create_training_task(self, model_id, task_id, user_id, training_type, input_model_id) -> TrainingTask:
        with get_db_session() as db:
            augs = [
                Augmentation(
                    name=train_aug.name,
                    parameters=train_aug.to_parameters(),
                    phase="train",
                )
                for train_aug in self.augmentation.train
            ] + [
                Augmentation(
                    name=inference_aug.name,
                    parameters=inference_aug.to_parameters(),
                    phase="inference",
                )
                for inference_aug in self.augmentation.inference
            ]
            hyperparameter = Hyperparameter(
                epochs=self.training.epochs,
                batch_size=self.environment.batch_size,
                optimizer=self.optimizer.asdict(),
                scheduler=self.scheduler.asdict(),
                augmentations=augs,
            )
            environment = Environment(
                seed=self.environment.seed,
                num_workers=self.environment.num_workers,
                gpus=self.environment.gpus,
            )
            if task_id:
                task = TrainingTask(
                    task_id=task_id,
                    pretrained_model=self.model_name,
                    task=self.task,
                    framework=Framework.PYTORCH,
                    input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size]).__dict__],
                    status=Status.IN_PROGRESS,
                    dataset=self.train_dataset,
                    hyperparameter=hyperparameter,
                    environment=environment,
                    model_id=model_id,
                    user_id=user_id,
                    training_type=training_type,
                    input_model_id=input_model_id,
                )
            else:
                task = TrainingTask(
                    pretrained_model=self.model_name,
                    task=self.task,
                    framework=Framework.PYTORCH,
                    input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size]).__dict__],
                    status=Status.IN_PROGRESS,
                    dataset=self.train_dataset,
                    hyperparameter=hyperparameter,
                    environment=environment,
                    model_id=model_id,
                    user_id=user_id,
                    training_type=training_type,
                    input_model_id=input_model_id,
                )
            task = training_task_repository.save(db=db, model=task)

        return task

    def create_performance(self, task: TrainingTask, training_summary):
        performance = Performance(
            train_losses=training_summary["train_losses"],
            valid_losses=training_summary["valid_losses"],
            train_metrics=training_summary["train_metrics"],
            valid_metrics=training_summary["valid_metrics"],
            metrics_list=training_summary["metrics_list"],
            primary_metric=training_summary["primary_metric"],
            flops=str(training_summary["flops"]),
            params=str(training_summary["params"]),
            total_train_time=training_summary["total_train_time"],
            best_epoch=training_summary["best_epoch"],
            last_epoch=training_summary["last_epoch"],
            total_epoch=training_summary["total_epoch"],
            status=training_summary["status"],
        )

        task.performance = performance

        task = self._save_train_task(train_task=task)

        return task

    def train(
        self,
        gpus: str,
        model_name: str,
        project_id: str,
        output_dir: Optional[str] = "./outputs",
        task_id: Optional[str] = None,
        training_type: Optional[str] = "training",
        input_model_id: Optional[str] = None,
    ) -> str:
        """Train the model with the specified configuration.

        Args:
            gpus (str): GPU ids to use, separated by commas.
            project_name (str): Project name to save the experiment.

        Returns:
            Dict: A dictionary containing information about the training.
        """
        from netspresso_trainer import train_with_yaml

        # Validate configuration and initialize
        self._validate_config()
        self._apply_img_size()

        # Initialize project and model
        model_name = model_name if model_name else f"{self.task}_{self.model_name}".lower()
        project = self.get_project(project_id=project_id)

        # Setup folder
        destination_folder = self._prepare_destination_folder(project.project_abs_path, model_name)

        # Create model and task
        model = self._initialize_model(model_name, project)
        train_task = self.create_training_task(
            model_id=model.model_id,
            task_id=task_id,
            user_id=project.user_id,
            training_type=training_type,
            input_model_id=input_model_id,
        )

        # Setup logging
        self._setup_logging(output_dir, destination_folder.name)
        self.environment.gpus = gpus

        # Create training configurations
        configs = self._create_training_configs()

        try:
            # Train model
            self._execute_training(gpus, configs)
        except Exception as e:
            self._handle_training_error(train_task, e)
        except KeyboardInterrupt:
            train_task.status = Status.STOPPED
            train_task.error_detail = FailedTrainingException(error_log="Training stopped by user").args[0]
        finally:
            # Cleanup and move files
            self._cleanup_and_move_files(configs, destination_folder)

            # Process training summary
            train_task = self._process_training_summary(train_task, destination_folder)

            # Save training task
            train_task = self._save_train_task(train_task=train_task)

        # Upload model files if completed
        if train_task.status == Status.COMPLETED:
            calibration_dataset = self.prepare_calibration_dataset(dataset_path=self.data.path.train.image, num_dataset=100)
            self._upload_model_files(train_task, model, destination_folder, calibration_dataset)

        return train_task.task_id

    def _prepare_destination_folder(self, project_abs_path, model_name):
        """Prepare the folder to save the trained model."""
        destination_folder = Path(project_abs_path) / SubFolder.TRAINED_MODELS.value / model_name
        return FileHandler.create_unique_folder(folder_path=destination_folder)

    def _initialize_model(self, model_name: str, project: Project) -> Model:
        """Initialize and save the trained model object."""
        model = self.save_trained_model(
            model_name=model_name,
            project_id=project.project_id,
            user_id=project.user_id,
        )
        object_path = f"{project.user_id}/{project.project_id}/{model.model_id}"
        model.object_path = object_path
        return self._save_model(model=model)

    def _setup_logging(self, output_dir, project_id):
        """Set up the logging directory."""
        self.logging.output_dir = output_dir
        self.logging.project_id = project_id
        self.logging_dir = Path(self.logging.output_dir) / self.logging.project_id / "version_0"

    def _create_training_configs(self):
        """Create training configurations."""
        return TrainerConfigs(
            self.data,
            self.augmentation,
            self.model,
            self.training,
            self.logging,
            self.environment,
        )

    def _execute_training(self, gpus: str, configs: TrainerConfigs):
        """Execute model training."""
        from netspresso_trainer import train_with_yaml

        train_with_yaml(
            gpus=gpus,
            data=configs.data,
            augmentation=configs.augmentation,
            model=configs.model,
            training=configs.training,
            logging=configs.logging,
            environment=configs.environment,
        )

    def _handle_training_error(self, train_task: TrainingTask, error):
        """Handle training errors."""
        e = FailedTrainingException(error_log=error.args[0])
        train_task.status = Status.ERROR
        train_task.error_detail = e.args[0]

    def _cleanup_and_move_files(self, configs: TrainerConfigs, destination_folder: Path):
        """Clean up temporary files and move result files."""
        FileHandler.remove_folder(configs.temp_folder)
        logger.info(f"Removed {configs.temp_folder} folder.")

        FileHandler.move_and_cleanup_folders(source_folder=self.logging_dir, destination_folder=destination_folder)
        logger.info(f"Files in {self.logging_dir} were moved to {destination_folder}.")

    def _process_training_summary(self, train_task: TrainingTask, destination_folder):
        """Process training summary file and update training task status."""
        summary_path = destination_folder / "training_summary.json"

        if not summary_path.exists():
            logger.error(f"Training summary file not found at {summary_path}")
            error_msg = f"Training summary file not found at {summary_path}"
            training_summary = self._create_default_error_summary(error_msg)
            train_task.status = Status.ERROR
            train_task.error_detail = FailedTrainingException(error_log=error_msg).args[0]
        else:
            try:
                training_summary = FileHandler.load_json(file_path=summary_path)
            except Exception as e:
                logger.error(f"Failed to load training summary: {e}")
                error_msg = f"Failed to load training summary: {str(e)}"
                training_summary = self._create_default_error_summary(error_msg)
                train_task.status = Status.ERROR
                train_task.error_detail = FailedTrainingException(error_log=error_msg).args[0]
                return train_task

        try:
            train_task = self.create_performance(train_task, training_summary)
        except Exception as e:
            logger.error(f"Error creating performance record: {e}, {training_summary}")
            train_task.status = Status.ERROR
            train_task.error_detail = FailedTrainingException(error_log=f"Failed to create performance record: {str(e)}").args[0]
            return train_task

        train_task.status = self._get_status_by_training_summary(training_summary.get("status"))
        if train_task.status == Status.ERROR:
            error_stats = training_summary.get("error_stats", "")
            train_task.error_detail = FailedTrainingException(error_log=error_stats).args[0]

        return train_task

    def _create_default_error_summary(self, error_msg):
        """Create default error summary."""
        return {
            "train_losses": {}, "valid_losses": {},
            "train_metrics": {}, "valid_metrics": {},
            "metrics_list": [], "primary_metric": "",
            "flops": "0", "params": "0",
            "total_train_time": 0, "best_epoch": 0,
            "last_epoch": 0, "total_epoch": 0, "status": "error",
            "error_stats": error_msg
        }

    def _upload_model_files(self, train_task, model, destination_folder, calibration_dataset):
        """Upload trained model files to storage."""
        try:
            pt_file, onnx_file = self.find_model_files(destination_folder)

            errors = []
            if not pt_file:
                errors.append("PyTorch (PT) model file not found")
            if not onnx_file:
                errors.append("ONNX model file not found")

            if errors:
                error_msg = f"Required model files missing after training: {', '.join(errors)}"
                logger.error(error_msg)
                self.update_task_status(task_id=train_task.task_id, status=Status.ERROR, error_message=FailedTrainingException(error_log=error_msg))
                return

            self._upload_file_with_retry(
                local_path=str(pt_file),
                object_path=f"{model.object_path}/model.pt",
                file_type="PT"
            )

            self._upload_file_with_retry(
                local_path=str(onnx_file),
                object_path=f"{model.object_path}/model.onnx",
                file_type="ONNX"
            )

            if Path(calibration_dataset).exists():
                self._upload_file_with_retry(
                    local_path=calibration_dataset,
                    object_path=f"{model.object_path}/{Path(calibration_dataset).name}",
                    file_type="Numpy"
                )

        except Exception as e:
            error_msg = f"Failed to upload model files to Zenko: {e}"
            logger.error(error_msg)
            self.update_task_status(task_id=train_task.task_id, status=Status.ERROR, error_message=FailedTrainingException(error_log=error_msg))

    def _upload_file_with_retry(self, local_path, object_path, file_type, max_retries=3, retry_delay=5):
        """Execute file upload with retry mechanism."""
        import time

        for attempt in range(max_retries):
            try:
                storage_handler.upload_file_to_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=local_path,
                    object_path=object_path,
                )
                logger.info(f"Uploaded {file_type} file to Zenko: {object_path}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"{file_type} file upload attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to upload {file_type} file after {max_retries} attempts: {e}")

    def get_all_available_models(self) -> Dict[str, List[str]]:
        """Get all available models for each task, excluding deprecated names.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each task to its available models.
        """
        all_models = {
            "classification": [model for model in CLASSIFICATION_MODELS if model not in self.deprecated_names],
            "detection": [model for model in DETECTION_MODELS if model not in self.deprecated_names],
            "segmentation": [model for model in SEGMENTATION_MODELS if model not in self.deprecated_names],
        }
        return all_models

    def get_all_available_optimizers(self) -> Dict[str, Dict]:
        return get_supported_optimizers()

    def get_all_available_schedulers(self) -> Dict[str, Dict]:
        return get_supported_schedulers()

    def find_model_files(self, folder_path: Union[str, Path]) -> tuple[Optional[Path], Optional[Path]]:
        """Find one .pt file and one .onnx file in the given folder

        Args:
            folder_path: Path to search for model files

        Returns:
            tuple[Optional[Path], Optional[Path]]: Tuple of (pt_file_path, onnx_file_path)
            Each can be None if not found
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return None, None

        pt_files = list(folder_path.glob("*best*.pt")) or list(folder_path.glob("*.pt"))
        onnx_files = list(folder_path.glob("*best*.onnx")) or list(folder_path.glob("*.onnx"))

        if not pt_files:
            logger.warning(f"No PyTorch model files found in {folder_path}")
            pt_file = None
        else:
            pt_file = max(pt_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found PT file: {pt_file.name} (Last modified: {pt_file.stat().st_mtime})")

        if not onnx_files:
            logger.warning(f"No ONNX model files found in {folder_path}")
            onnx_file = None
        else:
            onnx_file = max(onnx_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found ONNX file: {onnx_file.name} (Last modified: {onnx_file.stat().st_mtime})")

        return pt_file, onnx_file

    def download_dataset_for_training(
        self,
        dataset_uuid: str,
        output_dir: str = "/datasets",
        valid_split: float = 0.1,
        random_seed: int = 0,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> str:
        """
        Download dataset from DataForge and set up dataset configuration for training.
        If the dataset is already downloaded, it will use the existing files.

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            valid_split: Ratio of validation data to split from train data (0.0-1.0)
            random_seed: Random seed for reproducible splitting
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file (default: False)

        Returns:
            str: Path to the configured dataset
        """
        try:
            dataset_path = self.dataset_manager.download_dataset_for_training(
                dataset_uuid=dataset_uuid,
                output_dir=output_dir,
                valid_split=valid_split,
                random_seed=random_seed,
                max_retries=max_retries,
                retry_delay=retry_delay,
                verbose=verbose,
            )
            self.is_dataforge = True
            logger.info(f"Downloaded dataset from DataForge: {dataset_path}")

            return dataset_path

        except Exception as e:
            logger.error(f"Failed to download dataset for training: {str(e)}")
            raise e

    def download_dataset_for_evaluation(
        self,
        dataset_uuid: str,
        output_dir: str = "/datasets",
        split: str = Split.TEST,
        max_retries: int = 3,
        retry_delay: int = 5,
        verbose: bool = False,
    ) -> str:
        """
        Download dataset from DataForge for evaluation purposes

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            split: Dataset split to download (default: TEST)
            max_retries: Maximum number of retry attempts for network/storage errors
            retry_delay: Delay in seconds between retry attempts (will increase with each retry)
            verbose: Whether to log detailed progress for each file (default: False)

        Returns:
            str: Path to the configured evaluation dataset
        """
        try:
            dataset_path = self.dataset_manager.download_dataset_for_evaluation(
                dataset_uuid=dataset_uuid,
                output_dir=output_dir,
                split=split,
                max_retries=max_retries,
                retry_delay=retry_delay,
                verbose=verbose,
            )
            logger.info(f"Downloaded dataset from DataForge: {dataset_path}")

            return dataset_path

        except Exception as e:
            logger.error(f"Failed to download dataset for evaluation: {str(e)}")
            raise e

    def get_dataset_version_from_storage(self, dataset_uuid: str, split: Split) -> DatasetVersionInfo:
        dataset_version = self.dataset_manager.get_dataset_version_from_dataforge(dataset_uuid=dataset_uuid, split=split)

        return dataset_version

    def get_dataset_info_from_storage(self, project_id: str, dataset_uuid: str, split: Split) -> DatasetPayload:
        dataset_info = self.dataset_manager.get_dataset_info_from_dataforge(project_id=project_id, dataset_uuid=dataset_uuid, split=split)

        return dataset_info

    def prepare_calibration_dataset(self, dataset_path, num_dataset: int = 100) -> str:
        """Create a calibration dataset."""
        logger.info("Creating calibration dataset")

        preprocess_list = [
            asdict(aug) for aug in self.augmentation.train
            if aug.name != "totensor"
        ]
        logger.info(f"Using preprocess_list: {preprocess_list}")
        preprocessor = Preprocessor(preprocess_list)

        inputs_array = []

        # Support multiple image extensions
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(f"{dataset_path}/{ext}"))

        # Limit the number of images to num_dataset
        image_paths = image_paths[:num_dataset]

        if not image_paths:
            logger.warning(f"No images found in {dataset_path} with extensions {image_extensions}")
            return

        logger.info(f"Processing {len(image_paths)} images for calibration dataset")

        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocessor(img)
            inputs_array.append(img)

        if not inputs_array:
            logger.warning("No valid images were processed")
            return

        result_array = np.concatenate(inputs_array, axis=0)

        # save chunk data
        calibration_dataset_path = f"{Path(dataset_path).parts[0]}/calibration_dataset.npy"
        np.save(calibration_dataset_path, result_array, allow_pickle=True)
        logger.info(f"Calibration dataset saved to {calibration_dataset_path}")

        return calibration_dataset_path
