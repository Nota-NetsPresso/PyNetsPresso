import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from omegaconf import OmegaConf

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.launcher import launcher_client_v2
from netspresso.enums import Framework, ServiceTask, Status, Task
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
from netspresso.metadata.common import InputShape
from netspresso.trainer.augmentations import AUGMENTATION_CONFIG_TYPE, AugmentationConfig, Transform
from netspresso.trainer.data import DATA_CONFIG_TYPE, ImageLabelPathConfig, PathConfig
from netspresso.trainer.dataforge.dataforget import Split, dataforge
from netspresso.trainer.models import (
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    CheckpointConfig,
    ModelConfig,
)
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.trainer.training import TRAINING_CONFIG_TYPE, EnvironmentConfig, LoggingConfig, ScheduleConfig
from netspresso.utils import FileHandler
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.models.training import (
    Augmentation,
    Dataset,
    Environment,
    Hyperparameter,
    Performance,
    TrainingTask,
)
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"


class Trainer(NetsPressoBase):
    def __init__(
        self, token_handler: TokenHandler, task: Optional[Union[str, Task]] = None, yaml_path: Optional[str] = None
    ) -> None:
        """Initialize the Trainer.

        Args:
            task (Union[str, Task]], optional): The type of task (classification, detection, segmentation). Either 'task' or 'yaml_path' must be provided, but not both.
            yaml_path (str, optional): Path to the YAML configuration file. Either 'task' or 'yaml_path' must be provided, but not both.
        """

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

    def set_dataset(self, dataset_root_path: str):
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
        self.logging.sample_input_size = [img_size, img_size]

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

    def create_training_task(self, model_id) -> TrainingTask:
        with get_db_session() as db:
            dataset = Dataset(
                train_path="train",
                valid_path="valid",
                test_path="test",
                storage_location=StorageLocation.STORAGE,
                id_mapping=self.data.id_mapping,
                palette=self.data.pallete,
            )
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
            task = TrainingTask(
                pretrained_model=self.model_name,
                task=self.task,
                framework=Framework.PYTORCH,
                input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size]).__dict__],
                status=Status.IN_PROGRESS,
                dataset=dataset,
                hyperparameter=hyperparameter,
                environment=environment,
                model_id=model_id,
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
        self, gpus: str, model_name: str, project_id: str, output_dir: Optional[str] = "./outputs"
    ) -> TrainingTask:
        """Train the model with the specified configuration.

        Args:
            gpus (str): GPU ids to use, separated by commas.
            project_name (str): Project name to save the experiment.

        Returns:
            Dict: A dictionary containing information about the training.
        """

        from netspresso_trainer import train_with_yaml

        self._validate_config()
        self._apply_img_size()

        model_name = model_name if model_name else f"{self.task}_{self.model_name}".lower()
        project = self.get_project(project_id=project_id)
        project_abs_path = project.project_abs_path

        destination_folder = Path(project_abs_path) / SubFolder.TRAINED_MODELS.value / model_name
        destination_folder = FileHandler.create_unique_folder(folder_path=destination_folder)

        model = self.save_trained_model(
            model_name=model_name,
            project_id=project.project_id,
            user_id=project.user_id,
        )
        object_path = f"{project.user_id}/{project.project_id}/{model.model_id}"
        model.object_path = object_path
        model = self._save_model(model=model)
        train_task = self.create_training_task(model_id=model.model_id)

        try:
            self.logging.output_dir = output_dir
            self.logging.project_id = destination_folder.name
            self.logging_dir = Path(self.logging.output_dir) / self.logging.project_id / "version_0"
            self.environment.gpus = gpus

            configs = TrainerConfigs(
                self.data,
                self.augmentation,
                self.model,
                self.training,
                self.logging,
                self.environment,
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

        except Exception as e:
            e = FailedTrainingException(error_log=e.args[0])
            train_task = self.handle_error(train_task, ServiceTask.TRAINING, e.args[0])
        except KeyboardInterrupt:
            train_task = self.handle_stop(train_task, ServiceTask.TRAINING)
        finally:
            FileHandler.remove_folder(configs.temp_folder)
            logger.info(f"Removed {configs.temp_folder} folder.")

            FileHandler.move_and_cleanup_folders(source_folder=self.logging_dir, destination_folder=destination_folder)
            logger.info(f"Files in {self.logging_dir} were moved to {destination_folder}.")

            training_summary = FileHandler.load_json(file_path=destination_folder / "training_summary.json")
            train_task = self.create_performance(train_task, training_summary)

            train_task.status = self._get_status_by_training_summary(training_summary.get("status"))
            if train_task.status == Status.ERROR:
                error_stats = training_summary.get("error_stats", "")
                e = FailedTrainingException(error_log=error_stats)
                train_task.error_detail = e.args[0]

            train_task = self._save_train_task(train_task=train_task)

            # Zenko에 모델 파일 업로드
            if train_task.status == Status.COMPLETED:
                try:
                    # 모델 파일 찾기
                    pt_file, onnx_file = self.find_model_files(destination_folder)

                    # PT 파일 업로드
                    if pt_file:
                        storage_handler.upload_file_to_s3(
                            bucket_name=BUCKET_NAME,
                            local_path=str(pt_file),
                            object_path=f"{model.object_path}/model.pt"
                        )
                        logger.info(f"Uploaded PT file to Zenko: {model.object_path}/model.pt")

                    # ONNX 파일 업로드
                    if onnx_file:
                        storage_handler.upload_file_to_s3(
                            bucket_name=BUCKET_NAME,
                            local_path=str(onnx_file),
                            object_path=f"{model.object_path}/model.onnx"
                        )
                        logger.info(f"Uploaded ONNX file to Zenko: {model.object_path}/model.onnx")

                except Exception as e:
                    logger.error(f"Failed to upload model files to Zenko: {e}")
                    # 업로드 실패해도 학습은 성공으로 처리
                    pass

        return train_task

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

        pt_files = list(folder_path.glob('*.pt'))
        onnx_files = list(folder_path.glob('*.onnx'))

        pt_file = pt_files[0] if pt_files else None
        onnx_file = onnx_files[0] if onnx_files else None

        if pt_file:
            logger.info(f"Found PT file: {pt_file.name}")
        if onnx_file:
            logger.info(f"Found ONNX file: {onnx_file.name}")

        return pt_file, onnx_file

    def download_dataset_from_storage(self, dataset_uuid: str, output_dir: str = "./datasets", valid_split: float = 0.2, random_seed: int = 0) -> str:
        """
        Download dataset from DataForge and set up dataset configuration for training

        Args:
            dataset_uuid: The UUID of the dataset to download
            output_dir: Directory to save downloaded files
            valid_split: Ratio of validation data to split from train data (0.0-1.0)
            random_seed: Random seed for reproducible splitting

        Returns:
            str: Path to the configured dataset
        """
        try:
            # Create base output directory
            dataset_dir = Path(output_dir) / dataset_uuid
            dataset_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading dataset with UUID: {dataset_uuid}")

            # Get the latest dataset version
            try:
                dataset_version = dataforge.get_latest_dataset_version(dataset_uuid=dataset_uuid, split=Split.TRAIN)
                if not dataset_version or not dataset_version.data:
                    logger.error(f"Could not get dataset info for UUID: {dataset_uuid}")
                    return ""
            except Exception as e:
                logger.error(f"Error getting dataset version: {str(e)}")
                return ""

            # Create temporary directory for downloads
            temp_dir = dataset_dir / "temp_download"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Download train data
            split = Split.TRAIN
            logger.info(f"Downloading {split} split data")

            try:
                # Download data for this split
                result = dataforge.download_dataset(dataset_version=dataset_version, output_dir=str(temp_dir))

                if not result:
                    logger.error(f"Failed to download {split} split")
                    return ""

                logger.success(f"Successfully downloaded {split} split")
            except Exception as e:
                logger.error(f"Error downloading dataset: {str(e)}")
                return ""

            # Prepare directory structure for trainer
            # The trainer expects:
            # - images/train/ and images/valid/ for images
            # - labels/train/ and labels/valid/ for labels
            images_dir = dataset_dir / "images"
            labels_dir = dataset_dir / "labels"

            # Create train/valid directories
            train_images_dir = images_dir / "train"
            train_labels_dir = labels_dir / "train"
            valid_images_dir = images_dir / "valid"
            valid_labels_dir = labels_dir / "valid"

            for dir_path in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Save id_mapping
            try:
                id_mapping = dataset_version.data.dataset_metadata.id_mapping
                with open(dataset_dir / "id_mapping.json", "w") as f:
                    json.dump(id_mapping, f)
                logger.info(f"Saved id_mapping.json with {len(id_mapping)} classes")
            except Exception as e:
                logger.warning(f"Error saving id_mapping.json: {str(e)}")
                # Create a default mapping if necessary
                with open(dataset_dir / "id_mapping.json", "w") as f:
                    json.dump({"0": "background", "1": "object"}, f)

            # Get source file paths
            source_images_dir = temp_dir / dataset_uuid / "images"
            source_annotations_dir = temp_dir / dataset_uuid / "annotations"

            if not source_images_dir.exists() or not source_annotations_dir.exists():
                logger.error("Required source directories not found after download")
                return ""

            # Get list of all images and corresponding annotations
            image_files = [f for f in source_images_dir.iterdir() if f.is_file()]

            if not image_files:
                logger.error("No image files found in downloaded dataset")
                return ""

            logger.info(f"Found {len(image_files)} image files")

            # Get corresponding annotation files (maintain image-annotation pairing)
            file_pairs = []
            for img_file in image_files:
                # Find matching annotation file (assuming same name, different extension)
                ann_candidates = list(source_annotations_dir.glob(f"{img_file.stem}.*"))
                if ann_candidates:
                    file_pairs.append((img_file, ann_candidates[0]))
                else:
                    logger.warning(f"No matching annotation found for {img_file.name}")

            logger.info(f"Found {len(file_pairs)} valid image-annotation pairs")

            # Randomize and split the dataset
            random.seed(random_seed)
            random.shuffle(file_pairs)

            # Calculate split point
            valid_count = max(1, int(len(file_pairs) * valid_split))
            valid_pairs = file_pairs[:valid_count]
            train_pairs = file_pairs[valid_count:]

            logger.info(f"Splitting into {len(train_pairs)} training and {len(valid_pairs)} validation samples")

            # Copy training files
            for img_file, ann_file in train_pairs:
                try:
                    shutil.copy2(img_file, train_images_dir / img_file.name)
                    shutil.copy2(ann_file, train_labels_dir / ann_file.name)
                except Exception as e:
                    logger.warning(f"Error copying training file {img_file.name}: {str(e)}")

            # Copy validation files
            for img_file, ann_file in valid_pairs:
                try:
                    shutil.copy2(img_file, valid_images_dir / img_file.name)
                    shutil.copy2(ann_file, valid_labels_dir / ann_file.name)
                except Exception as e:
                    logger.warning(f"Error copying validation file {img_file.name}: {str(e)}")

            # Set up dataset configuration
            try:
                self.set_dataset(dataset_dir.as_posix())
            except Exception as e:
                logger.error(f"Error configuring dataset: {str(e)}")
                return ""

            # Clean up temporary files
            try:
                logger.info("Cleaning up temporary files")
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")

            logger.success(f"Dataset downloaded, split and configured at: {dataset_dir}")
            logger.info(f"Train samples: {len(train_pairs)}, Validation samples: {len(valid_pairs)}")
            return dataset_dir.as_posix()

        except Exception as e:
            logger.exception(f"Unexpected error in download_dataset_from_storage: {str(e)}")
            return ""
