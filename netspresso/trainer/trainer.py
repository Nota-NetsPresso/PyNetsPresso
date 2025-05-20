import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from omegaconf import OmegaConf

from netspresso.analytics import netspresso_analytics
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.launcher import launcher_client_v2
from netspresso.enums import Framework, Optimizer, Scheduler, ServiceTask, Status, Task
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
from netspresso.metadata.trainer import TrainerMetadata
from netspresso.trainer.augmentations import AUGMENTATION_CONFIG_TYPE, AugmentationConfig, Transform
from netspresso.trainer.data import DATA_CONFIG_TYPE, ImageLabelPathConfig, PathConfig
from netspresso.trainer.models import (
    CLASSIFICATION_MODELS,
    DETECTION_MODELS,
    SEGMENTATION_MODELS,
    CheckpointConfig,
    ModelConfig,
)
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.trainer.training import TRAINING_CONFIG_TYPE, EnvironmentConfig, LoggingConfig, ScheduleConfig
from netspresso.utils import FileHandler
from netspresso.utils.metadata import MetadataHandler


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
            "EfficientFormer": "EfficientFormer-L1",
            "MobileNetV3_Small": "MobileNetV3-S",
            "MobileNetV3_Large": "MobileNetV3-L",
            "ViT-Tiny": "ViT-T",
            "MixNet-Small": "MixNet-S",
            "MixNet-Medium": "MixNet-M",
            "MixNet-Large": "MixNet-L",
            "PIDNet": "PIDNet-S",
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
            raise NotSupportedModelException()

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

    def initialize_metadata(self, output_dir):
        def create_metadata_with_status(status, error_message=None):
            metadata = TrainerMetadata()
            metadata.status = status
            if error_message:
                logger.error(error_message)
            return metadata

        try:
            metadata = TrainerMetadata()
        except Exception as e:
            error_message = f"An unexpected error occurred during metadata initialization: {e}"
            metadata = create_metadata_with_status(Status.ERROR, error_message)
        except KeyboardInterrupt:
            warning_message = "Training task was interrupted by the user."
            metadata = create_metadata_with_status(Status.STOPPED, warning_message)
        finally:
            metadata.update_output_dir(output_dir.resolve().as_posix())
            metadata.update_model_info(
                task=self.task,
                model=self.model_name,
                dataset=self.data.name,
                input_shapes=[InputShape(batch=1, channel=3, dimension=[self.img_size, self.img_size])],
            )
            metadata.update_training_info(
                epochs=self.training.epochs,
                batch_size=self.environment.batch_size,
                learning_rate=self.training.optimizer["lr"],
                optimizer=Optimizer.to_display_name(self.training.optimizer["name"]),
                scheduler=Scheduler.to_display_name(self.training.scheduler["name"]),
            )
            metadata.update_hparams(hparams=(output_dir / "hparams.yaml").resolve().as_posix())
            MetadataHandler.save_metadata(data=metadata, folder_path=output_dir)

        return metadata

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

    def train(self, gpus: str, project_name: str, output_dir: Optional[str] = "./outputs") -> TrainerMetadata:
        """Train the model with the specified configuration.

        Args:
            gpus (str): GPU ids to use, separated by commas.
            project_name (str): Project name to save the experiment.

        Returns:
            Dict: A dictionary containing information about the training.
        """

        from netspresso_trainer import train_with_yaml

        netspresso_analytics.send_event(
            event_name="train_model_using_np",
            event_params={
                "task": self.task,
                "model": self.model_name,
            },
        )

        self._validate_config()
        self._apply_img_size()

        project_name = project_name if project_name else f"{self.task}_{self.model_name}".lower()
        destination_folder = Path(output_dir) / project_name
        destination_folder = FileHandler.create_unique_folder(folder_path=destination_folder)
        metadata = self.initialize_metadata(output_dir=destination_folder)

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

            available_options = self._get_available_options()
            metadata.update_available_options(available_options)

        except Exception as e:
            e = FailedTrainingException(error_log=e.args[0])
            metadata = self.handle_error(metadata, ServiceTask.TRAINING, e.args[0])
        except KeyboardInterrupt:
            metadata = self.handle_stop(metadata, ServiceTask.TRAINING)
        finally:
            FileHandler.remove_folder(configs.temp_folder)
            logger.info(f"Removed {configs.temp_folder} folder.")

            FileHandler.move_and_cleanup_folders(source_folder=self.logging_dir, destination_folder=destination_folder)
            logger.info(f"Files in {self.logging_dir} were moved to {destination_folder}.")

            runtime_config_path = self.create_runtime_config(yaml_path=destination_folder / "hparams.yaml")
            metadata.runtime = runtime_config_path.as_posix()

            training_summary = FileHandler.load_json(file_path=destination_folder / "training_summary.json")
            metadata.update_training_result(training_summary=training_summary)

            status = self._get_status_by_training_summary(training_summary.get("status"))
            metadata.update_status(status=status)
            if status == Status.ERROR:
                error_stats = training_summary.get("error_stats", "")
                e = FailedTrainingException(error_log=error_stats)
                metadata.update_message(exception_detail=e.args[0])

            best_fx_paths, best_onnx_paths = self.find_best_model_paths(destination_folder)
            if best_fx_paths:
                metadata.update_best_fx_model_path(best_fx_model_path=best_fx_paths[0].resolve().as_posix())
            if best_onnx_paths:
                metadata.update_best_onnx_model_path(best_onnx_model_path=best_onnx_paths[0].resolve().as_posix())
            MetadataHandler.save_metadata(data=metadata, folder_path=destination_folder)

        return metadata
