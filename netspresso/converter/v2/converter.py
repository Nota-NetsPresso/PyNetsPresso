import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import request

from loguru import logger

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.clients.launcher.v2.schemas.common import ModelOption
from netspresso.clients.launcher.v2.schemas.task.convert.response_body import ConvertTask
from netspresso.enums import DataType, DeviceName, ServiceTask, SoftwareVersion, Status, TaskStatusForDisplay
from netspresso.enums.conversion import SourceFramework, TargetFramework
from netspresso.enums.project import SubFolder
from netspresso.utils import FileHandler
from netspresso.utils.db.models.compression import CompressionTask
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.models.project import Project
from netspresso.utils.db.repositories.compression import compression_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"

class ConverterV2(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse):
        """Initialize the Converter."""
        super().__init__(token_handler)
        self.user_info = user_info

    def get_supported_options(self, framework: SourceFramework) -> List[ModelOption]:
        """Get supported options for the specified framework.

        Args:
            framework: Source framework to get options for

        Returns:
            List of supported model options
        """
        self.token_handler.validate_token()

        options_response = launcher_client_v2.converter.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=framework,
        )
        supported_options = options_response.data

        # Filter out DLC framework (will be removed when DLC is supported)
        supported_options = [
            supported_option
            for supported_option in supported_options
            if supported_option.framework != "dlc"
        ]

        return supported_options

    def _download_converted_model(self, convert_task: ConvertTask, local_path: str) -> None:
        """Download the converted model.

        Args:
            convert_task: Conversion task containing the model
            local_path: Path to save the downloaded model

        Raises:
            Exception: If download fails
        """
        self.token_handler.validate_token()

        try:
            download_url = launcher_client_v2.converter.download_model_file(
                convert_task_uuid=convert_task.convert_task_id,
                access_token=self.token_handler.tokens.access_token,
            ).data.presigned_download_url

            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e

    def get_input_model(self, input_model_id: str, user_id: str) -> Model:
        """Get model by ID.

        Args:
            input_model_id: ID of the model to retrieve
            user_id: User ID for verification

        Returns:
            Model object
        """
        with get_db_session() as db:
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)
            return input_model

    def get_compression_task(self, model_id: str) -> CompressionTask:
        with get_db_session() as db:
            compression_task = compression_task_repository.get_by_model_id(db=db, model_id=model_id)
            return compression_task

    def save_model(self, model_name: str, project_id: str, user_id: str) -> Model:
        """Create and save a new converted model.

        Args:
            model_name: Name of the model
            project_id: Project ID to associate with the model
            user_id: User ID who owns the model

        Returns:
            Saved model object
        """
        model = Model(
            name=model_name,
            type=SubFolder.CONVERTED_MODELS,
            is_retrainable=False,
            project_id=project_id,
            user_id=user_id,
        )
        return self._save_model(model)

    def _save_model(self, model: Model) -> Model:
        """Save model to database.

        Args:
            model: Model object to save

        Returns:
            Saved model with updated attributes
        """
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)
            return model

    def _save_conversion_task(self, conversion_task: ConversionTask) -> ConversionTask:
        """Save conversion task to database.

        Args:
            conversion_task: Conversion task to save

        Returns:
            Saved conversion task with updated attributes
        """
        with get_db_session() as db:
            conversion_task = conversion_task_repository.save(db=db, model=conversion_task)
            return conversion_task

    def create_conversion_task(
        self,
        framework: Union[str, TargetFramework],
        device_name: Union[str, DeviceName],
        software_version: Optional[Union[str, SoftwareVersion]],
        data_type: Union[str, DataType],
        input_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
        conversion_task_id: Optional[str] = None,
    ) -> ConversionTask:
        """Create a new conversion task.

        Args:
            framework: Target framework for conversion
            device_name: Target device name
            software_version: Target software version
            data_type: Target data type (precision)
            input_model_id: ID of the input model
            model_id: ID of the output model

        Returns:
            Created conversion task object
        """
        with get_db_session() as db:
            if conversion_task_id:
                conversion_task = ConversionTask(
                    task_id=conversion_task_id,
                    framework=framework,
                    device_name=device_name,
                    software_version=software_version,
                    precision=data_type,
                    status=Status.NOT_STARTED,
                    input_model_id=input_model_id,
                    model_id=model_id,
                    user_id=self.user_info.user_id,
                )
            else:
                conversion_task = ConversionTask(
                    framework=framework,
                    device_name=device_name,
                    software_version=software_version,
                    precision=data_type,
                    status=Status.NOT_STARTED,
                    input_model_id=input_model_id,
                    model_id=model_id,
                    user_id=self.user_info.user_id,
                )
            conversion_task = conversion_task_repository.save(db=db, model=conversion_task)
            return conversion_task

    def _get_enum_value(self, enum_obj: Any) -> str:
        """Safely extract the string value from an enum or string.

        Args:
            enum_obj: Enum object or string

        Returns:
            String value of the enum or the original string
        """
        if hasattr(enum_obj, 'value'):
            return enum_obj.value
        return str(enum_obj)

    def convert_model_from_id(
        self,
        input_model_id: str,
        target_framework: Union[str, TargetFramework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
        output_dir: Optional[str] = None,
        conversion_task_id: Optional[str] = None,
    ) -> str:
        """Convert a model using its model ID.

        Args:
            input_model_id: ID of the model to convert
            target_framework: Target framework name
            target_device_name: Target device name
            target_data_type: Data type of the model. Default is DataType.FP16
            target_software_version: Target software version. Required if target_device_name is one of the Jetson devices
            input_layer: Target input shape for conversion (e.g., dynamic batch to static batch)
            dataset_path: Path to the dataset. Useful for certain conversions
            wait_until_done: If True, wait for the conversion result before returning
            sleep_interval: Time to wait between task status checks
            output_dir: Local folder path to save the converted model. If None, a temporary directory will be used

        Returns:
            str: Conversion task ID
        """
        # Initialize temporary directory variable
        temp_dir = None

        try:
            # Handle output directory
            if output_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="netspresso_convert_")
                output_dir = temp_dir
            else:
                output_dir = FileHandler.create_unique_folder(folder_path=output_dir)

            # Load model object
            input_model = self.get_input_model(input_model_id, self.user_info.user_id)
            if not input_model:
                raise ValueError(f"Model with ID {input_model_id} not found")

            input_model.user_id = self.user_info.user_id
            project = self.get_project(project_id=input_model.project_id)

            # Download model to temporary directory
            download_dir = Path(output_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)

            if input_model.type == SubFolder.COMPRESSED_MODELS:
                remote_model_path = Path(input_model.object_path).parent / "model.onnx"
            else:
                remote_model_path = Path(input_model.object_path) / "model.onnx"

            local_path = download_dir / "model.onnx"

            logger.info(f"Downloading input model from Zenko: {remote_model_path}")
            storage_handler.download_file_from_s3(
                bucket_name=BUCKET_NAME,
                local_path=str(local_path),
                object_path=str(remote_model_path)
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            if target_data_type == DataType.INT8:
                if input_model.type == SubFolder.COMPRESSED_MODELS:
                    compression_task = self.get_compression_task(model_id=input_model.model_id)
                    input_model = self.get_input_model(input_model_id=compression_task.input_model_id, user_id=self.user_info.user_id)
                    remote_calibration_dataset_path = Path(input_model.object_path) / "calibration_dataset.npy"
                else:
                    remote_calibration_dataset_path = Path(input_model.object_path) / "calibration_dataset.npy"
                local_calibration_dataset_path = download_dir / "calibration_dataset.npy"

                logger.info(f"Downloading calibration dataset from Zenko: {remote_calibration_dataset_path}")
                storage_handler.download_file_from_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=str(local_calibration_dataset_path),
                    object_path=str(remote_calibration_dataset_path)
                )
                logger.info(f"Downloaded calibration dataset from Zenko: {local_calibration_dataset_path}")

                dataset_path = local_calibration_dataset_path.as_posix()

            # Execute common conversion logic
            return self._perform_conversion(
                input_model=input_model,
                project=project,
                input_model_path=str(local_path),
                output_dir=output_dir,
                target_framework=target_framework,
                target_device_name=target_device_name,
                target_data_type=target_data_type,
                target_software_version=target_software_version,
                input_layer=input_layer,
                dataset_path=dataset_path,
                wait_until_done=wait_until_done,
                sleep_interval=sleep_interval,
                conversion_task_id=conversion_task_id,
            )
        except Exception as e:
            logger.error(f"Error in convert_model_from_id: {e}")
            raise e
        finally:
            # Clean up temporary directory (if output directory is a temporary directory)
            if temp_dir and os.path.exists(temp_dir) and output_dir == temp_dir:
                logger.info(f"Cleaning up temporary files in: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Successfully removed temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")

    def convert_model_from_path(
        self,
        input_model_path: str,
        project_id: str,
        target_framework: Union[str, TargetFramework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
        output_dir: Optional[str] = None,
    ) -> str:
        """Convert a model using its file path.

        Args:
            input_model_path: File path where the model is located
            target_framework: Target framework name
            target_device_name: Target device name
            target_data_type: Data type of the model. Default is DataType.FP16
            target_software_version: Target software version. Required if target_device_name is one of the Jetson devices
            input_layer: Target input shape for conversion (e.g., dynamic batch to static batch)
            dataset_path: Path to the dataset. Useful for certain conversions
            wait_until_done: If True, wait for the conversion result before returning
            sleep_interval: Time to wait between task status checks
            output_dir: Local folder path to save the converted model. If None, a temporary directory will be used
            project_id: Project ID. If None, the default project will be used

        Returns:
            str: Conversion task ID
        """
        # Initialize temporary directory variable
        temp_dir = None

        try:
            # Verify model file exists
            FileHandler.check_input_model_path(input_model_path)

            # Handle output directory
            if output_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="netspresso_convert_")
                output_dir = temp_dir
            else:
                output_dir = FileHandler.create_unique_folder(folder_path=output_dir)

            # Use default project or specified project
            project = self.get_project(project_id=project_id)

            # Generate model name (extracted from file name)
            model_name = Path(input_model_path).stem

            # Create temporary model
            temp_model = Model(
                name=model_name,
                type=SubFolder.PRETRAINED_MODELS,
                is_retrainable=False,
                project_id=project.project_id,
                user_id=self.user_info.user_id,
                object_path=input_model_path,
            )
            temp_model = self._save_model(temp_model)

            # Execute common conversion logic
            return self._perform_conversion(
                input_model=temp_model,
                project=project,
                input_model_path=input_model_path,
                output_dir=output_dir,
                target_framework=target_framework,
                target_device_name=target_device_name,
                target_data_type=target_data_type,
                target_software_version=target_software_version,
                input_layer=input_layer,
                dataset_path=dataset_path,
                wait_until_done=wait_until_done,
                sleep_interval=sleep_interval,
            )
        except Exception as e:
            logger.error(f"Error in convert_model_from_path: {e}")
            raise e
        finally:
            # Clean up temporary directory (if output directory is a temporary directory)
            if temp_dir and os.path.exists(temp_dir) and output_dir == temp_dir:
                logger.info(f"Cleaning up temporary files in: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Successfully removed temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")

    def _perform_conversion(
        self,
        input_model: Model,
        project: Project,
        input_model_path: str,
        output_dir: str,
        target_framework: Union[str, TargetFramework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
        conversion_task_id: Optional[str] = None,
    ) -> str:
        """Perform the actual model conversion (common logic)

        Args:
            input_model: Model object to convert
            project: Project associated with the model
            input_model_path: Local path to the model file
            output_dir: Directory to save the converted model
            target_framework: Target framework for conversion
            target_device_name: Target device name
            target_data_type: Target data type (precision)
            target_software_version: Target software version
            input_layer: Input layer configuration
            dataset_path: Path to the dataset (if needed)
            wait_until_done: Whether to wait for conversion to complete
            sleep_interval: Time between status checks

        Returns:
            Conversion task ID
        """
        # Set output model path
        _ = FileHandler.get_default_model_path(folder_path=output_dir)
        extension = FileHandler.get_extension(framework=target_framework)

        # Generate model name with safe enum value handling
        model_name_parts = [
            input_model.name,
            self._get_enum_value(target_framework),
            self._get_enum_value(target_device_name),
        ]

        if target_software_version:  # Add only if not None
            model_name_parts.append(self._get_enum_value(target_software_version))

        model_name_parts.append(self._get_enum_value(target_data_type))
        model_name = "_".join(map(str, model_name_parts))

        logger.info(f"Model name: {model_name}")

        # Save converted model
        model = self.save_model(
            model_name=model_name,
            project_id=input_model.project_id,
            user_id=self.user_info.user_id,
        )

        object_path = f"{project.user_id}/{project.project_id}/{model.model_id}/model{extension}"
        logger.info(f"Object path: {object_path}")
        model.object_path = object_path
        model = self._save_model(model)

        # Create conversion task
        conversion_task = self.create_conversion_task(
            framework=target_framework,
            device_name=target_device_name,
            software_version=target_software_version,
            data_type=target_data_type,
            input_model_id=input_model.model_id,
            model_id=model.model_id,
            conversion_task_id=conversion_task_id,
        )

        try:
            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_CONVERT)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.converter.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.converter.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.converter.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Get input layer information
            actual_input_layer = input_layer
            if not actual_input_layer and validate_model_response.data.detail.input_layers:
                actual_input_layer = validate_model_response.data.detail.input_layers[0]

            # Start convert task
            convert_response = launcher_client_v2.converter.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                target_device_name=target_device_name,
                target_framework=target_framework,
                data_type=target_data_type,
                input_layer=actual_input_layer,
                software_version=target_software_version,
                dataset_path=dataset_path,
            )
            logger.info(f"Convert response: {convert_response.data}")

            conversion_task.convert_task_uuid = convert_response.data.convert_task_id
            conversion_task = self._save_conversion_task(conversion_task)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    convert_response = launcher_client_v2.converter.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=convert_response.data.convert_task_id,
                    )
                    if convert_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                        TaskStatusForDisplay.USER_CANCEL,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if convert_response.data.status in [TaskStatusForDisplay.IN_PROGRESS, TaskStatusForDisplay.IN_QUEUE]:
                conversion_task.status = Status.IN_PROGRESS
                logger.info(f"Conversion task running. Status: {convert_response.data.status}")
            elif convert_response.data.status == TaskStatusForDisplay.FINISHED:
                download_dir = Path(model.object_path).parent
                download_dir.mkdir(parents=True, exist_ok=True)
                self._download_converted_model(
                    convert_task=convert_response.data,
                    local_path=model.object_path,
                )
                storage_handler.upload_file_to_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=model.object_path,
                    object_path=model.object_path
                )
                logger.info(f"Uploaded Converted Model file to Zenko: {model.object_path}")
                self.print_remaining_credit(service_task=ServiceTask.MODEL_CONVERT)
                conversion_task.status = Status.COMPLETED
                logger.info("Conversion task completed successfully.")
            elif convert_response.data.status in [
                TaskStatusForDisplay.ERROR,
                TaskStatusForDisplay.USER_CANCEL,
                TaskStatusForDisplay.TIMEOUT,
            ]:
                conversion_task.status = Status.ERROR
                conversion_task.error_detail = convert_response.data.error_log
                conversion_task = self._save_conversion_task(conversion_task)
                logger.error(f"Conversion task failed. Error: {convert_response.data.error_log}")

        except Exception as e:
            conversion_task.status = Status.ERROR
            conversion_task.error_detail = str(e) if e.args else "Unknown error"
            logger.error(f"Exception during conversion: {e}")
        except KeyboardInterrupt:
            conversion_task.status = Status.STOPPED
            logger.info("Conversion task stopped by user")
        finally:
            conversion_task = self._save_conversion_task(conversion_task)

        return conversion_task.task_id

    def convert_model(
        self,
        input_model_path: str,
        output_dir: str,
        target_framework: Union[str, TargetFramework],
        target_device_name: Union[str, DeviceName],
        target_data_type: Union[str, DataType] = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        input_layer: Optional[InputLayer] = None,
        dataset_path: Optional[str] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
        input_model_id: Optional[str] = None,
        project_id: Optional[str] = None,
        conversion_task_id: Optional[str] = None,
    ) -> str:
        """Convert a model to the specified framework.

        Args:
            input_model_path: The file path where the model is located.
            output_dir: The local folder path to save the converted model.
            target_framework: The target framework name.
            target_device_name: Target device name.
            target_data_type: Data type of the model. Default is DataType.FP16.
            target_software_version: Target software version.
                Required if target_device_name is one of the Jetson devices.
            input_layer: Target input shape for conversion (e.g., dynamic batch to static batch).
            dataset_path: Path to the dataset. Useful for certain conversions.
            wait_until_done: If True, wait for the conversion result before returning.
                If False, request the conversion and return the function immediately.
            input_model_id: Model ID to convert (alternative to input_model_path)
            project_id: Project ID for the model (required when using input_model_path)

        Raises:
            ValueError: If neither input_model_id nor input_model_path is provided, or if
                       input_model_path is provided without project_id

        Returns:
            str: Conversion task ID.
        """
        # Maintain backward compatibility with original convert_model function
        # Redirect to new functions

        if input_model_id:
            return self.convert_model_from_id(
                input_model_id=input_model_id,
                target_framework=target_framework,
                target_device_name=target_device_name,
                target_data_type=target_data_type,
                target_software_version=target_software_version,
                input_layer=input_layer,
                dataset_path=dataset_path,
                wait_until_done=wait_until_done,
                sleep_interval=sleep_interval,
                output_dir=output_dir,
                conversion_task_id=conversion_task_id,
            )
        elif input_model_path:
            return self.convert_model_from_path(
                input_model_path=input_model_path,
                target_framework=target_framework,
                target_device_name=target_device_name,
                target_data_type=target_data_type,
                target_software_version=target_software_version,
                input_layer=input_layer,
                dataset_path=dataset_path,
                wait_until_done=wait_until_done,
                sleep_interval=sleep_interval,
                output_dir=output_dir,
                project_id=project_id,
            )
        else:
            raise ValueError("Either input_model_id or input_model_path must be provided")

    def get_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Get the conversion task information with given conversion task uuid.

        Args:
            conversion_task_id: Convert task UUID of the convert task.

        Raises:
            Exception: If an error occurs during retrieval.

        Returns:
            ConversionTask: Model conversion task data.
        """
        self.token_handler.validate_token()

        response = launcher_client_v2.converter.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data

    def cancel_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Cancel the conversion task with given conversion task uuid.

        Args:
            conversion_task_id: Convert task UUID of the convert task.

        Raises:
            Exception: If an error occurs during task cancellation.

        Returns:
            ConversionTask: Model conversion task data.
        """
        self.token_handler.validate_token()

        response = launcher_client_v2.converter.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=conversion_task_id,
        )
        return response.data

    def update_conversion_task_status(self, task_id: str) -> bool:
        """Update conversion task status in DB based on launcher status.

        Args:
            task_id (str): Conversion task ID to update

        Returns:
            bool: True if status was updated, False if task is still in progress
        """
        with get_db_session() as db:
            conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=task_id)
            if not conversion_task:
                logger.error(f"Conversion task {task_id} not found")
                return True

            launcher_status = self.get_conversion_task(conversion_task.convert_task_uuid)
            status_updated = False

            if launcher_status.status == TaskStatusForDisplay.FINISHED:
                conversion_task.status = Status.COMPLETED
                status_updated = True
                model = model_repository.get_by_model_id(db=db, model_id=conversion_task.model_id)
                download_dir = Path(model.object_path).parent
                download_dir.mkdir(parents=True, exist_ok=True)
                self._download_converted_model(
                    convert_task=launcher_status,
                    local_path=model.object_path,
                )
                logger.info(f"Downloaded model to {model.object_path}")
                storage_handler.upload_file_to_s3(
                    bucket_name=BUCKET_NAME,
                    local_path=model.object_path,
                    object_path=model.object_path
                )
                logger.info(f"Uploaded Converted Model file to Zenko: {model.object_path}")

            elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
                conversion_task.status = Status.ERROR
                conversion_task.error_detail = launcher_status.error_log
                status_updated = True

            elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
                conversion_task.status = Status.STOPPED
                status_updated = True

            if status_updated:
                conversion_task_repository.save(db, conversion_task)
                logger.info(f"Conversion task {task_id} status updated to {conversion_task.status}")

            return status_updated
