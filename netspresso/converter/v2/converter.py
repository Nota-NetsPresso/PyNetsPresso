import time
from pathlib import Path
from typing import List, Optional, Union
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
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.model import Model
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
        self.token_handler.validate_token()

        options_response = launcher_client_v2.converter.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=framework,
        )
        supported_options = options_response.data

        # TODO: Will be removed when we support DLC in the future
        supported_options = [
            supported_option
            for supported_option in supported_options
            if supported_option.framework != "dlc"
        ]

        return supported_options

    def _download_converted_model(self, convert_task: ConvertTask, local_path: str) -> None:
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConvertTask): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.
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
        with get_db_session() as db:
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)
            return input_model

    def save_model(self, model_name, project_id, user_id, object_path) -> Model:
        model = Model(
            name=model_name,
            type=SubFolder.CONVERTED_MODELS,
            is_retrainable=False,
            project_id=project_id,
            user_id=user_id,
            object_path=object_path,
        )
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)
            return model

    def _save_conversion_task(self, conversion_task: ConversionTask) -> ConversionTask:
        with get_db_session() as db:
            conversion_task = conversion_task_repository.save(db=db, model=conversion_task)

            return conversion_task

    def create_conversion_task(
        self,
        framework: Union[str, TargetFramework],
        device_name: Union[str, DeviceName],
        software_version: Union[str, SoftwareVersion],
        data_type: Union[str, DataType],
        input_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ConversionTask:
        with get_db_session() as db:
            conversion_task = ConversionTask(
                framework=framework,
                device_name=device_name,
                software_version=software_version,
                precision=data_type,
                status=Status.NOT_STARTED,
                input_model_id=input_model_id,
                model_id=model_id,
            )
            conversion_task = conversion_task_repository.save(db=db, model=conversion_task)
            return conversion_task

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
    ) -> str:
        """Convert a model to the specified framework.

        Args:
            input_model_path (str): The file path where the model is located.
            output_dir (str): The local folder path to save the converted model.
            target_framework (Union[str, Framework]): The target framework name.
            target_device_name (Union[str, DeviceName]): Target device name. Required if target_device is not specified.
            target_data_type (Union[str, DataType]): Data type of the model. Default is DataType.FP16.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version.
                Required if target_device_name is one of the Jetson devices.
            input_layer (InputShape, optional): Target input shape for conversion (e.g., dynamic batch to static batch).
            dataset_path (str, optional): Path to the dataset. Useful for certain conversions.
            wait_until_done (bool): If True, wait for the conversion result before returning the function.
                                If False, request the conversion and return  the function immediately.

        Raises:
            e: If an error occurs during the model conversion.

        Returns:
            ConverterMetadata: Convert metadata.
        """

        # FileHandler.check_input_model_path(input_model_path)
        output_dir = FileHandler.create_unique_folder(folder_path=output_dir)

        if input_model_id:
            input_model = self.get_input_model(input_model_id, self.user_info.user_id)
            input_model.user_id = self.user_info.user_id
            project = self.get_project(project_id=input_model.project_id)
            input_model_path = Path(input_model.object_path) / "model.onnx"
            download_dir = Path(output_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)  # 다운로드 폴더 생성

            local_path = download_dir / "model.onnx"
            input_model_path_str = str(input_model_path)

            logger.info(f"Downloading input model from Zenko: {input_model_path_str}")
            storage_handler.download_file_from_s3(
                bucket_name=BUCKET_NAME,
                local_path=str(local_path),
                object_path=input_model_path_str
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            # input_model_path를 local_path로 업데이트
            input_model_path = str(local_path)

        _ = FileHandler.get_default_model_path(folder_path=output_dir)
        extension = FileHandler.get_extension(framework=target_framework)
        object_path = f"{project.user_id}/{project.project_id}/{input_model.model_id}/model{extension}"

        # 모델 이름 생성
        model_name_parts = [
            input_model.name,
            target_framework,
            target_device_name,
        ]

        if target_software_version:  # None이 아닌 경우에만 추가
            model_name_parts.append(target_software_version)

        model_name_parts.append(target_data_type)
        model_name = "_".join(model_name_parts)

        logger.info(f"Model name: {model_name}")
        logger.info(f"Object path: {object_path}")

        model = self.save_model(
            model_name=model_name,
            project_id=input_model.project_id,
            user_id=self.user_info.user_id,
            object_path=object_path,
        )
        conversion_task = self.create_conversion_task(
            framework=target_framework,
            device_name=target_device_name,
            software_version=target_software_version,
            data_type=target_data_type,
            input_model_id=input_model_id,
            model_id=model.model_id,
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

            # Start convert task
            convert_response = launcher_client_v2.converter.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                target_device_name=target_device_name,
                target_framework=target_framework,
                data_type=target_data_type,
                input_layer=input_layer if input_layer else validate_model_response.data.detail.input_layers[0],
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
                logger.info(f"Conversion task was running. Status: {convert_response.data.status}")
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
                logger.info("Conversion task was completed successfully.")
            elif convert_response.data.status in [
                TaskStatusForDisplay.ERROR,
                TaskStatusForDisplay.USER_CANCEL,
                TaskStatusForDisplay.TIMEOUT,
            ]:
                conversion_task.status = Status.ERROR
                conversion_task.error_detail = convert_response.data.error_log
                conversion_task = self._save_conversion_task(conversion_task)
                logger.error(f"Conversion task was failed. Error: {convert_response.data.error_log}")

        except Exception as e:
            conversion_task.status = Status.ERROR
            conversion_task.error_detail = e.args[0]
        except KeyboardInterrupt:
            conversion_task.status = Status.STOPPED
        finally:
            conversion_task = self._save_conversion_task(conversion_task)

        return conversion_task.task_id

    def get_conversion_task(self, conversion_task_id: str) -> ConvertTask:
        """Get the conversion task information with given conversion task uuid.

        Args:
            conversion_task_id (str): Convert task UUID of the convert task.

        Raises:
            e: If an error occurs during the model conversion.

        Returns:
            ConversionTask: Model conversion task dictionary.
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
            conversion_task_id (str): Convert task UUID of the convert task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            ConversionTask: Model conversion task dictionary.
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
