from typing import List, Optional

from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    HardwareTypePayload,
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
)
from app.api.v1.schemas.task.conversion.conversion_task import (
    TargetFrameworkPayload,
)
from app.api.v1.schemas.task.evaluation.evaluation_task import EvaluationCreate, EvaluationCreateResponse
from app.api.v1.schemas.task.train.dataset import DatasetCreate
from app.api.v1.schemas.task.train.environment import EnvironmentCreate
from app.api.v1.schemas.task.train.hyperparameter import HyperparameterCreate
from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.services.conversion_task import conversion_task_service
from app.services.training_task import train_task_service
from app.services.user import user_service

# from app.worker.celery_app import evaluate_model_task
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import DataType, DeviceName, SoftwareVersion, Status
from netspresso.enums.conversion import SourceFramework, TargetFramework
from netspresso.evaluator.evaluator import Evaluator
from netspresso.exceptions.trainer import NotCompletedTrainingException
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"


class EvaluationTaskService:
    def get_supported_devices(
        self, db: Session, framework: SourceFramework, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            framework (SourceFramework): Framework to get supported devices for
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        converter = netspresso.converter_v2()
        supported_options = converter.get_supported_options(framework=framework)

        supported_framework = [TargetFramework.TENSORFLOW_LITE]

        return [self._create_supported_device_response(option) for option in supported_options if option.framework in supported_framework]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        return SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

    def _create_device_payload(self, device: DeviceInfo) -> SupportedDevicePayload:
        """Create SupportedDevicePayload from device information.

        Args:
            device: Device information containing name, versions, precisions, and hardware types

        Returns:
            SupportedDevicePayload: Payload containing device information
        """
        return SupportedDevicePayload(
            name=device.device_name,
            software_versions=[
                SoftwareVersionPayload(name=version.software_version) for version in device.software_versions
            ],
            precisions=[PrecisionForConversionPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def _find_existing_conversion_task(
        self,
        input_model_id: str,
        target_framework: TargetFramework,
        target_device_name: DeviceName,
        target_software_version: Optional[SoftwareVersion] = None,
        target_data_type: DataType = DataType.FP16
    ) -> Optional[ConversionTask]:
        """Find an existing conversion task that matches the given parameters.

        Args:
            input_model_id: ID of the input model
            target_framework: Target framework for conversion
            target_device_name: Target device for conversion
            target_software_version: Target software version (optional)
            target_data_type: Target data type/precision

        Returns:
            The task_id of the matching conversion task, or None if no match found
        """
        with get_db_session() as db:
            # Find conversion tasks for the input model
            conversion_tasks = conversion_task_repository.get_all_by_model_id(
                db=db,
                model_id=input_model_id
            )

            # Filter tasks by the conversion parameters
            for task in conversion_tasks:
                if (task.framework == target_framework and
                    task.device_name == target_device_name and
                    task.precision == target_data_type and
                    (target_software_version is None or task.software_version == target_software_version) and
                    task.status == Status.COMPLETED):
                    return task

        return None

    # def create_evaluation_task(
    #     self,
    #     db: Session,
    #     evaluation_in: EvaluationCreate,
    #     api_key: str,
    # ) -> EvaluationCreateResponse:
    #     """Create an evaluation task.

    #     Args:
    #         db (Session): Database session
    #         evaluation_in (EvaluationCreate): Evaluation creation request
    #         api_key (str): API key for authentication
    #         target_framework (TargetFramework): Target framework for conversion if needed
    #         target_device_name (DeviceName): Target device for conversion if needed
    #         target_software_version (SoftwareVersion, optional): Target software version
    #         target_data_type (DataType): Target data type/precision

    #     Returns:
    #         EvaluationCreateResponse: Response containing evaluation task
    #     """
    #     training_task = train_task_service.get_training_task(db=db, task_id=evaluation_in.training_task_id, api_key=api_key)

    #     if training_task.status != Status.COMPLETED:
    #         raise NotCompletedTrainingException(training_task_id=evaluation_in.training_task_id)

    #     netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

    #     # 1. Check if a matching conversion task already exists
    #     conversion_task = self._find_existing_conversion_task(
    #         input_model_id=evaluation_in.input_model_id,
    #         target_framework=evaluation_in.framework,
    #         target_device_name=evaluation_in.device_name,
    #         target_software_version=evaluation_in.software_version,
    #         target_data_type=evaluation_in.precision
    #     )

    #     # 2. If no matching conversion task exists, create a new one
    #     if conversion_task is None:
    #         # Get converter instance
    #         converter = netspresso.converter_v2()

    #         # Start conversion process
    #         conversion_task_id = converter.convert_model_from_id(
    #             input_model_id=evaluation_in.input_model_id,
    #             target_framework=evaluation_in.framework,
    #             target_device_name=evaluation_in.device_name,
    #             target_software_version=evaluation_in.software_version,
    #             target_data_type=evaluation_in.precision,
    #             wait_until_done=True  # Wait for conversion to complete
    #         )

    #     confidence_score: float = 0.3

    #     task = evaluate_model_task.delay(
    #         api_key=api_key,
    #         model_id=conversion_task.model_id,
    #         dataset_id=evaluation_in.dataset_id,
    #         training_task_id=evaluation_in.training_task_id,
    #         conversion_task_id=conversion_task.task_id,
    #         confidence_score=confidence_score,
    #     )
    #     task_id = task.get()
    #     return EvaluationCreateResponse(task_id=task_id)


evaluation_task_service = EvaluationTaskService()
