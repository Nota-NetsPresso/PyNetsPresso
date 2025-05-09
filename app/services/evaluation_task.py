from typing import List, Optional

from loguru import logger
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
from app.api.v1.schemas.task.evaluation.evaluation_task import EvaluationCreate, EvaluationPayload
from app.worker.evaluation_task import run_multiple_evaluations
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import DataType, DeviceName, SoftwareVersion, Status
from netspresso.enums.conversion import SourceFramework, TargetFramework
from netspresso.exceptions.conversion import ConversionTaskNotFoundException
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.conversion import conversion_task_repository


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
        netspresso = NetsPresso(api_key=api_key)
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
        db: Session,
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

        raise ConversionTaskNotFoundException()

    def create_evaluation_task(
        self,
        db: Session,
        evaluation_in: EvaluationCreate,
        api_key: str,
    ) -> str:
        confidence_scores = [0.3, 0.5, 0.6]

        conversion_task = self._find_existing_conversion_task(
            db=db,
            input_model_id=evaluation_in.input_model_id,
            target_framework=evaluation_in.framework,
            target_device_name=evaluation_in.device_name,
            target_software_version=evaluation_in.software_version,
            target_data_type=evaluation_in.precision
        )

        task_result = run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": conversion_task.model_id,
                "dataset_id": evaluation_in.dataset_id,
                "training_task_id": evaluation_in.training_task_id,
                "confidence_scores": confidence_scores,
            },
        )

        evaluation_task_id = task_result.get(timeout=5)

        logger.info(f"Evaluation task ID: {evaluation_task_id}")

        return evaluation_task_id

    def get_evaluation_tasks(
        self,
        db: Session,
        api_key: str,
    ) -> List[EvaluationPayload]:
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()
        evaluation_tasks = evaluator.get_evaluation_tasks(db=db, user_id=netspresso.user_info.user_id)

        return [EvaluationPayload.model_validate(evaluation_task) for evaluation_task in evaluation_tasks]

    def count_evaluation_task_by_user_id(
        self,
        db: Session,
        api_key: str,
    ) -> int:
        netspresso = NetsPresso(api_key=api_key)
        evaluator = netspresso.evaluator()

        return evaluator.count_evaluation_task_by_user_id(db=db, user_id=netspresso.user_info.user_id)

evaluation_task_service = EvaluationTaskService()
