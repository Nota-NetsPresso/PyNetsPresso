from pathlib import Path
from typing import List

from loguru import logger
from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    HardwareTypePayload,
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
    TargetDevicePayload,
)
from app.api.v1.schemas.task.conversion.conversion_task import (
    ConversionCreate,
    ConversionCreatePayload,
    ConversionPayload,
    TargetFrameworkPayload,
)
from app.services.project import project_service
from app.worker.conversion_task import convert_model
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums import Status, TaskStatusForDisplay
from netspresso.enums.conversion import SourceFramework
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.base import generate_uuid
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.base import Order, TimeSort
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository


class ConversionTaskService:
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

        return [self._create_supported_device_response(option) for option in supported_options]

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

    def create_conversion_task(
        self, db: Session, conversion_in: ConversionCreate, api_key: str
    ) -> ConversionCreatePayload:
        # Check if a task with the same options already exists
        existing_tasks = conversion_task_repository.get_all_by_model_id(
            db=db,
            model_id=conversion_in.input_model_id
        )

        # Filter tasks by conversion parameters
        for task in existing_tasks:
            # Check if this task has the same conversion parameters
            is_same_options = (
                task.framework == conversion_in.framework and
                task.device_name == conversion_in.device_name and
                task.precision == conversion_in.precision
            )

            # Software version can be None, handle it separately
            is_same_software_version = (
                conversion_in.software_version is None or
                task.software_version == conversion_in.software_version
            )

            if is_same_options and is_same_software_version:
                # If task is in NOT_STARTED, IN_PROGRESS, or COMPLETED state, return it
                reusable_states = [Status.NOT_STARTED, Status.IN_PROGRESS, Status.COMPLETED]
                if task.status in reusable_states:
                    logger.info(f"Returning existing conversion task with status {task.status}: {task.task_id}")
                    return ConversionCreatePayload(task_id=task.task_id)

                # For STOPPED or ERROR, we'll create a new task below
                logger.info(f"Previous conversion task ended with status {task.status}, creating new task")
                break

        # Get model from trained models repository
        model = model_repository.get_by_model_id(db=db, model_id=conversion_in.input_model_id)
        project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

        # Create output directory path as a 'converted' subfolder of input model path
        project_abs_path = Path(project.project_abs_path)
        input_model_dir = project_abs_path / model.object_path

        input_model_path = input_model_dir / "model.onnx"
        output_dir = input_model_dir / "converted"
        logger.info(f"Input model path: {input_model_path}")
        logger.info(f"Conversion Info: {conversion_in.model_dump()}")
        logger.info(f"Output dir: {output_dir}")

        conversion_task_id = generate_uuid(entity="task")
        _ = convert_model.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_path": input_model_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "target_framework": conversion_in.framework,
                "target_device_name": conversion_in.device_name,
                "target_data_type": conversion_in.precision,
                "target_software_version": conversion_in.software_version,
                "input_model_id": conversion_in.input_model_id,
            },
            conversion_task_id=conversion_task_id,
        )
        return ConversionCreatePayload(task_id=conversion_task_id)

    def get_conversion_task(self, db: Session, task_id: str, api_key: str) -> ConversionPayload:
        conversion_task = conversion_task_repository.get_by_task_id(db, task_id)

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload

    def cancel_conversion_task(self, db: Session, task_id: str, api_key: str):
        netspresso = NetsPresso(api_key=api_key)
        converter = netspresso.converter_v2()
        conversion_task = conversion_task_repository.get_by_task_id(db, task_id)
        convert_task = converter.cancel_conversion_task(conversion_task.convert_task_uuid)

        if convert_task.status == TaskStatusForDisplay.USER_CANCEL:
            conversion_task.status = Status.STOPPED
            conversion_task = conversion_task_repository.save(db, conversion_task)
        else:
            raise ValueError(f"Failed to cancel conversion task: {convert_task.status}")

        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        conversion_payload = ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

        return conversion_payload

    def _create_conversion_payload(self, conversion_task: ConversionTask) -> ConversionPayload:
        framework = TargetFrameworkPayload(name=conversion_task.framework)
        device = TargetDevicePayload(name=conversion_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=conversion_task.software_version) if conversion_task.software_version else None
        )
        precision = PrecisionForConversionPayload(name=conversion_task.precision)

        return ConversionPayload(
            task_id=conversion_task.task_id,
            model_id=conversion_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            precision=precision,
            status=conversion_task.status,
            is_deleted=conversion_task.is_deleted,
            error_detail=conversion_task.error_detail,
            input_model_id=conversion_task.input_model_id,
            created_at=conversion_task.created_at,
            updated_at=conversion_task.updated_at,
        )

    def get_conversion_tasks(self, db: Session, model_id: str, api_key: str) -> List[ConversionPayload]:
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db, model_id=model_id, order=Order.DESC, time_sort=TimeSort.CREATED_AT,
        )

        return [self._create_conversion_payload(task) for task in conversion_tasks]

    def delete_conversion_task(self, db: Session, task_id: str, api_key: str) -> ConversionPayload:
        conversion_task = conversion_task_repository.get_by_task_id(db=db, task_id=task_id)
        conversion_task = conversion_task_repository.soft_delete(db=db, model=conversion_task)

        # Delete converted model from model repository
        model = model_repository.get_by_model_id(db=db, model_id=conversion_task.model_id)
        model = model_repository.soft_delete(db=db, model=model)

        return self._create_conversion_payload(conversion_task)


conversion_task_service = ConversionTaskService()
