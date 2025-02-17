from pathlib import Path
from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    HardwareTypePayload,
    PrecisionForBenchmarkPayload,
    SoftwareVersionPayload,
    SupportedDevicePayload,
    SupportedDeviceResponse,
    TargetDevicePayload,
)
from app.api.v1.schemas.task.benchmark.benchmark_task import (
    BenchmarkCreate,
    BenchmarkCreatePayload,
    BenchmarkPayload,
    BenchmarkResponse,
    TargetFrameworkPayload,
)
from app.services.conversion_task import conversion_task_service
from app.services.project import project_service
from app.services.user import user_service
from app.worker.celery_app import benchmark_model_task
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums.metadata import Status
from netspresso.enums.task import TaskStatusForDisplay
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.model import model_repository


class BenchmarkTaskService:
    def get_supported_devices(
        self, db: Session, conversion_task_id: str, api_key: str
    ) -> List[SupportedDeviceResponse]:
        """Get supported devices for conversion tasks.

        Args:
            db (Session): Database session
            conversion_task_id (str): Conversion task ID
            api_key (str): API key for authentication

        Returns:
            List[SupportedDeviceResponse]: List of supported devices grouped by framework
        """
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()

        conversion_task = conversion_task_service.get_conversion_task(
            db=db, task_id=conversion_task_id, api_key=api_key
        )

        framework = conversion_task.framework.name
        device = conversion_task.device.name
        software_version = conversion_task.software_version.name if conversion_task.software_version else None

        supported_options = benchmarker.get_supported_options(
            framework=framework, device=device, software_version=software_version
        )

        return [self._create_supported_device_response(option) for option in supported_options]

    def _create_supported_device_response(self, option) -> SupportedDeviceResponse:
        """Create SupportedDeviceResponse from converter option.

        Args:
            option: Converter option containing framework and devices information

        Returns:
            SupportedDeviceResponse: Response containing framework and supported devices
        """
        response = SupportedDeviceResponse(
            framework=TargetFrameworkPayload(name=option.framework),
            devices=[self._create_device_payload(device) for device in option.devices],
        )

        return response

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
            precisions=[PrecisionForBenchmarkPayload(name=precision) for precision in device.data_types],
            hardware_types=[HardwareTypePayload(name=hardware_type) for hardware_type in device.hardware_types],
        )

    def create_benchmark_task(self, db: Session, benchmark_in: BenchmarkCreate, api_key: str) -> BenchmarkCreatePayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        # Get model from trained models repository
        model = model_repository.get_by_model_id(
            db=db, model_id=benchmark_in.input_model_id, user_id=netspresso.user_info.user_id
        )
        project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

        # Create output directory path as a 'converted' subfolder of input model path
        project_abs_path = Path(project.project_abs_path)
        input_model_path = project_abs_path / model.object_path

        print(f"Input model path: {input_model_path}")

        task = benchmark_model_task.delay(
            api_key=api_key,
            input_model_path=input_model_path.as_posix(),
            target_device_name=benchmark_in.device_name,
            target_software_version=benchmark_in.software_version,
            target_hardware_type=benchmark_in.hardware_type,
            input_model_id=benchmark_in.input_model_id,
        )
        task_id = task.get()
        return BenchmarkCreatePayload(task_id=task_id)

    def get_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkResponse:
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()

        if benchmark_task.status == Status.NOT_STARTED or benchmark_task.status == Status.IN_PROGRESS:
            # Check launcher server status
            launcher_status = benchmarker.get_benchmark_task(benchmark_task.benchmark_task_id)

            if launcher_status.status in [TaskStatusForDisplay.FINISHED]:
                benchmark_task.status = Status.COMPLETED
            elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
                benchmark_task.status = Status.ERROR
                benchmark_task.error_detail = launcher_status.error_log
            elif launcher_status.status in [TaskStatusForDisplay.USER_CANCEL]:
                benchmark_task.status = Status.STOPPED

            benchmark_task = benchmark_task_repository.save(db, benchmark_task)

        framework = TargetFrameworkPayload(name=benchmark_task.framework)
        device = TargetDevicePayload(name=benchmark_task.device_name)
        software_version = (
            SoftwareVersionPayload(name=benchmark_task.software_version) if benchmark_task.software_version else None
        )
        hardware_type = (
            HardwareTypePayload(name=benchmark_task.hardware_type) if benchmark_task.hardware_type else None
        )
        precision = PrecisionForBenchmarkPayload(name=benchmark_task.precision)

        benchmark_payload = BenchmarkPayload(
            task_id=benchmark_task.task_id,
            model_id=benchmark_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            hardware_type=hardware_type,
            precision=precision,
            status=benchmark_task.status,
            is_deleted=benchmark_task.is_deleted,
            error_detail=benchmark_task.error_detail,
            input_model_id=benchmark_task.input_model_id,
            created_at=benchmark_task.created_at,
            updated_at=benchmark_task.updated_at,
        )

        return benchmark_payload


benchmark_task_service = BenchmarkTaskService()
