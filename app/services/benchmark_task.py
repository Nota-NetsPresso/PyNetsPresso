from pathlib import Path
from typing import List

from loguru import logger
from sqlalchemy.orm import Session

from app.api.v1.schemas.device import (
    BenchmarkResultPayload,
    HardwareTypePayload,
    PrecisionForBenchmarkPayload,
    SoftwareVersionPayload,
    SupportedDeviceForBenchmarkPayload,
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
from app.worker.benchmark_task import benchmark_model
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums.metadata import Status
from netspresso.enums.model import Framework
from netspresso.enums.project import SubFolder
from netspresso.enums.task import TaskStatusForDisplay
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.models.base import generate_uuid
from netspresso.utils.db.models.benchmark import BenchmarkTask
from netspresso.utils.db.repositories.base import Order, TimeSort
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository


class BenchmarkTaskService:
    def _create_device_payload(self, device_info: DeviceInfo, input_model_id: str, data_type: str) -> SupportedDeviceForBenchmarkPayload:
        """Create device payload from device information"""
        return SupportedDeviceForBenchmarkPayload(
            input_model_id=input_model_id,
            name=device_info.device_name,
            software_version=(
                device_info.software_versions[0].software_version if device_info.software_versions else None
            ),
            data_type=data_type,
            hardware_type=device_info.hardware_types[0] if device_info.hardware_types else None,
        )

    def _create_benchmark_payload(self, benchmark_task: BenchmarkTask) -> BenchmarkPayload:
        """Create benchmark payload from task"""
        return BenchmarkPayload(
            task_id=benchmark_task.task_id,
            model_id=benchmark_task.model_id,
            framework=TargetFrameworkPayload(name=benchmark_task.framework),
            device=TargetDevicePayload(name=benchmark_task.device_name),
            software_version=SoftwareVersionPayload(name=benchmark_task.software_version) if benchmark_task.software_version else None,
            hardware_type=HardwareTypePayload(name=benchmark_task.hardware_type) if benchmark_task.hardware_type else None,
            precision=PrecisionForBenchmarkPayload(name=benchmark_task.precision),
            result=BenchmarkResultPayload(**benchmark_task.result.__dict__) if benchmark_task.result else BenchmarkResultPayload(),
            status=benchmark_task.status,
            is_deleted=benchmark_task.is_deleted,
            error_detail=benchmark_task.error_detail,
            input_model_id=benchmark_task.input_model_id,
            created_at=benchmark_task.created_at,
            updated_at=benchmark_task.updated_at,
        )

    def get_supported_devices(
        self, db: Session, model_id: str, api_key: str
    ) -> List[SupportedDeviceForBenchmarkPayload]:
        """Get list of supported devices for benchmark"""
        netspresso = NetsPresso(api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if model.type not in [SubFolder.TRAINED_MODELS, SubFolder.COMPRESSED_MODELS]:
            raise ValueError("Model is not a trained or compressed model")

        unique_conversions = conversion_task_repository.get_unique_completed_tasks(db=db, model_id=model_id)
        logger.info(f"Found {len(unique_conversions)} unique completed conversions")

        unique_device_keys = set()
        unique_devices = []
        checked_combinations = set()

        for conversion_task in unique_conversions:
            framework = conversion_task.framework
            data_type = conversion_task.precision
            input_model_id = conversion_task.model_id
            is_device_specific = framework in [Framework.TENSORRT, Framework.DRPAI]

            framework_data_type = (framework, data_type)
            if not is_device_specific and framework_data_type in checked_combinations:
                continue

            checked_combinations.add(framework_data_type)
            device = conversion_task.device_name
            software_version = conversion_task.software_version if conversion_task.software_version else None

            _supported_options = benchmarker.get_supported_options(
                framework=framework, device=device, software_version=software_version
            )

            for option in _supported_options:
                if option.framework != framework:
                    continue

                for device_info in option.devices:
                    if is_device_specific and device_info.device_name != device:
                        continue

                    if data_type not in device_info.data_types:
                        continue

                    device_key = (device_info.device_name, input_model_id, data_type)
                    if device_key not in unique_device_keys:
                        unique_device_keys.add(device_key)
                        device_payload = self._create_device_payload(device_info, input_model_id, data_type)
                        unique_devices.append(device_payload)

        return unique_devices

    def create_benchmark_task(self, db: Session, benchmark_in: BenchmarkCreate, api_key: str) -> BenchmarkCreatePayload:
        """Create new benchmark task"""
        # Check if a task with the same options already exists
        existing_tasks = benchmark_task_repository.get_all_by_model_id(
            db=db,
            model_id=benchmark_in.input_model_id
        )

        # Filter tasks by benchmark parameters
        for task in existing_tasks:
            # Check if this task has the same benchmark parameters
            is_same_options = (
                task.device_name == benchmark_in.device_name and
                task.hardware_type == benchmark_in.hardware_type
            )

            # Software version can be None, handle it separately
            is_same_software_version = (
                benchmark_in.software_version is None or
                task.software_version == benchmark_in.software_version
            )

            if is_same_options and is_same_software_version:
                # If task is in NOT_STARTED, IN_PROGRESS, or COMPLETED state, return it
                reusable_states = [Status.NOT_STARTED, Status.IN_PROGRESS, Status.COMPLETED]
                if task.status in reusable_states:
                    logger.info(f"Returning existing benchmark task with status {task.status}: {task.task_id}")
                    return BenchmarkCreatePayload(task_id=task.task_id)

                # For STOPPED or ERROR, we'll create a new task below
                logger.info(f"Previous benchmark task ended with status {task.status}, creating new task")
                break

        model = model_repository.get_by_model_id(db=db, model_id=benchmark_in.input_model_id)
        project = project_service.get_project(db=db, project_id=model.project_id, api_key=api_key)

        input_model_path = Path(project.project_abs_path) / model.object_path
        logger.info(f"Input model path: {input_model_path}")
        logger.info(f"Benchmark Info: {benchmark_in.model_dump()}")

        benchmark_task_id = generate_uuid(entity="task")
        _ = benchmark_model.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_path": input_model_path.as_posix(),
                "target_device_name": benchmark_in.device_name,
                "target_software_version": benchmark_in.software_version,
                "target_hardware_type": benchmark_in.hardware_type,
                "input_model_id": benchmark_in.input_model_id,
            },
            benchmark_task_id=benchmark_task_id,
        )

        return BenchmarkCreatePayload(task_id=benchmark_task_id)

    def get_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkResponse:
        """Get benchmark task status and details"""
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

        return self._create_benchmark_payload(benchmark_task)

    def get_benchmark_tasks(self, db: Session, model_id: str, api_key: str) -> List[BenchmarkPayload]:
        """Get benchmark tasks for a model"""
        conversion_tasks = conversion_task_service.get_conversion_tasks(db, model_id, api_key)

        if not conversion_tasks:
            return []

        conv_model_ids = [task.model_id for task in conversion_tasks]

        benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
            db=db,
            converted_model_ids=conv_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        return [self._create_benchmark_payload(benchmark_task) for benchmark_task in benchmark_tasks]

    def cancel_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkPayload:
        """Cancel benchmark task"""
        netspresso = NetsPresso(api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

        launcher_status = benchmarker.cancel_benchmark_task(benchmark_task.benchmark_task_id)

        if launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
            benchmark_task.status = Status.STOPPED
            benchmark_task = benchmark_task_repository.save(db, benchmark_task)
        else:
            raise ValueError(f"Failed to cancel benchmark task: {launcher_status.status}")

        return self._create_benchmark_payload(benchmark_task)

    def delete_benchmark_task(self, db: Session, task_id: str, api_key: str) -> BenchmarkPayload:
        """Delete benchmark task"""
        benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=task_id)
        benchmark_task = benchmark_task_repository.soft_delete(db=db, model=benchmark_task)

        # Delete benchmarked model from model repository
        model = model_repository.get_by_model_id(db=db, model_id=benchmark_task.model_id)
        model = model_repository.soft_delete(db=db, model=model)

        return self._create_benchmark_payload(benchmark_task)


benchmark_task_service = BenchmarkTaskService()
