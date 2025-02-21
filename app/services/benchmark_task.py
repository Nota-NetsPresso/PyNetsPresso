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
from app.services.project import project_service
from app.services.user import user_service
from app.worker.celery_app import benchmark_model_task
from netspresso.clients.launcher.v2.schemas.common import DeviceInfo
from netspresso.enums.metadata import Status
from netspresso.enums.model import Framework
from netspresso.enums.project import SubFolder
from netspresso.enums.task import TaskStatusForDisplay
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository


class BenchmarkTaskService:
    def get_supported_devices(self, db: Session, model_id: str, api_key: str) -> List[SupportedDeviceForBenchmarkPayload]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()

        model = model_repository.get_by_model_id(db=db, model_id=model_id, user_id=netspresso.user_info.user_id)
        if model.type not in [SubFolder.TRAINED_MODELS, SubFolder.COMPRESSED_MODELS]:
            raise ValueError("Model is not a trained or compressed model")

        unique_conversions = conversion_task_repository.get_unique_completed_tasks(db=db, model_id=model_id)
        logger.info(f"Found {len(unique_conversions)} unique completed conversions")
        for conv in unique_conversions:
            logger.info(f"Conversion: framework={conv.framework}, precision={conv.precision}, device={conv.device_name}")

        unique_device_keys = set()
        unique_devices = []
        checked_combinations = set()

        for conversion_task in unique_conversions:
            framework = conversion_task.framework
            data_type = conversion_task.precision
            input_model_id = conversion_task.model_id

            is_device_specific = framework in [Framework.TENSORRT, Framework.DRPAI]
            logger.info(f"\nProcessing conversion: framework={framework}, data_type={data_type}")
            logger.info(f"Is device specific: {is_device_specific}")

            framework_data_type = (framework, data_type)
            if not is_device_specific and framework_data_type in checked_combinations:
                logger.info(f"Skipping framework {framework} with data_type {data_type} - already checked")
                continue

            checked_combinations.add(framework_data_type)

            if is_device_specific:
                device = conversion_task.device_name
                software_version = conversion_task.software_version if conversion_task.software_version else None
                logger.info(f"Using specific device: {device} with sw version: {software_version}")
            else:
                device = conversion_task.device_name
                software_version = None
                logger.info("Using all available devices")

            _supported_options = benchmarker.get_supported_options(
                framework=framework,
                device=device,
                software_version=software_version
            )

            for option in _supported_options:
                if option.framework != framework:
                    logger.info(f"Skipping option with framework {option.framework} - doesn't match {framework}")
                    continue

                for device_info in option.devices:
                    if is_device_specific and device_info.device_name != device:
                        logger.info(f"Skipping device {device_info.device_name} - doesn't match {device}")
                        continue

                    logger.info(f"\nChecking device {device_info.device_name} for {data_type}")
                    logger.info(f"Device supports: {device_info.data_types}")

                    if data_type not in device_info.data_types:
                        logger.info(f"Skipping {data_type} for device {device_info.device_name} - not supported")
                        continue

                    device_key = self._create_device_key(device_info, input_model_id, data_type)
                    logger.info(f"Device key: {device_key}")

                    if device_key not in unique_device_keys:
                        unique_device_keys.add(device_key)
                        device_payload = self._create_device_payload(
                            input_model_id=input_model_id,
                            device_info=device_info,
                            data_type=data_type
                        )
                        unique_devices.append(device_payload)
                        logger.info(f"Added device: {device_payload}")
                    else:
                        logger.info("Device already added - skipping")

        logger.info(f"\nFinal device count: {len(unique_devices)}")
        return unique_devices

    def _create_device_key(self, device_info: DeviceInfo, input_model_id: str, data_type: str) -> tuple:
        """디바이스 키에 input_model_id와 data_type을 포함하여 생성"""
        base_key = (
            device_info.device_name,
            tuple(v.software_version for v in device_info.software_versions),
            tuple(device_info.hardware_types),
            input_model_id,
            data_type
        )
        return base_key

    def _create_device_payload(
        self,
        input_model_id: str,
        device_info: DeviceInfo,
        data_type: str
    ) -> SupportedDeviceForBenchmarkPayload:
        """Create device payload with specific data type.

        Args:
            input_model_id: ID of the converted model
            device_info: Device information from launcher
            data_type: Data type to use for this device

        Returns:
            SupportedDeviceForBenchmarkPayload: Device payload for response
        """
        return SupportedDeviceForBenchmarkPayload(
            input_model_id=input_model_id,
            name=device_info.device_name,
            software_version=device_info.software_versions[0].software_version if device_info.software_versions else None,
            data_type=data_type,
            hardware_type=device_info.hardware_types[0] if device_info.hardware_types else None
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
        if benchmark_task.result:
            result = BenchmarkResultPayload(
                memory_footprint_gpu=benchmark_task.result.memory_footprint_gpu,
                memory_footprint_cpu=benchmark_task.result.memory_footprint_cpu,
                power_consumption=benchmark_task.result.power_consumption,
                ram_size=benchmark_task.result.ram_size,
                latency=benchmark_task.result.latency,
                file_size=benchmark_task.result.file_size,
            )
        else:
            result = BenchmarkResultPayload()

        benchmark_payload = BenchmarkPayload(
            task_id=benchmark_task.task_id,
            model_id=benchmark_task.model_id,
            framework=framework,
            device=device,
            software_version=software_version,
            hardware_type=hardware_type,
            precision=precision,
            result=result,
            status=benchmark_task.status,
            is_deleted=benchmark_task.is_deleted,
            error_detail=benchmark_task.error_detail,
            input_model_id=benchmark_task.input_model_id,
            created_at=benchmark_task.created_at,
            updated_at=benchmark_task.updated_at,
        )

        return benchmark_payload

    def cancel_benchmark_task(self, db: Session, task_id: str, api_key: str):
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        benchmarker = netspresso.benchmarker_v2()
        benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)
        benchmark_task = benchmarker.cancel_benchmark_task(benchmark_task.benchmark_task_id)

        if benchmark_task.status == TaskStatusForDisplay.USER_CANCEL:
            benchmark_task.status = Status.STOPPED
            benchmark_task = benchmark_task_repository.save(db, benchmark_task)
        else:
            raise ValueError(f"Failed to cancel benchmark task: {benchmark_task.status}")

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

