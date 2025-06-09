import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.base import NetsPressoBase
from netspresso.clients.auth import TokenHandler
from netspresso.clients.auth.response_body import UserResponse
from netspresso.clients.compressor.v2.schemas.common import DeviceInfo
from netspresso.clients.launcher import launcher_client_v2
from netspresso.clients.launcher.v2.schemas.common import ModelOption
from netspresso.clients.launcher.v2.schemas.task.benchmark.response_body import BenchmarkTask as BenchmarkTaskInfo
from netspresso.enums import Status, TaskStatusForDisplay
from netspresso.enums.conversion import TargetFramework
from netspresso.enums.credit import ServiceTask
from netspresso.enums.device import DeviceName, HardwareType, SoftwareVersion
from netspresso.enums.model import DataType
from netspresso.enums.project import SubFolder
from netspresso.metadata.benchmarker import BenchmarkerMetadata
from netspresso.utils.db.models.benchmark import BenchmarkResult, BenchmarkTask
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.session import get_db_session
from netspresso.utils.metadata import MetadataHandler

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"

class BenchmarkerV2(NetsPressoBase):
    def __init__(self, token_handler: TokenHandler, user_info: UserResponse) -> None:
        """Initialize the Benchmarker."""

        super().__init__(token_handler)
        self.user_info = user_info

    def filter_device_by_version(
        self, device: DeviceInfo, target_software_version: Optional[SoftwareVersion] = None
    ) -> Optional[DeviceInfo]:
        """Filter device by software version.

        Args:
            device: Device information to filter
            target_software_version: Target software version to filter by

        Returns:
            Optional[DeviceInfo]: Filtered device info or None if no matching version
        """
        if target_software_version is None and device.device_name == DeviceName.AWS_T4:
            return device

        filtered_versions = [
            version for version in device.software_versions if version.software_version == target_software_version
        ]

        if filtered_versions:
            device.software_versions = filtered_versions
            return device
        return None

    def filter_devices_by_name_and_version(
        self, devices: List[DeviceInfo], target_device: DeviceName, target_version: Optional[SoftwareVersion] = None
    ) -> List[DeviceInfo]:
        """Filter devices by name and software version.

        Args:
            devices: List of devices to filter
            target_device: Target device name to filter by
            target_version: Target software version to filter by

        Returns:
            List[DeviceInfo]: List of filtered devices
        """
        filtered_devices = [
            self.filter_device_by_version(device, target_version)
            for device in devices
            if device.device_name == target_device
        ]
        return [device for device in filtered_devices if device]

    def get_supported_options(
        self, framework: TargetFramework, device: DeviceName, software_version: Optional[SoftwareVersion] = None
    ) -> List[ModelOption]:
        """Get supported options for given framework, device and software version.

        Args:
            framework: Target framework
            device: Target device name
            software_version: Target software version

        Returns:
            List[ModelOption]: List of supported model options
        """
        self.token_handler.validate_token()

        # Get all options from launcher
        options_response = launcher_client_v2.benchmarker.read_framework_options(
            access_token=self.token_handler.tokens.access_token,
            framework=framework,
        )
        supported_options = options_response.data

        # Filter options for specific frameworks
        if framework in [TargetFramework.TENSORRT, TargetFramework.DRPAI]:
            for option in supported_options:
                if option.framework == framework:
                    option.devices = self.filter_devices_by_name_and_version(
                        devices=option.devices, target_device=device, target_version=software_version
                    )

        return supported_options

    def get_data_type(self, input_model_dir):
        metadata_path = input_model_dir / "metadata.json"
        try:
            if metadata_path.exists():
                metadata = MetadataHandler.load_json(metadata_path)
                convert_task_info = metadata.get("convert_task_info", {})
                return convert_task_info.get("data_type", DataType.FP32)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Error loading metadata: {e}")

        return DataType.FP32

    def get_input_model(self, input_model_id: str, user_id: str) -> Model:
        with get_db_session() as db:
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)
            return input_model

    def get_conversion_task(self, input_model_id: str) -> ConversionTask:
        with get_db_session() as db:
            conversion_task = conversion_task_repository.get_by_model_id(db=db, model_id=input_model_id)
            return conversion_task

    def save_model(self, model_name, project_id, user_id, object_path) -> Model:
        model = Model(
            name=model_name,
            type=SubFolder.BENCHMARKED_MODELS,
            is_retrainable=False,
            project_id=project_id,
            user_id=user_id,
            object_path=object_path,
        )
        with get_db_session() as db:
            model = model_repository.save(db=db, model=model)
            return model

    def _save_benchmark_task(self, benchmark_task: BenchmarkTask) -> BenchmarkTask:
        with get_db_session() as db:
            benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)

            return benchmark_task

    def save_benchmark_result(self, benchmark_task_id: str, benchmark_result: BenchmarkResult) -> BenchmarkTask:
        """결과를 객체 공유 없이 태스크 ID로 저장"""
        with get_db_session() as db:
            benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=benchmark_task_id)
            if not benchmark_task:
                return

            result = BenchmarkResult(
                processor=benchmark_result.processor,
                memory_footprint_gpu=benchmark_result.memory_footprint_gpu,
                memory_footprint_cpu=benchmark_result.memory_footprint_cpu,
                power_consumption=benchmark_result.power_consumption,
                ram_size=benchmark_result.ram_size,
                latency=benchmark_result.latency,
                task_id=benchmark_task_id
            )
            benchmark_task.result = result
            db.add(benchmark_task)
            db.commit()

            return benchmark_task

    def create_benchmark_result(self, benchmark_task: BenchmarkTask, file_size: float) -> BenchmarkTask:
        benchmark_result = BenchmarkResult(file_size=file_size, task_id=benchmark_task.task_id)
        with get_db_session() as db:
            benchmark_task.result = benchmark_result
            benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)

            return benchmark_task

    def create_benchmark_task(
        self,
        framework: TargetFramework,
        device_name: Union[str, DeviceName],
        software_version: Union[str, SoftwareVersion],
        data_type: Union[str, DataType],
        input_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
        benchmark_task_id: Optional[str] = None,
    ) -> BenchmarkTask:
        with get_db_session() as db:
            if benchmark_task_id:
                benchmark_task = BenchmarkTask(
                    task_id=benchmark_task_id,
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
                benchmark_task = BenchmarkTask(
                    framework=framework,
                    device_name=device_name,
                    software_version=software_version,
                    precision=data_type,
                    status=Status.NOT_STARTED,
                    input_model_id=input_model_id,
                    model_id=model_id,
                    user_id=self.user_info.user_id,
                )
            benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)
            return benchmark_task

    def benchmark_model(
        self,
        input_model_path: str,
        target_device_name: DeviceName,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
        sleep_interval: int = 30,
        input_model_id: Optional[str] = None,
        benchmark_task_id: Optional[str] = None,
    ) -> BenchmarkerMetadata:
        """Benchmark the specified model on the specified device.

        Args:
            input_model_path (str): The file path where the model is located.
            target_device_name (DeviceName): Target device name.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.
            target_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.
            wait_until_done (bool): If True, wait for the benchmark result before returning the function.
                                If False, request the benchmark and return the function immediately.

        Raises:
            e: If an error occurs during the benchmarking of the model.

        Returns:
            BenchmarkerMetadata: Benchmark metadata.
        """
        # 임시 디렉토리 생성을 위한 변수 초기화
        temp_dir = None

        if input_model_id:
            input_model = self.get_input_model(input_model_id, self.user_info.user_id)
            input_model.user_id = self.user_info.user_id
            input_model_path = Path(input_model.object_path)
            conversion_task = self.get_conversion_task(input_model_id)
            framework = conversion_task.framework
            data_type = conversion_task.precision

            # 임시 디렉토리 생성 - output_dir 오류 수정
            temp_dir = tempfile.mkdtemp(prefix="netspresso_benchmark_")
            download_dir = Path(temp_dir) / "input_model"
            download_dir.mkdir(parents=True, exist_ok=True)  # 다운로드 폴더 생성

            local_path = download_dir / input_model_path.name
            input_model_path_str = str(input_model_path)

            logger.info(f"Downloading input model from Zenko to temp directory: {temp_dir}")
            storage_handler.download_file_from_s3(
                bucket_name=BUCKET_NAME,
                local_path=str(local_path),
                object_path=input_model_path_str
            )
            logger.info(f"Downloaded input model from Zenko: {local_path}")

            # input_model_path를 local_path로 업데이트
            input_model_path = str(local_path)

        model = self.save_model(
            model_name=f"{input_model.name}_benchmarked",
            project_id=input_model.project_id,
            user_id=self.user_info.user_id,
            object_path=input_model_path,
        )
        benchmark_task = self.create_benchmark_task(
            framework=framework,
            device_name=target_device_name,
            software_version=target_software_version,
            data_type=data_type,
            input_model_id=input_model_id,
            model_id=model.model_id,
            benchmark_task_id=benchmark_task_id,
        )

        try:

            self.validate_token_and_check_credit(service_task=ServiceTask.MODEL_BENCHMARK)

            # Get presigned_model_upload_url
            presigned_url_response = launcher_client_v2.benchmarker.presigned_model_upload_url(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
            )

            # Upload model_file
            launcher_client_v2.benchmarker.upload_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                presigned_upload_url=presigned_url_response.data.presigned_upload_url,
            )

            # Validate model_file
            validate_model_response = launcher_client_v2.benchmarker.validate_model_file(
                access_token=self.token_handler.tokens.access_token,
                input_model_path=input_model_path,
                ai_model_id=presigned_url_response.data.ai_model_id,
            )

            # Start benchmark task
            benchmark_response = launcher_client_v2.benchmarker.start_task(
                access_token=self.token_handler.tokens.access_token,
                input_model_id=presigned_url_response.data.ai_model_id,
                data_type=validate_model_response.data.detail.data_type,
                target_device_name=target_device_name,
                hardware_type=target_hardware_type,
                input_layer=validate_model_response.data.detail.input_layers[0],
                software_version=target_software_version,
            )

            benchmark_task.benchmark_task_id = benchmark_response.data.benchmark_task_id
            benchmark_task = self._save_benchmark_task(benchmark_task)
            benchmark_task = self.create_benchmark_result(benchmark_task, validate_model_response.data.file_size_in_mb)

            if wait_until_done:
                while True:
                    self.token_handler.validate_token()
                    benchmark_response = launcher_client_v2.benchmarker.read_task(
                        access_token=self.token_handler.tokens.access_token,
                        task_id=benchmark_response.data.benchmark_task_id,
                    )
                    if benchmark_response.data.status in [
                        TaskStatusForDisplay.FINISHED,
                        TaskStatusForDisplay.ERROR,
                        TaskStatusForDisplay.TIMEOUT,
                        TaskStatusForDisplay.USER_CANCEL,
                    ]:
                        break

                    time.sleep(sleep_interval)

            if benchmark_response.data.status in [TaskStatusForDisplay.IN_PROGRESS, TaskStatusForDisplay.IN_QUEUE]:
                benchmark_task.status = Status.IN_PROGRESS
                logger.info(f"Benchmark task was running. Status: {benchmark_response.data.status}")
            elif benchmark_response.data.status == TaskStatusForDisplay.FINISHED:
                self.print_remaining_credit(service_task=ServiceTask.MODEL_BENCHMARK)
                benchmark_task.status = Status.COMPLETED

                # Save benchmark results
                _benchmark_result = benchmark_response.data.benchmark_result
                benchmark_task = self._save_benchmark_task(benchmark_task)
                benchmark_task = self.save_benchmark_result(benchmark_task.task_id, _benchmark_result)

                logger.info("Benchmark task was completed successfully.")
            elif benchmark_response.data.status in [
                TaskStatusForDisplay.ERROR,
                TaskStatusForDisplay.USER_CANCEL,
                TaskStatusForDisplay.TIMEOUT,
            ]:
                benchmark_task.status = Status.ERROR
                benchmark_task.error_detail = benchmark_response.data.error_log
                benchmark_task = self._save_benchmark_task(benchmark_task)
                logger.error(f"Benchmark task was failed. Error: {benchmark_response.data.error_log}")

        except Exception as e:
            benchmark_task.status = Status.ERROR
            benchmark_task.error_detail = e.args[0]
        except KeyboardInterrupt:
            benchmark_task.status = Status.STOPPED
        finally:
            benchmark_task = self._save_benchmark_task(benchmark_task)

            # 임시 파일 및 디렉토리 정리
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary files in: {temp_dir}")
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Successfully removed temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary files: {cleanup_error}")

        return benchmark_task.task_id

    def get_benchmark_task(self, benchmark_task_id: str) -> BenchmarkTaskInfo:
        """Get information about the specified benchmark task using the benchmark task UUID.

        Args:
            benchmark_task_id (str): Benchmark task UUID of the benchmark task.

        Raises:
            e: If an error occurs while retrieving information about the benchmark task.

        Returns:
            BenchmarkTaskInfo: Model benchmark task object.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.read_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=benchmark_task_id,
        )
        return response.data

    def cancel_benchmark_task(self, benchmark_task_id: str) -> BenchmarkTaskInfo:
        """Cancel the benchmark task with given benchmark task uuid.

        Args:
            benchmark_task_id (str): Benchmark task UUID of the benchmark task.

        Raises:
            e: If an error occurs during the task cancel.

        Returns:
            BenchmarkTaskInfo: Model benchmark task dictionary.
        """

        self.token_handler.validate_token()

        response = launcher_client_v2.benchmarker.cancel_task(
            access_token=self.token_handler.tokens.access_token,
            task_id=benchmark_task_id,
        )
        return response.data

    def update_benchmark_task_status(self, task_id: str) -> bool:
        """Update benchmark task status in DB based on launcher status.

        Args:
            task_id (str): Benchmark task ID to update

        Returns:
            bool: True if status was updated, False if task is still in progress
        """
        try:
            with get_db_session() as db:
                # 매번 새 세션에서 객체를 가져옴
                benchmark_task = benchmark_task_repository.get_by_task_id(db=db, task_id=task_id)
                if not benchmark_task:
                    logger.error(f"Benchmark task {task_id} not found")
                    return True

                launcher_status = self.get_benchmark_task(benchmark_task.benchmark_task_id)
                status_updated = False

                if launcher_status.status == TaskStatusForDisplay.FINISHED:
                    benchmark_task.status = Status.COMPLETED
                    status_updated = True

                    # 결과 저장을 별도 객체로 처리
                    if launcher_status.benchmark_result:
                        result = BenchmarkResult(
                            processor=launcher_status.benchmark_result.processor,
                            memory_footprint_gpu=launcher_status.benchmark_result.memory_footprint_gpu,
                            memory_footprint_cpu=launcher_status.benchmark_result.memory_footprint_cpu,
                            power_consumption=launcher_status.benchmark_result.power_consumption,
                            ram_size=launcher_status.benchmark_result.ram_size,
                            latency=launcher_status.benchmark_result.latency,
                            task_id=task_id
                        )
                        benchmark_task.result = result

                elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
                    benchmark_task.status = Status.ERROR
                    benchmark_task.error_detail = launcher_status.error_log
                    status_updated = True

                elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
                    benchmark_task.status = Status.STOPPED
                    status_updated = True

                if status_updated:
                    # 현재 세션에서 변경사항 저장
                    db.add(benchmark_task)
                    db.commit()
                    logger.info(f"Benchmark task {task_id} status updated to {benchmark_task.status}")

                return status_updated
        except Exception as e:
            logger.error(f"Error updating benchmark task status: {e}")
            # 오류가 발생해도 태스크가 계속 재시도되지 않도록 True 반환
            return True
