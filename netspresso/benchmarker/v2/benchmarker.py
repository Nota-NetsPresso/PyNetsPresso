import time
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

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
from netspresso.utils import FileHandler
from netspresso.utils.db.models.benchmark import BenchmarkResult, BenchmarkTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.session import get_db_session
from netspresso.utils.metadata import MetadataHandler


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
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id, user_id=user_id)
            return input_model

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

    def save_benchmark_result(self, benchmark_task: BenchmarkTask, benchmark_result, file_size_in_mb) -> BenchmarkTask:
        benchmark_result = BenchmarkResult(
            processor=benchmark_result.processor,
            memory_footprint_gpu=benchmark_result.memory_footprint_gpu,
            memory_footprint_cpu=benchmark_result.memory_footprint_cpu,
            power_consumption=benchmark_result.power_consumption,
            ram_size=benchmark_result.ram_size,
            latency=benchmark_result.latency,
            file_size=file_size_in_mb,
        )

        with get_db_session() as db:
            benchmark_task.result = benchmark_result
            benchmark_task = benchmark_task_repository.save(db=db, model=benchmark_task)

            return benchmark_task

    def create_benchmark_task(
        self,
        device_name: Union[str, DeviceName],
        software_version: Union[str, SoftwareVersion],
        data_type: Union[str, DataType],
        input_model_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> BenchmarkTask:
        with get_db_session() as db:
            benchmark_task = BenchmarkTask(
                device_name=device_name,
                software_version=software_version,
                precision=data_type,
                status=Status.NOT_STARTED,
                input_model_id=input_model_id,
                model_id=model_id,
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

        FileHandler.check_input_model_path(input_model_path)
        output_dir = Path(input_model_path).parent

        if input_model_id:
            input_model = self.get_input_model(input_model_id, self.user_info.user_id)
            input_model.user_id = self.user_info.user_id

        data_type = self.get_data_type(output_dir)
        model = self.save_model(
            model_name=f"{input_model.name}_benchmarked",
            project_id=input_model.project_id,
            user_id=self.user_info.user_id,
            object_path=input_model_path,
        )
        benchmark_task = self.create_benchmark_task(
            device_name=target_device_name,
            software_version=target_software_version,
            data_type=data_type,
            input_model_id=input_model_id,
            model_id=model.model_id,
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
            benchmark_task.framework = benchmark_response.data.benchmark_task_option.framework
            benchmark_task = self._save_benchmark_task(benchmark_task)

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
                benchmark_task = self.save_benchmark_result(benchmark_task, _benchmark_result, validate_model_response.data.file_size_in_mb)

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
