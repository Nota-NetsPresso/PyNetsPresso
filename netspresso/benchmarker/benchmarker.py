import time
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger

from netspresso.clients.auth import TokenHandler, auth_client
from netspresso.clients.auth.schemas.auth import UserInfo
from netspresso.clients.launcher import launcher_client
from netspresso.clients.launcher.schemas import BenchmarkTask, TargetDeviceFilter
from netspresso.enums.credit import ServiceCredit
from netspresso.enums.device import DeviceName, HardwareType, SoftwareVersion, TaskStatus
from netspresso.enums.metadata import Status, TaskType
from netspresso.enums.model import DataType
from netspresso.enums.module import Module

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler


class Benchmarker:
    def __init__(self, token_handler: TokenHandler, user_info: UserInfo) -> None:
        """Initialize the Benchmarker."""

        self.token_handler = token_handler
        self.user_info = user_info

    def benchmark_model(
        self,
        input_model_path: str,
        target_device_name: DeviceName,
        target_data_type: DataType = DataType.FP16,
        target_software_version: Optional[Union[str, SoftwareVersion]] = None,
        target_hardware_type: Optional[Union[str, HardwareType]] = None,
        wait_until_done: bool = True,
    ) -> Dict:
        """Benchmark the specified model on the specified device.

        Args:
            input_model_path (str): The file path where the model is located.
            target_device_name (DeviceName): Target device name.
            target_data_type (DataType): Data type of the model.
            target_software_version (Union[str, SoftwareVersion], optional): Target software version. Required if target_device_name is one of the Jetson devices.
            target_hardware_type (Union[str, HardwareType], optional): Hardware type. Acceleration options for processing the model inference.
            wait_until_done (bool): If True, wait for the conversion result before returning the function.
                                If False, request the conversion and return the function immediately.

        Raises:
            e: If an error occurs during the benchmarking of the model.

        Returns:
            Dict: Model benchmark task dictionary.
        """

        FileHandler.check_input_model_path(input_model_path)

        self.token_handler.validate_token()

        try:
            folder_path = Path(input_model_path).parent

            metadata = MetadataHandler.get_default_metadata(TaskType.BENCHMARK)
            if FileHandler.check_exists(folder_path / "benchmark.json"):
                metadatas = MetadataHandler.load_json(folder_path / "benchmark.json")
                metadatas.append(metadata.asdict())
            else:
                metadatas = [metadata.asdict()]
            MetadataHandler.save_json(metadatas, folder_path, file_name="benchmark")

            current_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, self.token_handler.verify_ssl
            )
            check_credit_balance(user_credit=current_credit, service_credit=ServiceCredit.MODEL_BENCHMARK)
            model = launcher_client.upload_model(
                model_file_path=input_model_path,
                target_function=Module.BENCHMARK,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            model_uuid = model.model_uuid

            if target_device_name in DeviceName.JETSON_DEVICES and target_software_version is None:
                raise ValueError(
                    "The benchmark is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            # Check available int8 converting devices
            if target_data_type == DataType.INT8:
                if target_device_name not in DeviceName.AVAILABLE_INT8_DEVICES:
                    raise ValueError(f"int8 converting supports only {DeviceName.AVAILABLE_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in DeviceName.ONLY_INT8_DEVICES:
                    raise ValueError(f"{DeviceName.ONLY_INT8_DEVICES} only support int8 data types.")

            if target_hardware_type == HardwareType.HELIUM and target_device_name not in DeviceName.ONLY_INT8_DEVICES:
                raise ValueError(f"{DeviceName.ONLY_INT8_DEVICES} only support helium hardware type.")

            devices = TargetDeviceFilter.filter_devices_with_device_name(
                name=target_device_name, devices=model.available_devices
            )
            if target_device_name in DeviceName.JETSON_DEVICES:
                devices = TargetDeviceFilter.filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )
            devices = TargetDeviceFilter.filter_devices_with_hardware_type(
                hardware_type=target_hardware_type, devices=devices
            )

            if not devices:
                raise NotImplementedError(
                    "The benchmark is unavailable. There is no available device with given target_device_name and target_software_version."
                )

            target_device = devices[0]
            target_device_name = target_device.device_name
            target_software_version = target_device.software_version
            target_hardware_type = target_device.hardware_type

            if target_device_name is None:
                raise NotImplementedError(
                    "There is no avaliable function for given paremeter. Please specify the target device."
                )

            model_benchmark: BenchmarkTask = launcher_client.benchmark_model(
                user_uuid=self.user_info.user_id,
                model_uuid=model_uuid,
                target_device=target_device_name,
                data_type=target_data_type,
                software_version=target_software_version,
                hardware_type=target_hardware_type,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )
            model_benchmark = self.get_benchmark_task(benchmark_task=model_benchmark)

            if wait_until_done:
                while model_benchmark.status in [
                    TaskStatus.IN_QUEUE,
                    TaskStatus.IN_PROGRESS,
                ]:
                    model_benchmark = self.get_benchmark_task(benchmark_task=model_benchmark)
                    time.sleep(1)

            metadata.update_benchmark_info(
                target_device=model_benchmark.target_device,
                file_name=model_benchmark.filename,
                data_type=model_benchmark.data_type,
                processor=model_benchmark.processor,
                software_version=model_benchmark.software_version,
                hardware_type=model_benchmark.hardware_type,
                input_model_uuid=model_benchmark.input_model_uuid,
                benchmark_task_uuid=model_benchmark.benchmark_task_uuid,
                devicefarm_benchmark_task_uuid=model_benchmark.devicefarm_benchmark_task_uuid,
                devicefarm_model_uuid=model_benchmark.devicefarm_model_uuid,
            )
            metadata.update_result(
                memory_footprint_gpu=model_benchmark.memory_footprint_gpu,
                memory_footprint_cpu=model_benchmark.memory_footprint_cpu,
                latency=model_benchmark.latency,
                ram_size=model_benchmark.ram_size,
                power_consumption=model_benchmark.power_consumption,
                file_size=model_benchmark.file_size,
            )
            metadata.update_status(status=Status.COMPLETED)
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name="benchmark")

            remaining_credit = auth_client.get_credit(
                self.token_handler.tokens.access_token, self.token_handler.verify_ssl
            )
            logger.info(
                f"{ServiceCredit.MODEL_BENCHMARK} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

        except Exception as e:
            logger.error(f"Benchmark failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name="benchmark")
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=folder_path, file_name="benchmark")

        return metadata.asdict()

    def get_benchmark_task(self, benchmark_task: Union[str, BenchmarkTask]) -> BenchmarkTask:
        """Get information about the specified benchmark task using either the benchmark task object or its UUID.

        Args:
            benchmark_task (Union[BenchmarkTask, str]): Benchmark task object or the UUID of the benchmark task.

        Raises:
            e: If an error occurs while retrieving information about the benchmark task.

        Returns:
            BenchmarkTask: Model benchmark task object.
        """

        self.token_handler.validate_token()

        try:
            task_uuid = None
            if isinstance(benchmark_task, str):
                task_uuid = benchmark_task
            elif type(benchmark_task) is BenchmarkTask:
                task_uuid = benchmark_task.benchmark_task_uuid
            else:
                raise NotImplementedError(
                    "There is no available function for the given parameter. The 'benchmark_task' should be a UUID string or a ModelBenchmark object."
                )

            return launcher_client.get_benchmark(
                benchmark_task_uuid=task_uuid,
                access_token=self.token_handler.tokens.access_token,
                verify_ssl=self.token_handler.verify_ssl,
            )

        except Exception as e:
            logger.error(f"Get benchmark failed. Error: {e}")
