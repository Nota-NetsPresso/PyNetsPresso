import time
from pathlib import Path
from typing import Dict, Union

from loguru import logger

from netspresso.clients.auth import BaseClient, validate_token
from netspresso.clients.launcher import LauncherAPIClient
from netspresso.enums import (
    DataType,
    DeviceName,
    HardwareType,
    Module,
    Framework,
    SoftwareVersion,
    TaskStatus,
)
from netspresso.clients.launcher.schemas.model import (
    BenchmarkTask,
    Model,
)
from netspresso.enums import ServiceCredit, TaskType, Status
from netspresso.clients.launcher.schemas import TargetDeviceFilter

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler


class Benchmarker(BaseClient):
    def __init__(self, email=None, password=None, user_session=None):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors:
            Benchmarker(email='USER_EMAIL',password='PASSWORD')
            Benchmarker(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """
        super().__init__(email=email, password=password, user_session=user_session)
        self.client = LauncherAPIClient(user_sessoin=self.user_session)

    @validate_token
    def benchmark_model(
        self,
        model_path: Union[Path, str],
        target_framework: Union[str, Framework],
        data_type: DataType = DataType.FP16,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
        hardware_type: HardwareType = None,
        wait_until_done: bool = True,
    ) -> Dict:
        """Benchmark given model on the specified device.

        Args:
            model_path (str): The file path where the model is located.
            data_type (DataType): data type of the model.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.
            hardware_type (HardwareType): hardware_type. Acceleration options for the processor to the model inference.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.

        Raises:
            e: If an error occurs while benchmarking of the model.

        Returns:
            Dict: model benchmark task dict.
        """
        try:
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=model_path, framework=target_framework
            )

            metadata = MetadataHandler.get_default_metadata(TaskType.BENCHMARK)
            if FileHandler.check_exists(Path(model_path) / f"benchmark.json"):
                metadatas = MetadataHandler.load_json(Path(model_path) / f"benchmark.json")
                metadatas.append(metadata.asdict())
            else:
                metadatas = [metadata.asdict()]
            MetadataHandler.save_json(metadatas, model_path, file_name="benchmark")

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit, service_credit=ServiceCredit.MODEL_BENCHMARK
            )
            model = self.client.upload_model(model_file_path=default_model_path.with_suffix(extension), target_function=Module.BENCHMARK)
            model_uuid = model.model_uuid

            if target_device_name is None:
                raise ValueError(
                    "The benchmark is unavailable. Please set target_device_name."
                )

            if target_device_name in DeviceName.JETSON_DEVICES and target_software_version is None:
                raise ValueError(
                    "The benchmark is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            # Check available int8 converting devices
            if data_type == DataType.INT8:
                if target_device_name not in DeviceName.AVAILABLE_INT8_DEVICES:
                    raise ValueError(f"int8 converting supports only {DeviceName.AVAILABLE_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in DeviceName.ONLY_INT8_DEVICES:
                    raise ValueError(f"{DeviceName.ONLY_INT8_DEVICES} only support int8 data types.")

            if (
                hardware_type == HardwareType.HELIUM
                and target_device_name not in DeviceName.ONLY_INT8_DEVICES
            ):
                raise ValueError(f"{DeviceName.ONLY_INT8_DEVICES} only support helium hardware type.")

            devices = TargetDeviceFilter.filter_devices_with_device_name(
                name=target_device_name, devices=model.available_devices
            )
            if target_device_name in DeviceName.JETSON_DEVICES:
                devices = TargetDeviceFilter.filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )
            devices = TargetDeviceFilter.filter_devices_with_hardware_type(
                hardware_type=hardware_type, devices=devices
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

            model_benchmark: BenchmarkTask = self.client.benchmark_model(
                model_uuid=model_uuid,
                target_device=target_device_name,
                data_type=data_type,
                software_version=target_software_version,
                hardware_type=target_hardware_type,
            )
            model_benchmark = self.get_benchmark_task(benchmark_task=model_benchmark)

            if wait_until_done:
                while model_benchmark.status in [
                    TaskStatus.IN_QUEUE,
                    TaskStatus.IN_PROGRESS,
                ]:
                    model_benchmark = self.get_benchmark_task(
                        benchmark_task=model_benchmark
                    )
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
            MetadataHandler.save_json(data=metadatas, folder_path=model_path, file_name="benchmark")

            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.MODEL_BENCHMARK} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

        except Exception as e:
            logger.error(f"Benchmark failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=model_path, file_name="benchmark")
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            metadatas[-1] = metadata.asdict()
            MetadataHandler.save_json(data=metadatas, folder_path=model_path, file_name="benchmark")

        return metadata.asdict()

    @validate_token
    def get_benchmark_task(
        self, benchmark_task: Union[str, BenchmarkTask]
    ) -> BenchmarkTask:
        """Get the benchmark task information with given benchmark task or benchmark task uuid.

        Args:
            benchmark_task (BenchmarkTask | str): Launcher Benchmark Object or the uuid of the benchmark task.

        Raises:
            e: If an error occurs while getting the benchmark task information.

        Returns:
            BenchmarkTask: model benchmark task object.
        """
        try:
            task_uuid = None
            if type(benchmark_task) is str:
                task_uuid = benchmark_task
            elif type(benchmark_task) is BenchmarkTask:
                task_uuid = benchmark_task.benchmark_task_uuid
            else:
                raise NotImplementedError(
                    "There is no available function for the given parameter. The 'benchmark_task' should be a UUID string or a ModelBenchmark object."
                )

            return self.client.get_benchmark(benchmark_task_uuid=task_uuid)

        except Exception as e:
            logger.error(f"Get benchmark failed. Error: {e}")
