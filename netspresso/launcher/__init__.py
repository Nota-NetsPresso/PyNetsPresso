import sys
import time
from pathlib import Path
from typing import Dict, Union
from urllib import request

from loguru import logger

from netspresso.clients.auth import BaseClient, validate_token
from netspresso.clients.launcher import LauncherAPIClient
from netspresso.clients.launcher.enums import (
    JETSON_DEVICES,
    AVAILABLE_INT8_DEVICES,
    ONLY_INT8_DEVICES,
    DataType,
    DeviceName,
    HardwareType,
    LauncherFunction,
    ModelFramework,
    SoftwareVersion,
    TaskStatus,
)
from netspresso.clients.launcher.schemas.model import (
    BenchmarkTask,
    ConversionTask,
    InputShape,
    Model,
    TargetDevice,
)
from netspresso.enums import ServiceCredit, TaskType, Status
from netspresso.launcher.utils.devices import (
    filter_devices_with_device_name,
    filter_devices_with_device_software_version,
    filter_devices_with_hardware_type,
)

from ..utils import FileHandler, check_credit_balance
from ..utils.metadata import MetadataHandler


class Launcher(BaseClient):
    target_function: LauncherFunction = LauncherFunction.GENERAL

    def __init__(self, email=None, password=None, user_session=None):
        """Initialize the Model Compressor.

        Args:
            email (str): The email address for a user account.
            password (str): The password for a user account.
            user_session (SessionClient): The SessionClient object.

        Available constructors:
            Launcher(email='USER_EMAIL',password='PASSWORD')
            Launcher(user_session=SessionClient(email='USER_EMAIL',password='PASSWORD')
        """
        super().__init__(email=email, password=password, user_session=user_session)
        self.client = LauncherAPIClient(user_sessoin=self.user_session)

    @validate_token
    def _upload_model(self, model_file_path: Union[Path, str]) -> Model:
        """Upload a model for launcher.

        Args:
            model_file_path (str): The file path of the model.

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            Model: Uploaded launcher model object.
        """
        return self.client.upload_model(
            model_file_path=model_file_path,
            target_function=self.__class__.target_function,
        )


class Converter(Launcher):
    target_function: LauncherFunction = LauncherFunction.CONVERT

    @validate_token
    def convert_model(
        self,
        model_path: Union[Path, str],
        output_path: Union[Path, str],
        target_framework: Union[str, ModelFramework],
        data_type: DataType = DataType.FP16,
        target_device: TargetDevice = None,
        wait_until_done: bool = True,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
        input_shape: InputShape = None,
        dataset_path: str = None,
    ) -> Dict:
        """Convert a model into the type the specific framework.

        Args:
            model_path (str): The file path where the model is located.
            output_path (str): The local path to save the converted model.
            target_framework (ModelFramework | str): the target framework name.
            data_type (DataType): data type of the model.
            target_device (TargetDevice): target device. If it's not set, target_device_name and target_software_version have to be set.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.
            input_shape (InputShape) : target input shape to convert. (ex: dynamic batch to static batch)

        Raises:
            e: If an error occurs while converting the model.

        Returns:
            Dict: model conversion task dict.
        """
        try:
            default_model_path, extension = FileHandler.get_path_and_extension(
                folder_path=output_path, framework=target_framework
            )
            FileHandler.create_folder(folder_path=output_path)
            metadata = MetadataHandler.init_metadata(folder_path=output_path, task_type=TaskType.CONVERT)

            current_credit = self.user_session.get_credit()
            check_credit_balance(
                user_credit=current_credit, service_credit=ServiceCredit.MODEL_CONVERT
            )
            model = self._upload_model(model_path)

            model_uuid = model
            if type(model) is Model:
                model_uuid = model.model_uuid
                if input_shape is None and model.input_shape is not None:
                    input_shape = model.input_shape
                if target_framework is None and model.framework is not None:
                    target_framework = model.framework

            if target_device is None:
                if target_device_name is None:
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set target_device or target_device_name."
                    )

                elif (
                    target_device_name in JETSON_DEVICES and target_software_version is None
                ):
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                    )

                if type(model) is not Model:
                    raise NotImplementedError(
                        "The conversion is unavailable. Please set target_device while using model's uuid string for the conversion."
                    )

                # Check available int8 converting devices
                if data_type == DataType.INT8:
                    if target_device_name not in AVAILABLE_INT8_DEVICES:
                        raise Exception(
                            f"int8 converting supports only {AVAILABLE_INT8_DEVICES}."
                        )
                else:  # FP16, FP32
                    if target_device_name in ONLY_INT8_DEVICES:
                        raise Exception(
                            f"{ONLY_INT8_DEVICES} only support int8 data types."
                        )

                devices = filter_devices_with_device_name(
                    name=target_device_name, devices=model.available_devices
                )

                if target_device_name in JETSON_DEVICES:
                    devices = filter_devices_with_device_software_version(
                        software_version=target_software_version, devices=devices
                    )

                if len(devices) < 1:
                    raise NotImplementedError(
                        "The conversion is unavailable. There is no available device with given target_device_name and target_software_version."
                    )

                target_device = devices[0]

            logger.info(
                f"Converting Model for {target_device.device_name} ({target_framework})"
            )

            conversion_task = self.client.convert_model(
                model_uuid=model_uuid,
                input_shape=input_shape,
                target_framework=target_framework,
                target_device=target_device.device_name,
                data_type=data_type,
                software_version=target_device.software_version,
                dataset_path=dataset_path,
            )

            conversion_task = self.get_conversion_task(conversion_task)

            if wait_until_done:
                while conversion_task.status in [
                    TaskStatus.IN_QUEUE,
                    TaskStatus.IN_PROGRESS,
                ]:
                    conversion_task = self.get_conversion_task(conversion_task)
                    time.sleep(1)

            self.download_converted_model(
                conversion_task, default_model_path.with_suffix(extension)
            )

            converter_uploaded_model = self.client.upload_model(
                model_file_path=default_model_path.with_suffix(extension),
                target_function=LauncherFunction.BENCHMARK,
            )

            metadata.update_model_info(
                data_type=model.data_type,
                framework=model.framework,
                input_shape=model.input_shape,
            )
            metadata.update_convert_info(
                target_framework=conversion_task.target_framework,
                target_device_name=conversion_task.target_device_name,
                data_type=conversion_task.data_type,
                software_version=conversion_task.software_version,
                model_file_name=conversion_task.model_file_name,
                convert_task_uuid=conversion_task.convert_task_uuid,
                input_model_uuid=conversion_task.input_model_uuid,
                output_model_uuid=conversion_task.output_model_uuid,
            )
            metadata.update_status(status=Status.COMPLETED)
            metadata.update_available_devices(converter_uploaded_model.available_devices)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)

            remaining_credit = self.user_session.get_credit()
            logger.info(
                f"{ServiceCredit.MODEL_CONVERT} credits have been consumed. Remaining Credit: {remaining_credit}"
            )

            return metadata.asdict()
        
        except Exception as e:
            logger.error(f"Convert failed. Error: {e}")
            metadata.update_status(status=Status.ERROR)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)
            raise e

        except KeyboardInterrupt:
            metadata.update_status(status=Status.STOPPED)
            MetadataHandler.save_json(data=metadata.asdict(), folder_path=output_path)

    @validate_token
    def get_conversion_task(
        self, conversion_task: Union[str, ConversionTask]
    ) -> ConversionTask:
        """Get the conversion task information with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        try:
            conversion_task_uuid = None
            if type(conversion_task) is str:
                conversion_task_uuid = conversion_task
            elif type(conversion_task) is ConversionTask:
                conversion_task_uuid = conversion_task.convert_task_uuid
            else:
                raise NotImplementedError(
                    "There is no available function for the given parameter. The 'conversion_task' should be a UUID string or a ModelConversion object."
                )
            return self.client.get_conversion_task(
                conversion_task_uuid=conversion_task_uuid
            )

        except Exception as e:
            logger.error(f"Get conversion task failed. Error: {e}")
            raise e

    @validate_token
    def download_converted_model(
        self, conversion_task: Union[str, ConversionTask], local_path: str
    ):
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        try:
            conversion_task_uuid = None
            if type(conversion_task) is str:
                conversion_task_uuid = conversion_task
            elif type(conversion_task) is ConversionTask:
                conversion_task_uuid = conversion_task.convert_task_uuid

            conversion_result: ConversionTask = self.get_conversion_task(
                conversion_task_uuid
            )
            if conversion_result.status is TaskStatus.ERROR:
                raise FileNotFoundError(
                    "The conversion is Failed. There is no file available for download."
                )
            if conversion_result.status is not TaskStatus.FINISHED:
                raise FileNotFoundError(
                    "The conversion is in progress. There is no file available for download at the moment."
                )

            download_url = self.client.get_converted_model(
                conversion_task_uuid=conversion_result.convert_task_uuid
            )
            request.urlretrieve(download_url, local_path)
            logger.info(f"Model downloaded at {Path(local_path)}")

        except Exception as e:
            logger.error(f"Download converted model failed. Error: {e}")
            raise e


class Benchmarker(Launcher):
    target_function: LauncherFunction = LauncherFunction.BENCHMARK

    @validate_token
    def benchmark_model(
        self,
        model_path: Union[Path, str],
        target_framework: Union[str, ModelFramework],
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
            model = self._upload_model(default_model_path.with_suffix(extension))
            model_uuid = model.model_uuid

            if target_device_name is None:
                raise ValueError(
                    "The benchmark is unavailable. Please set target_device_name."
                )

            if target_device_name in JETSON_DEVICES and target_software_version is None:
                raise ValueError(
                    "The benchmark is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            # Check available int8 converting devices
            if data_type == DataType.INT8:
                if target_device_name not in AVAILABLE_INT8_DEVICES:
                    raise ValueError(f"int8 converting supports only {AVAILABLE_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in ONLY_INT8_DEVICES:
                    raise ValueError(f"{ONLY_INT8_DEVICES} only support int8 data types.")

            if (
                hardware_type == HardwareType.HELIUM
                and target_device_name not in ONLY_INT8_DEVICES
            ):
                raise ValueError(f"{ONLY_INT8_DEVICES} only support helium hardware type.")

            devices = filter_devices_with_device_name(
                name=target_device_name, devices=model.available_devices
            )
            if target_device_name in JETSON_DEVICES:
                devices = filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )
            devices = filter_devices_with_hardware_type(
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
