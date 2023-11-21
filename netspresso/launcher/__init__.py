from typing import Dict, List, Union
from loguru import logger
from urllib import request
import time

from netspresso.client import BaseClient, validate_token
from netspresso.launcher.schemas import (
    LauncherFunction,
    ModelFramework,
    TaskStatus,
    DataType,
    SoftwareVersion,
    DeviceName,
    JETSON_DEVICES,
    HardwareType,
    ONLY_INT8_DEVICES,
)
from netspresso.launcher.schemas.model import Model, ConversionTask, BenchmarkTask, InputShape, TargetDevice
from netspresso.launcher.utils.devices import (
    filter_devices_with_device_name,
    filter_devices_with_device_software_version,
    filter_devices_with_hardware_type,
)
from netspresso.launcher.client import LauncherAPIClient


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
    def upload_model(self, model_file_path: str) -> Model:
        """Upload a model for launcher.

        Args:
            model_file_path (str): The file path of the model.

        Raises:
            e: If an error occurs while uploading the model.

        Returns:
            Model: Uploaded launcher model object.
        """
        return self.client.upload_model(model_file_path=model_file_path, target_function=self.__class__.target_function)


class ModelConverter(Launcher):
    target_function: LauncherFunction = LauncherFunction.CONVERT

    @validate_token
    def convert_model(
        self,
        model: Union[str, Model],
        input_shape: InputShape,
        target_framework: Union[str, ModelFramework],
        data_type: DataType = DataType.FP16,
        target_device: TargetDevice = None,
        wait_until_done: bool = True,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
    ) -> ConversionTask:
        """Convert a model into the type the specific framework.

        Args:
            model (str): The uuid of the model or Launcher Model Object.
            input_shape (InputShape) : target input shape to convert. (ex: dynamic batch to static batch)
            target_framework (ModelFramework | str): the target framework name.
            data_type (DataType): data type of the model.
            target_device (TargetDevice): target device. If it's not set, target_device_name and target_software_version have to be set.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.

        Raises:
            e: If an error occurs while converting the model.

        Returns:
            ConversionTask: model conversion task object.
        """
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

            elif target_device_name in JETSON_DEVICES and target_software_version is None:
                raise NotImplementedError(
                    "The conversion is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            if type(model) is not Model:
                raise NotImplementedError(
                    "The conversion is unavailable. Please set target_device while using model's uuid string for the conversion."
                )

            # Check available int8 converting devices
            if data_type == DataType.INT8:
                if target_device_name not in ONLY_INT8_DEVICES:
                    raise Exception(f"int8 converting supports only {ONLY_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in ONLY_INT8_DEVICES:
                    raise Exception(f"{ONLY_INT8_DEVICES} only support int8 data types.")

            devices = filter_devices_with_device_name(name=target_device_name, devices=model.available_devices)

            if target_device_name in JETSON_DEVICES:
                devices = filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )

            if len(devices) < 1:
                raise NotImplementedError(
                    "The conversion is unavailable. There is no available device with given target_device_name and target_software_version."
                )

            target_device = devices[0]

        logger.info(f"Converting Model for {target_device.device_name} ({target_framework})")

        conversion_task = self.client.convert_model(
            model_uuid=model_uuid,
            input_shape=input_shape,
            target_framework=target_framework,
            target_device=target_device.device_name,
            data_type=data_type,
            software_version=target_device.software_version,
        )

        conversion_task = self.get_conversion_task(conversion_task)

        if wait_until_done:
            while conversion_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
                conversion_task = self.get_conversion_task(conversion_task)
                time.sleep(1)

        return conversion_task

    @validate_token
    def get_conversion_task(self, conversion_task: Union[str, ConversionTask]) -> ConversionTask:
        """Get the conversion task information with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        conversion_task_uuid = None
        if type(conversion_task) is str:
            conversion_task_uuid = conversion_task
        elif type(conversion_task) is ConversionTask:
            conversion_task_uuid = conversion_task.convert_task_uuid
        else:
            raise NotImplementedError(
                "There is no available function for the given parameter. The 'conversion_task' should be a UUID string or a ModelConversion object."
            )
        return self.client.get_conversion_task(conversion_task_uuid=conversion_task_uuid)

    @validate_token
    def download_converted_model(self, conversion_task: Union[str, ConversionTask], dst: str):
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        conversion_task_uuid = None
        if type(conversion_task) is str:
            conversion_task_uuid = conversion_task
        elif type(conversion_task) is ConversionTask:
            conversion_task_uuid = conversion_task.convert_task_uuid

        conversion_result: ConversionTask = self.get_conversion_task(conversion_task_uuid)
        if conversion_result.status is TaskStatus.ERROR:
            raise FileNotFoundError("The conversion is Failed. There is no file available for download.")
        if conversion_result.status is not TaskStatus.FINISHED:
            raise FileNotFoundError(
                "The conversion is in progress. There is no file available for download at the moment."
            )

        download_url = self.client.get_converted_model(conversion_task_uuid=conversion_result.convert_task_uuid)
        request.urlretrieve(download_url, dst)
        logger.info(f"The file has been successfully downloaded at : {dst}")


class ModelBenchmarker(Launcher):
    target_function: LauncherFunction = LauncherFunction.BENCHMARK

    @validate_token
    def benchmark_model(
        self,
        model: Union[ConversionTask, Model, str],
        target_device: TargetDevice = None,
        data_type: DataType = DataType.FP16,
        wait_until_done: bool = True,
        target_device_name: DeviceName = None,
        target_software_version: SoftwareVersion = None,
        hardware_type: HardwareType = None,
    ) -> BenchmarkTask:
        """Benchmark given model on the specified device.

        Args:
            model (ConversionTask | Model | str): The conversion task object, Launcher Model Object or the uuid of the model.
            target_device (TargetDevice): target device. If it's not set, target_device_name and target_software_version have to be set.
            data_type (DataType): data type of the model.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.
            target_device_name (DeviceName): target device name. Necessary field if target_device is not given.
            target_software_version (SoftwareVersion): target_software_version. Necessary field if target_device_name is one of jetson devices.
            hardware_type (HardwareType): hardware_type. Acceleration options for the processor to the model inference.

        Raises:
            e: If an error occurs while benchmarking of the model.

        Returns:
            BenchmarkTask: model benchmark task object.
        """
        model_uuid = None
        benchmark_data_type = data_type

        if target_device is None and type(model) is not ConversionTask:
            if target_device_name is None:
                raise NotImplementedError(
                    "The benchmark is unavailable. Please set target_device or target_device_name."
                )

            elif target_device_name in JETSON_DEVICES and target_software_version is None:
                raise NotImplementedError(
                    "The benchmark is unavailable. Please set JetPack version with target_software_version for Jetson Devices."
                )

            if type(model) is not Model:
                raise NotImplementedError(
                    "The benchmark is unavailable. Please set target_device while using model's uuid string for the conversion."
                )

            # Check available int8 converting devices
            if data_type == DataType.INT8:
                if target_device_name not in ONLY_INT8_DEVICES:
                    raise Exception(f"int8 converting supports only {ONLY_INT8_DEVICES}.")
            else:  # FP16, FP32
                if target_device_name in ONLY_INT8_DEVICES:
                    raise Exception(f"{ONLY_INT8_DEVICES} only support int8 data types.")

            if hardware_type == HardwareType.HELIUM and target_device_name not in ONLY_INT8_DEVICES:
                raise Exception(f"{ONLY_INT8_DEVICES} only support helium hardware type.")

            devices = filter_devices_with_device_name(name=target_device_name, devices=model.available_devices)

            if target_device_name in JETSON_DEVICES:
                devices = filter_devices_with_device_software_version(
                    software_version=target_software_version, devices=devices
                )

            devices = filter_devices_with_hardware_type(hardware_type=hardware_type, devices=devices)

            if len(devices) < 1:
                raise NotImplementedError(
                    "The benchmark is unavailable. There is no available device with given target_device_name and target_software_version."
                )

            target_device = devices[0]

        target_device_name = target_device.device_name if target_device is not None else None
        target_software_version = target_device.software_version if target_device is not None else None
        target_hardware_type = target_device.hardware_type if target_device is not None else None

        if type(model) is str:
            model_uuid = model
        elif type(model) is Model:
            model_uuid = model.model_uuid
        elif type(model) is ConversionTask:
            model_uuid = model.output_model_uuid
            if target_device_name is None:
                target_device_name = model.target_device_name
            if target_software_version is None:
                target_software_version = model.software_version
            if target_hardware_type is None:
                target_hardware_type = model.hardware_type

        if target_device_name is None:
            raise NotImplementedError(
                "There is no avaliable function for given paremeter. Please specify the target device."
            )

        model_benchmark: BenchmarkTask = self.client.benchmark_model(
            model_uuid=model_uuid,
            target_device=target_device_name,
            data_type=benchmark_data_type,
            software_version=target_software_version,
            hardware_type=target_hardware_type,
        )
        model_benchmark = self.get_benchmark_task(benchmark_task=model_benchmark)

        if wait_until_done:
            while model_benchmark.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
                model_benchmark = self.get_benchmark_task(benchmark_task=model_benchmark)
                time.sleep(1)

        return model_benchmark

    @validate_token
    def get_benchmark_task(self, benchmark_task: Union[str, BenchmarkTask]) -> BenchmarkTask:
        """Get the benchmark task information with given benchmark task or benchmark task uuid.

        Args:
            benchmark_task (BenchmarkTask | str): Launcher Benchmark Object or the uuid of the benchmark task.

        Raises:
            e: If an error occurs while getting the benchmark task information.

        Returns:
            BenchmarkTask: model benchmark task object.
        """
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
