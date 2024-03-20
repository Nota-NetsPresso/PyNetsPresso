import json

import requests

from netspresso.clients.config import Config, Module
from netspresso.clients.launcher.v1.schemas.model import (
    BenchmarkTask,
    ConversionTask,
    InputShape,
    Model,
    ModelBenchmarkRequest,
    ModelConversionRequest,
)
from netspresso.clients.utils.common import get_files, get_headers
from netspresso.enums.device import DeviceName
from netspresso.enums.model import DataType, Framework


class LauncherAPIClient:
    def __init__(self):
        self.config = Config(Module.LAUNCHER)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"

    def upload_model(self, model_file_path: str, target_function: str, access_token, verify_ssl: bool = True) -> Model:
        url = f"{self.url}/{target_function.value.lower()}/upload_model"
        files = get_files(model_file_path)
        response = requests.post(url, files=files, headers=get_headers(access_token), verify=verify_ssl)
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return Model(**response_body)
        else:
            raise Exception(response_body["detail"])

    def convert_model(
        self,
        user_uuid: str,
        model_uuid: str,
        input_shape: InputShape,
        target_framework: Framework,
        target_device: DeviceName,
        data_type: DataType,
        software_version: str,
        dataset_path: str,
        access_token,
        verify_ssl: bool = True,
    ) -> ConversionTask:
        """Convert a model into the type the specific framework.

        Args:
            model_uuid (str): The uuid of the launcher model.
            input_shape (InputShape) : target input shape to convert. (ex: dynamic batch to static batch)
            target_framework (Framework): the target framework name.
            target_device (DeviceName): target device.
            data_type (DataType): data type of the model.
            software_version (str): target device's software version.

        Raises:
            e: If an error occurs while converting the model.

        Returns:
            ConversionTask: model conversion task object.
        """

        url = f"{self.url}/convert"
        request_data = ModelConversionRequest(
            user_uuid=user_uuid,
            target_device_name=target_device,
            input_model_uuid=model_uuid,
            target_framework=target_framework,
            data_type=data_type,
            input_shape=input_shape,
            software_version=software_version,
        )
        if software_version is not None:
            request_data.software_version = software_version

        files = None
        if dataset_path:
            files = get_files(dataset_path)

        response = requests.post(
            url,
            data=request_data.dict(),
            files=files,
            headers=get_headers(access_token),
            verify=verify_ssl,
        )
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return ConversionTask(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_conversion_task(self, conversion_task_uuid: str, access_token, verify_ssl: bool = True) -> ConversionTask:
        """Get the conversion task information with given conversion task uuid.

        Args:
            conversion_task_uuid (str): The uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """

        url = f"{self.url}/convert/{conversion_task_uuid}"
        response = requests.get(url, headers=get_headers(access_token), verify=verify_ssl)
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return ConversionTask(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_converted_model(self, conversion_task_uuid: str, access_token, verify_ssl: bool = True):
        """Download the converted model with given conversion task or conversion task uuid.

        Args:
            conversion_task (ConversionTask | str): Launcher Model Object or the uuid of the conversion task.

        Raises:
            e: If an error occurs while getting the conversion task information.

        Returns:
            ConversionTask: model conversion task object.
        """
        url = f"{self.url}/convert/{conversion_task_uuid}/download"
        response = requests.get(url, headers=get_headers(access_token), verify=verify_ssl)
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return response_body
        else:
            raise Exception(response_body["detail"])

    def benchmark_model(
        self,
        user_uuid: str,
        model_uuid: str,
        target_device: DeviceName,
        data_type: DataType,
        access_token,
        software_version: str = None,
        hardware_type: str = None,
        verify_ssl: bool = True,
    ) -> BenchmarkTask:
        """Benchmark given model on the specified device.

        Args:
            model_uuid (str): The conversion task object, Launcher Model Object or the uuid of the model.
            target_device (TargetDevice): target device.
            data_type (DataType): data type of the model.
            wait_until_done (bool): if true, wait for the conversion result before returning the function. If false, request the conversion and return the function immediately.

        Raises:
            e: If an error occurs while benchmarking of the model.

        Returns:
            BenchmarkTask: model benchmark task object.
        """
        url = f"{self.url}/benchmark"
        request_data = ModelBenchmarkRequest(
            user_uuid=user_uuid,
            input_model_uuid=model_uuid,
            target_device=target_device,
            data_type=data_type,
            software_version=software_version,
            hardware_type=hardware_type,
        )
        response = requests.post(
            url,
            json=request_data.dict(),
            headers=get_headers(access_token),
            verify=verify_ssl,
        )
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return BenchmarkTask(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_benchmark(self, benchmark_task_uuid: str, access_token, verify_ssl: bool = True) -> BenchmarkTask:
        """Get the benchmark task information with given benchmark task uuid.

        Args:
            benchmark_task_uuid (str): the uuid of the benchmark task.

        Raises:
            e: If an error occurs while getting the benchmark task information.

        Returns:
            BenchmarkTask: model benchmark task object.
        """
        url = f"{self.url}/benchmark/{benchmark_task_uuid}"
        response = requests.get(url, headers=get_headers(access_token), verify=verify_ssl)
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return BenchmarkTask(**response_body)
        else:
            raise Exception(response_body["detail"])


launcher_client = LauncherAPIClient()
