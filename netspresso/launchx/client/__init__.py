import json
import requests

from netspresso.client.utils.common import get_files, get_headers
from netspresso.client import Config, EndPoint, SessionClient
from netspresso.launchx.schemas.model import Model, ModelBenchmarkRequest, BenchmarkTask, ModelConversionRequest, ConversionTask, InputShape
from netspresso.launchx.schemas import LaunchXFunction, ModelFramework, DeviceName, DataType


class LaunchXAPIClient:
    def __init__(self, user_sessoin: SessionClient):
        self.config = Config(EndPoint.LAUNCHX)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.user_session = user_sessoin
        self.url = f"{self.host}:{self.port}{self.prefix}"

    def upload_model(self, model_file_path: str, target_function: LaunchXFunction):
        url = f"{self.url}/{target_function.value.lower()}/upload_model"
        files = get_files(model_file_path)
        # files = {"file": open(model_file_path, "rb")}
        response = requests.post(url, files=files, headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return Model(**response_body)
        else:
            raise Exception(response_body["detail"])
    def convert_model(self, model_uuid: str,
                      input_shape: InputShape,
                      target_framework: ModelFramework,
                      target_device: DeviceName,
                      software_version: str):
        url = f"{self.url}/convert"
        request_data = ModelConversionRequest(user_uuid=self.user_session.user_id,
                                              target_device_name=target_device,
                                              input_model_uuid=model_uuid,
                                              target_framework=target_framework,
                                              input_shape=input_shape,
                                              software_version=software_version)
        response = requests.post(url, json=request_data.dict(), headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return ConversionTask(**response_body)
        else:
            raise Exception(response_body["detail"])
        
    def get_conversion_task(self, conversion_task_uuid: str):
        url = f"{self.url}/convert/{conversion_task_uuid}"
        response = requests.get(url, headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return ConversionTask(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_converted_model(self, conversion_task_uuid: str):
        url = f"{self.url}/convert/{conversion_task_uuid}/download"
        response = requests.get(url, headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return response_body
        else:
            raise Exception(response_body["detail"])
        
    def benchmark_model(self, model_uuid:str, target_device: DeviceName, data_type:DataType, software_version: str = None):
        url = f"{self.url}/benchmark"
        request_data = ModelBenchmarkRequest(user_uuid=self.user_session.user_id,
                                             input_model_uuid=model_uuid,
                                             target_device=target_device,
                                             data_type=data_type)
        response = requests.post(url, json=request_data.dict(), headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return BenchmarkTask(**response_body)
        else:
            raise Exception(response_body["detail"])
        
    def get_benchmark(self, benchmark_task_uuid: str):
        url = f"{self.url}/benchmark/{benchmark_task_uuid}"
        response = requests.get(url, headers=get_headers(self.user_session.access_token))
        response_body = json.loads(response.text)
        if response.status_code < 300:
            return BenchmarkTask(**response_body)
        else:
            raise Exception(response_body["detail"])
        
