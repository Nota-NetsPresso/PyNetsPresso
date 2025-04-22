import os

from loguru import logger

from netspresso.clients.launcher.v2.implements import BenchmarkTaskAPI, ModelAPI
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    InputLayer,
    RequestBenchmark,
    RequestModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseBenchmarkFrameworkOptionItems,
    ResponseBenchmarkOptionItems,
    ResponseBenchmarkStatusItem,
    ResponseBenchmarkTaskItem,
    ResponseModelItem,
    ResponseModelOptions,
    ResponseModelUploadUrl,
    UploadFile,
)
from netspresso.clients.utils.common import read_file_bytes
from netspresso.enums import DataType, DeviceName, Framework, HardwareType, LauncherTask, SoftwareVersion


class Benchmarker:
    def __init__(self, url):
        self.task_type = LauncherTask.BENCHMARK.value
        self.benchmark_task = BenchmarkTaskAPI(url=url)
        self.benchmark_model = ModelAPI(url=url, task_type=self.task_type)

    def presigned_model_upload_url(self, access_token: str, input_model_path: str) -> ResponseModelUploadUrl:
        object_name = os.path.basename(input_model_path)

        get_upload_url_request_body = RequestModelUploadUrl(object_name=object_name, task=self.task_type)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Benchmark model upload: path - {input_model_path}")

        upload_url_response_body = self.benchmark_model.get_upload_url(
            request_params=get_upload_url_request_body, headers=token_header
        )
        logger.info(f"Request Benchmark upload_url result: {upload_url_response_body}")
        return upload_url_response_body

    def upload_model_file(self, access_token: str, input_model_path: str, presigned_upload_url: str) -> str:
        object_name = os.path.basename(input_model_path)
        file_content = read_file_bytes(file_path=input_model_path)

        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Benchmark model validation:"
            f" path | {input_model_path} "
            f" presigned_upload_url | {presigned_upload_url}"
        )

        get_model_upload_request_body = RequestUploadModel(url=presigned_upload_url)
        model_file_object = UploadFile(file_name=object_name, file_content=file_content)

        upload_result = self.benchmark_model.upload(
            request_body=get_model_upload_request_body,
            file=model_file_object,
            headers=token_header,
        )
        logger.info(f"Request Benchmark upload_model_file result: {upload_result}")
        return upload_result

    def validate_model_file(self, access_token: str, input_model_path: str, ai_model_id: str) -> ResponseModelItem:
        object_name = os.path.basename(input_model_path)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Benchmark model validation:" f" path - {input_model_path}" f" ai_model_id - {ai_model_id}"
        )

        get_validate_model_request_body = RequestValidateModel(
            ai_model_id=ai_model_id,
            display_name=object_name,
        )
        validated_model = self.benchmark_model.validate(
            request_body=get_validate_model_request_body, headers=token_header
        )
        logger.info(f"Request Benchmark validate_model result: {validated_model}")
        return validated_model

    def read_model_task_options(self, access_token, ai_model_id) -> ResponseModelOptions:
        token_header = AuthorizationHeader(access_token=access_token)
        model_task_options = self.benchmark_model.options(headers=token_header, ai_model_id=ai_model_id)
        logger.info(f"Request model task_options: {model_task_options}")
        return model_task_options

    def start_task(
        self,
        access_token: str,
        input_model_id: str,
        target_device_name: DeviceName,
        data_type: DataType = None,
        hardware_type: HardwareType = None,
        input_layer: InputLayer = None,
        software_version: SoftwareVersion = None,
    ) -> ResponseBenchmarkTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        request_body = RequestBenchmark(
            input_model_id=input_model_id,
            target_device_name=target_device_name,
            data_type=data_type,
            hardware_type=hardware_type,
            input_layer=input_layer,
            software_version=software_version,
        )
        logger.info(f"Request Benchmark body: {request_body}")

        benchmark_task_response = self.benchmark_task.start(request_body=request_body, headers=token_header)
        logger.info(f"Request Benchmark result: {benchmark_task_response}")
        return benchmark_task_response

    def cancel_task(self, access_token: str, task_id: str) -> ResponseBenchmarkTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Benchmark Cancel task_id: {task_id}")

        benchmark_task_response = self.benchmark_task.cancel(headers=token_header, task_id=task_id)
        logger.info(f"Request Benchmark Cancel result: {benchmark_task_response}")
        return benchmark_task_response

    def read_task(self, access_token: str, task_id: str) -> ResponseBenchmarkTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Benchmark Read task_id: {task_id}")

        benchmark_task_response = self.benchmark_task.read(headers=token_header, task_id=task_id)
        logger.info(f"Request Benchmark Task Info: {benchmark_task_response}")
        return benchmark_task_response

    def delete_task(self, access_token: str, task_id: str) -> ResponseBenchmarkTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Benchmark delete task_id: {task_id}")

        benchmark_task_response = self.benchmark_task.delete(headers=token_header, task_id=task_id)
        logger.info(f"Request Benchmark Delete Info: {benchmark_task_response}")
        return benchmark_task_response

    def read_status(self, access_token: str, task_id: str) -> ResponseBenchmarkStatusItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Benchmark read_status task_id: {task_id}")

        benchmark_task_response = self.benchmark_task.status(headers=token_header, task_id=task_id)
        logger.info(f"Request Benchmark Task Status: {benchmark_task_response}")
        return benchmark_task_response

    def read_options(self, access_token: str) -> ResponseBenchmarkOptionItems:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info("Request Benchmark options")

        benchmark_task_option_response = self.benchmark_task.options(headers=token_header)
        logger.info(f"Request Benchmark Task Options: {benchmark_task_option_response}")
        return benchmark_task_option_response

    def read_framework_options(self, access_token: str, framework: Framework) -> ResponseBenchmarkFrameworkOptionItems:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info("Request Benchmark options")

        convert_task_option_response = self.benchmark_task.options_by_model_framework(
            headers=token_header, model_framework=framework
        )
        logger.info(f"Request Benchmark Task Options by model_framework: {convert_task_option_response}")
        return convert_task_option_response
