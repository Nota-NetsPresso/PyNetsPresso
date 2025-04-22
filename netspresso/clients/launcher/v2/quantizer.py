import os
from typing import List, Optional

from loguru import logger

from netspresso.clients.launcher.v2.implements import ModelAPI, QuantizeTaskAPI
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    InputLayer,
    RequestModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseModelItem,
    ResponseModelOptions,
    ResponseModelUploadUrl,
    ResponseQuantizeDownloadModelUrlItem,
    ResponseQuantizeOptionItems,
    ResponseQuantizeStatusItem,
    ResponseQuantizeTaskItem,
    UploadDataset,
    UploadFile,
)
from netspresso.clients.launcher.v2.schemas.task.quantize.request_body import QuantizationOptions, RequestQuantizeTask
from netspresso.clients.utils.common import read_file_bytes
from netspresso.enums import LauncherTask, QuantizationMode


class Quantizer:
    def __init__(self, url):
        self.task_type = LauncherTask.QUANTIZE.value
        self.quantize_task = QuantizeTaskAPI(url=url)
        self.quantize_model = ModelAPI(url=url, task_type=self.task_type)

    def presigned_model_upload_url(self, access_token: str, input_model_path: str) -> ResponseModelUploadUrl:
        object_name = os.path.basename(input_model_path)

        get_upload_url_request_body = RequestModelUploadUrl(object_name=object_name, task=self.task_type)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize model upload: path - {input_model_path}")

        upload_url_response_body = self.quantize_model.get_upload_url(
            request_params=get_upload_url_request_body, headers=token_header
        )
        logger.info(f"Request Quantize upload_url result: {upload_url_response_body}")
        return upload_url_response_body

    def upload_model_file(self, access_token: str, input_model_path: str, presigned_upload_url: str) -> str:
        object_name = os.path.basename(input_model_path)
        file_content = read_file_bytes(file_path=input_model_path)

        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Quantize model validation:"
            f" path | {input_model_path} "
            f" presigned_upload_url | {presigned_upload_url}"
        )

        get_model_upload_request_body = RequestUploadModel(url=presigned_upload_url)
        model_file_object = UploadFile(file_name=object_name, file_content=file_content)

        upload_result = self.quantize_model.upload(
            request_body=get_model_upload_request_body,
            file=model_file_object,
            headers=token_header,
        )
        logger.info(f"Request Quantize upload_model_file result: {upload_result}")
        return upload_result

    def validate_model_file(self, access_token: str, input_model_path: str, ai_model_id: str) -> ResponseModelItem:
        object_name = os.path.basename(input_model_path)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize model validation:" f" path - {input_model_path}" f" ai_model_id - {ai_model_id}")

        get_validate_model_request_body = RequestValidateModel(
            ai_model_id=ai_model_id,
            display_name=object_name,
        )

        validated_model = self.quantize_model.validate(
            request_body=get_validate_model_request_body, headers=token_header
        )
        logger.info(f"Request Quantize validate_model result: {validated_model}")
        return validated_model

    def download_model_file(self, access_token, quantize_task_uuid) -> ResponseQuantizeDownloadModelUrlItem:
        token_header = AuthorizationHeader(access_token=access_token)
        download_url = self.quantize_task.get_download_url(headers=token_header, quantize_task_uuid=quantize_task_uuid)
        logger.info(f"Request quantizeed model download_url: {download_url}")
        return download_url

    def read_model_task_options(self, access_token, ai_model_id) -> ResponseModelOptions:
        token_header = AuthorizationHeader(access_token=access_token)
        model_task_options = self.quantize_model.options(headers=token_header, ai_model_id=ai_model_id)
        logger.info(f"Request model task_options: {model_task_options}")
        return model_task_options

    def create_upload_dataset(self, dataset_path: str) -> Optional[UploadDataset]:
        if dataset_path:
            dataset_filename = os.path.basename(dataset_path)
            return UploadDataset(
                file_name=dataset_filename,
                file_content=read_file_bytes(dataset_path),
            )
        return None

    def start_task(
        self,
        access_token: str,
        input_model_id: str,
        quantization_mode: QuantizationMode,
        quantization_options: QuantizationOptions,
        input_layers: List[InputLayer] = None,
        dataset_path: str = None,
    ):
        request_body = RequestQuantizeTask(
            input_model_id=input_model_id,
            quantization_mode=quantization_mode,
            quantization_options=quantization_options,
            input_layers=input_layers,
        )
        logger.info(f"Request Quantize body: {request_body}")
        token_header = AuthorizationHeader(access_token=access_token)

        file_object = self.create_upload_dataset(dataset_path)

        if quantization_mode == QuantizationMode.UNIFORM_PRECISION_QUANTIZATION:
            response = self.quantize_task.start_plain_quantization(request_body, token_header, file_object)
        elif quantization_mode == QuantizationMode.RECOMMEND_QUANTIZATION:
            response = self.quantize_task.start_recommendation_precision(request_body, token_header, file_object)
        elif quantization_mode == QuantizationMode.CUSTOM_PRECISION_QUANTIZATION:
            response = self.quantize_task.start_custom_quantization(request_body, token_header, file_object)
        elif quantization_mode == QuantizationMode.AUTOMATIC_QUANTIZATION:
            response = self.quantize_task.start_auto_quantization(request_body, token_header, file_object)

        logger.info(f"Request Quantize result: {response}")
        return response

    def cancel_task(self, access_token: str, task_id: str) -> ResponseQuantizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize Cancel task_id: {task_id}")

        quantize_task_response = self.quantize_task.cancel(headers=token_header, task_id=task_id)
        logger.info(f"Request Quantize Cancel result: {quantize_task_response}")
        return quantize_task_response

    def read_task(self, access_token: str, task_id: str) -> ResponseQuantizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize Read task_id: {task_id}")

        quantize_task_response = self.quantize_task.read(headers=token_header, task_id=task_id)
        logger.info(f"Request Quantize Task Info: {quantize_task_response}")
        return quantize_task_response

    def delete_task(self, access_token: str, task_id: str) -> ResponseQuantizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize delete task_id: {task_id}")

        quantize_task_response = self.quantize_task.delete(headers=token_header, task_id=task_id)
        logger.info(f"Request Quantize Delete Info: {quantize_task_response}")
        return quantize_task_response

    def read_status(self, access_token: str, task_id: str) -> ResponseQuantizeStatusItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Quantize read_status task_id: {task_id}")

        quantize_task_response = self.quantize_task.status(headers=token_header, task_id=task_id)
        logger.info(f"Request Quantize Task Status: {quantize_task_response}")
        return quantize_task_response

    def read_options(self, access_token: str) -> ResponseQuantizeOptionItems:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info("Request Quantize options")

        quantize_task_option_response = self.quantize_task.options(headers=token_header)
        logger.info(f"Response Quantize Task Options: {quantize_task_option_response}")
        return quantize_task_option_response
