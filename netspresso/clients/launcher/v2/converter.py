import os

from loguru import logger

from netspresso.clients.launcher.v2.implements import ConvertTaskAPI, ModelAPI
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    InputLayer,
    RequestConvert,
    RequestModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseConvertDownloadModelUrlItem,
    ResponseConvertFrameworkOptionItems,
    ResponseConvertOptionItems,
    ResponseConvertStatusItem,
    ResponseConvertTaskItem,
    ResponseModelItem,
    ResponseModelOptions,
    ResponseModelUploadUrl,
    UploadDataset,
    UploadFile,
)
from netspresso.clients.utils.common import read_file_bytes
from netspresso.enums import DataType, DeviceName, Framework, LauncherTask, SoftwareVersion


class Converter:
    def __init__(self, url):
        self.task_type = LauncherTask.CONVERT.value
        self.convert_task = ConvertTaskAPI(url=url)
        self.convert_model = ModelAPI(url=url, task_type=self.task_type)

    def presigned_model_upload_url(self, access_token: str, input_model_path: str) -> ResponseModelUploadUrl:
        object_name = os.path.basename(input_model_path)

        get_upload_url_request_body = RequestModelUploadUrl(object_name=object_name, task=self.task_type)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert model upload: path - {input_model_path}")

        upload_url_response_body = self.convert_model.get_upload_url(
            request_params=get_upload_url_request_body, headers=token_header
        )
        logger.info(f"Request Convert upload_url result: {upload_url_response_body}")
        return upload_url_response_body

    def upload_model_file(self, access_token: str, input_model_path: str, presigned_upload_url: str) -> str:
        object_name = os.path.basename(input_model_path)
        file_content = read_file_bytes(file_path=input_model_path)

        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Convert model validation:"
            f" path | {input_model_path} "
            f" presigned_upload_url | {presigned_upload_url}"
        )

        get_model_upload_request_body = RequestUploadModel(url=presigned_upload_url)
        model_file_object = UploadFile(file_name=object_name, file_content=file_content)

        upload_result = self.convert_model.upload(
            request_body=get_model_upload_request_body,
            file=model_file_object,
            headers=token_header,
        )
        logger.info(f"Request Convert upload_model_file result: {upload_result}")
        return upload_result

    def validate_model_file(self, access_token: str, input_model_path: str, ai_model_id: str) -> ResponseModelItem:
        object_name = os.path.basename(input_model_path)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert model validation:" f" path - {input_model_path}" f" ai_model_id - {ai_model_id}")

        get_validate_model_request_body = RequestValidateModel(
            ai_model_id=ai_model_id,
            display_name=object_name,
        )

        validated_model = self.convert_model.validate(
            request_body=get_validate_model_request_body, headers=token_header
        )
        logger.info(f"Request Convert validate_model result: {validated_model}")
        return validated_model

    def download_model_file(self, access_token, convert_task_uuid) -> ResponseConvertDownloadModelUrlItem:
        token_header = AuthorizationHeader(access_token=access_token)
        download_url = self.convert_task.get_download_url(headers=token_header, convert_task_uuid=convert_task_uuid)
        logger.info(f"Request converted model download_url: {download_url}")
        return download_url

    def read_model_task_options(self, access_token, ai_model_id) -> ResponseModelOptions:
        token_header = AuthorizationHeader(access_token=access_token)
        model_task_options = self.convert_model.options(headers=token_header, ai_model_id=ai_model_id)
        logger.info(f"Request model task_options: {model_task_options}")
        return model_task_options

    def start_task(
        self,
        access_token: str,
        input_model_id: str,
        target_device_name: DeviceName,
        target_framework: Framework,
        data_type: DataType = None,
        input_layer: InputLayer = None,
        software_version: SoftwareVersion = None,
        dataset_path: str = None,
    ) -> ResponseConvertTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        request_body = RequestConvert(
            input_model_id=input_model_id,
            target_framework=target_framework,
            target_device_name=target_device_name,
            data_type=data_type,
            input_layer=input_layer,
            software_version=software_version,
        )
        logger.info(f"Request Convert body: {request_body}")

        if dataset_path:
            dataset_filename = os.path.basename(dataset_path)
            file_object = UploadDataset(
                file_name=dataset_filename,
                file_content=read_file_bytes(dataset_path),
            )
        else:
            file_object = None
        convert_task_response = self.convert_task.start(
            request_body=request_body, headers=token_header, file=file_object
        )
        logger.info(f"Request Convert result: {convert_task_response}")
        return convert_task_response

    def cancel_task(self, access_token: str, task_id: str) -> ResponseConvertTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert Cancel task_id: {task_id}")

        convert_task_response = self.convert_task.cancel(headers=token_header, task_id=task_id)
        logger.info(f"Request Convert Cancel result: {convert_task_response}")
        return convert_task_response

    def read_task(self, access_token: str, task_id: str) -> ResponseConvertTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert Read task_id: {task_id}")

        convert_task_response = self.convert_task.read(headers=token_header, task_id=task_id)
        logger.info(f"Request Convert Task Info: {convert_task_response}")
        return convert_task_response

    def delete_task(self, access_token: str, task_id: str) -> ResponseConvertTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert delete task_id: {task_id}")

        convert_task_response = self.convert_task.delete(headers=token_header, task_id=task_id)
        logger.info(f"Request Convert Delete Info: {convert_task_response}")
        return convert_task_response

    def read_status(self, access_token: str, task_id: str) -> ResponseConvertStatusItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Convert read_status task_id: {task_id}")

        convert_task_response = self.convert_task.status(headers=token_header, task_id=task_id)
        logger.info(f"Request Convert Task Status: {convert_task_response}")
        return convert_task_response

    def read_options(self, access_token: str) -> ResponseConvertOptionItems:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info("Request Convert options")

        convert_task_option_response = self.convert_task.options(headers=token_header)
        logger.info(f"Response Convert Task Options: {convert_task_option_response}")
        return convert_task_option_response

    def read_framework_options(self, access_token: str, framework: Framework) -> ResponseConvertFrameworkOptionItems:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info("Request Convert options")

        convert_task_option_response = self.convert_task.options_by_model_framework(
            headers=token_header, model_framework=framework
        )
        logger.info(f"Request Convert Task Options by model_framework: {convert_task_option_response}")
        return convert_task_option_response
