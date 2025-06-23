import os

from loguru import logger

from netspresso.clients.launcher.v2.implements import ModelAPI, SimulateTaskAPI
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    RequestModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseModelItem,
    ResponseModelOptions,
    ResponseModelUploadUrl,
    UploadFile,
)
from netspresso.clients.launcher.v2.schemas.common import UploadDataset
from netspresso.clients.launcher.v2.schemas.task.simulate.request_body import RequestCreateSimulateTask
from netspresso.clients.launcher.v2.schemas.task.simulate.response_body import (
    ResponseSimulateStatusItem,
    ResponseSimulateTaskItem,
)
from netspresso.clients.utils.common import read_file_bytes
from netspresso.enums import LauncherTask
from netspresso.enums.simulate import SimulateTaskType


class Simulator:
    def __init__(self, url):
        self.task_type = LauncherTask.SIMULATE.value
        self.simulate_task = SimulateTaskAPI(url=url)
        self.simulate_model = ModelAPI(url=url, task_type=self.task_type)

    def presigned_model_upload_url(self, access_token: str, input_model_path: str) -> ResponseModelUploadUrl:
        object_name = os.path.basename(input_model_path)

        get_upload_url_request_body = RequestModelUploadUrl(object_name=object_name, task=self.task_type)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate model upload: path - {input_model_path}")

        upload_url_response_body = self.simulate_model.get_upload_url(
            request_params=get_upload_url_request_body, headers=token_header
        )
        logger.info(f"Request Simulate upload_url result: {upload_url_response_body}")
        return upload_url_response_body

    def upload_model_file(self, access_token: str, input_model_path: str, presigned_upload_url: str) -> str:
        object_name = os.path.basename(input_model_path)
        file_content = read_file_bytes(file_path=input_model_path)

        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Simulate model validation:"
            f" path | {input_model_path} "
            f" presigned_upload_url | {presigned_upload_url}"
        )

        get_model_upload_request_body = RequestUploadModel(url=presigned_upload_url)
        model_file_object = UploadFile(file_name=object_name, file_content=file_content)

        upload_result = self.simulate_model.upload(
            request_body=get_model_upload_request_body,
            file=model_file_object,
            headers=token_header,
        )
        logger.info(f"Request Simulate upload_model_file result: {upload_result}")
        return upload_result

    def validate_model_file(self, access_token: str, input_model_path: str, ai_model_id: str) -> ResponseModelItem:
        object_name = os.path.basename(input_model_path)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate model validation:" f" path - {input_model_path}" f" ai_model_id - {ai_model_id}")

        get_validate_model_request_body = RequestValidateModel(
            ai_model_id=ai_model_id,
            display_name=object_name,
        )

        validated_model = self.simulate_model.validate(
            request_body=get_validate_model_request_body, headers=token_header
        )
        logger.info(f"Request Simulate validate_model result: {validated_model}")
        return validated_model

    def read_model_task_options(self, access_token, ai_model_id) -> ResponseModelOptions:
        token_header = AuthorizationHeader(access_token=access_token)
        model_task_options = self.simulate_model.options(headers=token_header, ai_model_id=ai_model_id)
        logger.info(f"Request model task_options: {model_task_options}")
        return model_task_options

    def start_task(
        self,
        access_token: str,
        base_model_id: str,
        target_model_id: str,
        type: SimulateTaskType,
        dataset_path: str = None,
    ) -> ResponseSimulateTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        request_body = RequestCreateSimulateTask(
            base_model_id=base_model_id,
            target_model_id=target_model_id,
            type=type,
        )
        logger.info(f"Request Simulate body: {request_body}")

        if dataset_path:
            dataset_filename = os.path.basename(dataset_path)
            file_object = UploadDataset(
                file_name=dataset_filename,
                file_content=read_file_bytes(dataset_path),
            )
        else:
            file_object = None

        simulate_task_response = self.simulate_task.start(
            request_body=request_body, headers=token_header, file=file_object
        )
        logger.info(f"Request Simulate result: {simulate_task_response}")
        return simulate_task_response

    def cancel_task(self, access_token: str, task_id: str) -> ResponseSimulateTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate Cancel task_id: {task_id}")

        simulate_task_response = self.simulate_task.cancel(headers=token_header, task_id=task_id)
        logger.info(f"Request Simulate Cancel result: {simulate_task_response}")
        return simulate_task_response

    def read_task(self, access_token: str, task_id: str) -> ResponseSimulateTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate Read task_id: {task_id}")

        simulate_task_response = self.simulate_task.read(headers=token_header, task_id=task_id)
        logger.info(f"Request Simulate Task Info: {simulate_task_response}")
        return simulate_task_response

    def delete_task(self, access_token: str, task_id: str) -> ResponseSimulateTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate delete task_id: {task_id}")

        simulate_task_response = self.simulate_task.delete(headers=token_header, task_id=task_id)
        logger.info(f"Request Simulate Delete Info: {simulate_task_response}")
        return simulate_task_response

    def read_status(self, access_token: str, task_id: str) -> ResponseSimulateStatusItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Simulate read_status task_id: {task_id}")

        simulate_task_response = self.simulate_task.status(headers=token_header, task_id=task_id)
        logger.info(f"Request Simulate Task Status: {simulate_task_response}")
        return simulate_task_response
