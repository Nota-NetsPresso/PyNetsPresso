import os
from typing import List

from loguru import logger

from netspresso.clients.launcher.v2.implements import GraphOptimizeTaskAPI, ModelAPI
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    RequestCreateGraphOptimizeTask,
    RequestModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseGraphOptimizeDownloadModelUrlItem,
    ResponseGraphOptimizeStatusItem,
    ResponseGraphOptimizeTaskItem,
    ResponseModelItem,
    ResponseModelOptions,
    ResponseModelUploadUrl,
    UploadFile,
)
from netspresso.clients.utils.common import read_file_bytes
from netspresso.enums import LauncherTask
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler


class GraphOptimizer:
    def __init__(self, url):
        self.task_type = LauncherTask.GRAPH_OPTIMIZE.value
        self.graph_optimize_task = GraphOptimizeTaskAPI(url=url)
        self.graph_optimize_model = ModelAPI(url=url, task_type=self.task_type)

    def presigned_model_upload_url(self, access_token: str, input_model_path: str) -> ResponseModelUploadUrl:
        object_name = os.path.basename(input_model_path)

        get_upload_url_request_body = RequestModelUploadUrl(object_name=object_name, task=self.task_type)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize model upload: path - {input_model_path}")

        upload_url_response_body = self.graph_optimize_model.get_upload_url(
            request_params=get_upload_url_request_body, headers=token_header
        )
        logger.info(f"Request Graph Optimize upload_url result: {upload_url_response_body}")
        return upload_url_response_body

    def upload_model_file(self, access_token: str, input_model_path: str, presigned_upload_url: str) -> str:
        object_name = os.path.basename(input_model_path)
        file_content = read_file_bytes(file_path=input_model_path)

        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(
            f"Request Graph Optimize model validation:"
            f" path | {input_model_path} "
            f" presigned_upload_url | {presigned_upload_url}"
        )

        get_model_upload_request_body = RequestUploadModel(url=presigned_upload_url)
        model_file_object = UploadFile(file_name=object_name, file_content=file_content)

        upload_result = self.graph_optimize_model.upload(
            request_body=get_model_upload_request_body,
            file=model_file_object,
            headers=token_header,
        )
        logger.info(f"Request Graph Optimize upload_model_file result: {upload_result}")
        return upload_result

    def validate_model_file(self, access_token: str, input_model_path: str, ai_model_id: str) -> ResponseModelItem:
        object_name = os.path.basename(input_model_path)
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize model validation:" f" path - {input_model_path}" f" ai_model_id - {ai_model_id}")

        get_validate_model_request_body = RequestValidateModel(
            ai_model_id=ai_model_id,
            display_name=object_name,
        )

        validated_model = self.graph_optimize_model.validate(
            request_body=get_validate_model_request_body, headers=token_header
        )
        logger.info(f"Request Graph Optimize validate_model result: {validated_model}")
        return validated_model

    def download_model_file(self, access_token, graph_optimize_task_id) -> ResponseGraphOptimizeDownloadModelUrlItem:
        token_header = AuthorizationHeader(access_token=access_token)
        download_url = self.graph_optimize_task.get_download_url(
            headers=token_header, graph_optimize_task_uuid=graph_optimize_task_id
        )
        logger.info(f"Request graph optimized model download_url: {download_url}")
        return download_url

    def read_model_task_options(self, access_token, ai_model_id) -> ResponseModelOptions:
        token_header = AuthorizationHeader(access_token=access_token)
        model_task_options = self.graph_optimize_model.options(headers=token_header, ai_model_id=ai_model_id)
        logger.info(f"Request model task_options: {model_task_options}")
        return model_task_options

    def start_task(
        self,
        access_token: str,
        input_model_id: str,
        pattern_handlers: List[GraphOptimizePatternHandler],
    ) -> ResponseGraphOptimizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        request_body = RequestCreateGraphOptimizeTask(
            ai_model_id=input_model_id,
            pattern_handlers=pattern_handlers,
        )
        logger.info(f"Request Graph Optimize body: {request_body}")

        graph_optimize_task_response = self.graph_optimize_task.start(
            request_body=request_body, headers=token_header
        )
        logger.info(f"Request Graph Optimize result: {graph_optimize_task_response}")
        return graph_optimize_task_response

    def cancel_task(self, access_token: str, task_id: str) -> ResponseGraphOptimizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize Cancel task_id: {task_id}")

        graph_optimize_task_response = self.graph_optimize_task.cancel(headers=token_header, task_id=task_id)
        logger.info(f"Request Graph Optimize Cancel result: {graph_optimize_task_response}")
        return graph_optimize_task_response

    def read_task(self, access_token: str, task_id: str) -> ResponseGraphOptimizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize Read task_id: {task_id}")

        graph_optimize_task_response = self.graph_optimize_task.read(headers=token_header, task_id=task_id)
        logger.info(f"Request Graph Optimize Task Info: {graph_optimize_task_response}")
        return graph_optimize_task_response

    def delete_task(self, access_token: str, task_id: str) -> ResponseGraphOptimizeTaskItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize delete task_id: {task_id}")

        graph_optimize_task_response = self.graph_optimize_task.delete(headers=token_header, task_id=task_id)
        logger.info(f"Request Graph Optimize Delete Info: {graph_optimize_task_response}")
        return graph_optimize_task_response

    def read_status(self, access_token: str, task_id: str) -> ResponseGraphOptimizeStatusItem:
        token_header = AuthorizationHeader(access_token=access_token)
        logger.info(f"Request Graph Optimize read_status task_id: {task_id}")

        graph_optimize_task_response = self.graph_optimize_task.status(headers=token_header, task_id=task_id)
        logger.info(f"Request Graph Optimize Task Status: {graph_optimize_task_response}")
        return graph_optimize_task_response
