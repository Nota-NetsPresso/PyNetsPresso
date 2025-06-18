from dataclasses import asdict
from enum import Enum
from typing import Optional

from loguru import logger

from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    RequestCreateGraphOptimizeTask,
    ResponseGraphOptimizeDownloadModelUrlItem,
    ResponseGraphOptimizeStatusItem,
    ResponseGraphOptimizeTaskItem,
    UploadFile,
)
from netspresso.clients.utils.requester import Requester
from netspresso.enums import LauncherTask


class GraphOptimizeTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.GRAPH_OPTIMIZE.value
        self.base_url = url
        self.task_base_url = f"{self.base_url}/{self.task_type}/tasks"
        self.model_base_url = f"{self.base_url}/{self.task_type}/models"

    @staticmethod
    def custom_asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return {k: convert_value(v) for k, v in data}

    def start(
        self,
        request_body: RequestCreateGraphOptimizeTask,
        headers: AuthorizationHeader,
        file: Optional[UploadFile] = None,
    ) -> ResponseGraphOptimizeTaskItem:
        endpoint = f"{self.task_base_url}"

        logger.info(f"Request_Body: {asdict(request_body)}")
        response = Requester().post_as_json(url=endpoint, request_body=asdict(request_body), headers=headers.to_dict())
        return ResponseGraphOptimizeTaskItem(**response.json())

    def cancel(self, headers: AuthorizationHeader, task_id: str) -> ResponseGraphOptimizeTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(url=endpoint, request_body={}, headers=headers.to_dict())
        return ResponseGraphOptimizeTaskItem(**response.json())

    def read(self, headers: AuthorizationHeader, task_id: str) -> ResponseGraphOptimizeTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseGraphOptimizeTaskItem(**response.json())

    def status(self, headers: AuthorizationHeader, task_id: str) -> ResponseGraphOptimizeStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseGraphOptimizeStatusItem(**response.json())

    def get_download_url(
        self, headers: AuthorizationHeader, graph_optimize_task_uuid: str
    ) -> ResponseGraphOptimizeDownloadModelUrlItem:
        endpoint = f"{self.task_base_url}/{graph_optimize_task_uuid}/models/download"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseGraphOptimizeDownloadModelUrlItem(**response.json())

    def options(self, headers: AuthorizationHeader):
        pass

    def options_by_model_framework(self, headers: AuthorizationHeader, model_framework: str):
        pass

    def delete(self, headers: AuthorizationHeader, task_id: str):
        pass
