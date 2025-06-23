from dataclasses import asdict
from enum import Enum

from loguru import logger

from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    RequestCreateSimulateTask,
    ResponseSimulateStatusItem,
    ResponseSimulateTaskItem,
    UploadFile,
)
from netspresso.clients.utils.requester import Requester
from netspresso.enums import LauncherTask


class SimulateTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.SIMULATE.value
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
        request_body: RequestCreateSimulateTask,
        headers: AuthorizationHeader,
        file: UploadFile = None,
    ) -> ResponseSimulateTaskItem:
        endpoint = f"{self.task_base_url}/compare"

        logger.info(f"Request_Body: {asdict(request_body)}")
        response = Requester().post_as_form(
            url=endpoint,
            request_body=asdict(request_body, dict_factory=self.custom_asdict_factory),
            headers=headers.to_dict(),
            binary=file.files if file else None,
        )
        return ResponseSimulateTaskItem(**response.json())

    def cancel(self, headers: AuthorizationHeader, task_id: str) -> ResponseSimulateTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(url=endpoint, request_body={}, headers=headers.to_dict())
        return ResponseSimulateTaskItem(**response.json())

    def read(self, headers: AuthorizationHeader, task_id: str) -> ResponseSimulateTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseSimulateTaskItem(**response.json())

    def status(self, headers: AuthorizationHeader, task_id: str) -> ResponseSimulateStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseSimulateStatusItem(**response.json())

    def get_download_url(self, headers: AuthorizationHeader, simulate_task_uuid: str):
        pass

    def options(self, headers: AuthorizationHeader):
        pass

    def options_by_model_framework(self, headers: AuthorizationHeader, model_framework: str):
        pass

    def delete(self, headers: AuthorizationHeader, task_id: str):
        pass
