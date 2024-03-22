from dataclasses import asdict
from loguru import logger

from netspresso.clients.launcher.v2.enums import LauncherTask
from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    UploadFile,
    RequestConvert,
    ResponseConvertTaskItem,
    ResponseConvertOptionItems,
    ResponseConvertStatusItem,
)
from netspresso.clients.utils.requester import Requester


class ConvertTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.CONVERT.value
        self.base_url = url
        self.task_base_url = f"{self.base_url}/{self.task_type}/tasks"
        self.option_base_url = f"{self.base_url}/{self.task_type}/options"

    def start(
        self,
        request_body: RequestConvert,
        headers: AuthorizationHeader,
        file: UploadFile = None,
    ) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}"

        logger.info(f"Request_Body: {asdict(request_body)}")
        response = Requester().post_as_form(
            url=endpoint,
            request_body=asdict(request_body),
            headers=asdict(headers),
            binary=file.files if file else None,
        )
        return ResponseConvertTaskItem(**response.json())

    def cancel(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(
            url=endpoint, request_body={}, headers=asdict(headers)
        )
        return ResponseConvertTaskItem(**response.json())

    def read(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseConvertTaskItem(**response.json())

    def delete(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseConvertTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().delete(url=endpoint, headers=asdict(headers))
        return ResponseConvertTaskItem(**response.json())

    def status(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseConvertStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseConvertStatusItem(**response.json())

    def options(self, headers: AuthorizationHeader) -> ResponseConvertOptionItems:
        endpoint = f"{self.option_base_url}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseConvertOptionItems(**response.json())

    def option_by_model_framework(
        self, headers: AuthorizationHeader, model_framework: str
    ) -> ResponseConvertOptionItems:
        endpoint = f"{self.option_base_url}/framework/{model_framework}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseConvertOptionItems(**response.json())

    def option_by_target_device(
        self, headers: AuthorizationHeader, target_framework: str
    ) -> ResponseConvertOptionItems:
        endpoint = f"{self.option_base_url}/details/{target_framework}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseConvertOptionItems(**response.json())
