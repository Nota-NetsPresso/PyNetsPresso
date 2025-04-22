from dataclasses import asdict
from enum import Enum

from loguru import logger

from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    ResponseQuantizeDownloadModelUrlItem,
    ResponseQuantizeOptionItems,
    ResponseQuantizeStatusItem,
    ResponseQuantizeTaskItem,
    UploadFile,
)
from netspresso.clients.launcher.v2.schemas.common import UploadDataset
from netspresso.clients.launcher.v2.schemas.task.quantize.request_body import RequestQuantizeTask
from netspresso.clients.utils.requester import Requester
from netspresso.enums import LauncherTask


class QuantizeTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.QUANTIZE.value
        self.base_url = url
        self.task_base_url = f"{self.base_url}/{self.task_type}/tasks"
        self.option_base_url = f"{self.base_url}/{self.task_type}/options"
        self.model_base_url = f"{self.base_url}/{self.task_type}/models"

    @staticmethod
    def custom_asdict_factory(data):
        def convert_value(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return {k: convert_value(v) for k, v in data}

    def _request_to_quantizer(self, request_body, endpoint, headers, file):
        logger.info(f"Request_Body: {asdict(request_body)}")
        response = Requester().post_as_form(
            url=endpoint,
            request_body=asdict(request_body, dict_factory=self.custom_asdict_factory),
            headers=headers.to_dict(),
            binary=file.files if file else None,
        )

        return ResponseQuantizeTaskItem(**response.json())

    def start_plain_quantization(
        self, request_body: RequestQuantizeTask, headers: AuthorizationHeader, file: UploadDataset = None
    ) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}"

        response = self._request_to_quantizer(request_body, endpoint, headers, file)

        return response

    def start_recommendation_precision(
        self, request_body: RequestQuantizeTask, headers: AuthorizationHeader, file: UploadDataset = None
    ) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}/recommendation"

        response = self._request_to_quantizer(request_body, endpoint, headers, file)

        return response

    def start_custom_quantization(
        self, request_body: RequestQuantizeTask, headers: AuthorizationHeader, file: UploadDataset = None
    ) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}/custom"

        response = self._request_to_quantizer(request_body, endpoint, headers, file)

        return response

    def start_auto_quantization(
        self, request_body: RequestQuantizeTask, headers: AuthorizationHeader, file: UploadDataset = None
    ) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}/auto"

        response = self._request_to_quantizer(request_body, endpoint, headers, file)

        return response

    def cancel(self, headers: AuthorizationHeader, task_id: str) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(url=endpoint, request_body={}, headers=headers.to_dict())
        return ResponseQuantizeTaskItem(**response.json())

    def read(self, headers: AuthorizationHeader, task_id: str) -> ResponseQuantizeTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseQuantizeTaskItem(**response.json())

    def status(self, headers: AuthorizationHeader, task_id: str) -> ResponseQuantizeStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseQuantizeStatusItem(**response.json())

    def options(self, headers: AuthorizationHeader) -> ResponseQuantizeOptionItems:
        endpoint = f"{self.option_base_url}"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseQuantizeOptionItems(**response.json())

    def get_download_url(
        self, headers: AuthorizationHeader, quantize_task_uuid: str
    ) -> ResponseQuantizeDownloadModelUrlItem:
        endpoint = f"{self.task_base_url}/{quantize_task_uuid}/results"
        response = Requester().get(url=endpoint, headers=headers.to_dict())
        return ResponseQuantizeDownloadModelUrlItem(**response.json())

    def start(self, request_body, headers, file, endpoint):
        pass

    def options_by_model_framework(self, headers: AuthorizationHeader, model_framework: str):
        pass

    def delete(self, headers: AuthorizationHeader, task_id: str):
        pass
