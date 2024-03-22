from dataclasses import asdict

from netspresso.clients.utils.requester import Requester
from netspresso.clients.launcher.v2.enums import LauncherTask
from netspresso.clients.launcher.v2.interfaces import TaskInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    UploadFile,
    RequestBenchmark,
    ResponseBenchmarkTaskItem,
    ResponseBenchmarkOptionItems,
    ResponseBenchmarkStatusItem,
)


class BenchmarkTaskAPI(TaskInterface):
    def __init__(self, url):
        self.task_type = LauncherTask.BENCHMARK.value
        self.base_url = url
        self.task_base_url = f"{self.base_url}/{self.task_type}/tasks"
        self.option_base_url = f"{self.base_url}/{self.task_type}/options"

    def start(
        self,
        request_body: RequestBenchmark,
        headers: AuthorizationHeader,
        file: UploadFile = None,
    ) -> ResponseBenchmarkTaskItem:
        endpoint = f"{self.task_base_url}"
        response = Requester().post_as_json(
            url=endpoint, request_body=asdict(request_body), headers=asdict(headers)
        )
        return ResponseBenchmarkTaskItem(**response.json())

    def cancel(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseBenchmarkTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}/cancel"
        response = Requester().post_as_json(
            url=endpoint, request_body={}, headers=asdict(headers)
        )
        return ResponseBenchmarkTaskItem(**response.json())

    def read(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseBenchmarkTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkTaskItem(**response.json())

    def delete(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseBenchmarkTaskItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().delete(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkTaskItem(**response.json())

    def status(
        self, headers: AuthorizationHeader, task_id: str
    ) -> ResponseBenchmarkStatusItem:
        endpoint = f"{self.task_base_url}/{task_id}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkStatusItem(**response.json())

    def options(self, headers: AuthorizationHeader) -> ResponseBenchmarkOptionItems:
        endpoint = f"{self.option_base_url}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkOptionItems(**response.json())

    def option_by_model_framework(
        self, headers: AuthorizationHeader, model_framework: str
    ) -> ResponseBenchmarkOptionItems:
        endpoint = f"{self.option_base_url}/framework/{model_framework}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkOptionItems(**response.json())

    def option_by_target_device(
        self, headers: AuthorizationHeader, target_device: str
    ) -> ResponseBenchmarkOptionItems:
        endpoint = f"{self.option_base_url}/details/{target_device}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseBenchmarkOptionItems(**response.json())
