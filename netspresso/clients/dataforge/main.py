import os

from netspresso.clients.config import Config, ServiceModule, ServiceName
from netspresso.clients.dataforge.schemas.response_body import (
    DatasetResponse,
    DatasetsResponse,
    DatasetVersionResponse,
    DatasetVersionsResponse,
)
from netspresso.clients.utils.requester import Requester


def get_headers(api_key=None, json_type=False):
    headers = {}
    if api_key:
        headers["api_key"] = f"{api_key}"
    if json_type:
        headers["Content-Type"] = "application/json"
    return headers


class DataForgeClient:
    def __init__(self, https: bool = False):
        self.config = Config(ServiceName.DATAFORGE, ServiceModule.DATAFORGE)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.api_key = os.getenv("DATAFORGE_API_KEY")

        if https:
            self.url = f"https://{self.host}:{self.port}{self.prefix}"
        else:
            self.url = f"http://{self.host}:{self.port}{self.prefix}"

    def get_datasets(self, project_id: str) -> DatasetsResponse:
        url = f"{self.url}/dataset/{project_id}"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(self.api_key), verify=False)

        return DatasetsResponse(**response.json())

    def get_dataset(self, project_id: str, dataset_uuid: str) -> DatasetResponse:
        url = f"{self.url}/dataset/{project_id}/{dataset_uuid}"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(self.api_key), verify=False)

        return DatasetResponse(**response.json())

    def get_dataset_versions(self, dataset_uuid: str, split: str) -> DatasetVersionsResponse:
        url = f"{self.url}/dataset/version/{dataset_uuid}/{split}/all"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(self.api_key), verify=False)

        return DatasetVersionsResponse(**response.json())

    def get_latest_dataset_version(self, dataset_uuid: str, split: str) -> DatasetVersionResponse:
        url = f"{self.url}/dataset/version/{dataset_uuid}/{split}/latest"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(self.api_key), verify=False)

        return DatasetVersionResponse(**response.json())
