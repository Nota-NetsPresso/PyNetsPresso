from netspresso.clients.config import Config, ServiceModule, ServiceName
from netspresso.clients.dataforge.schemas.response_body import (
    DatasetResponse,
    DatasetsResponse,
    DatasetVersionResponse,
    DatasetVersionsResponse,
)
from netspresso.clients.utils.common import get_headers
from netspresso.clients.utils.requester import Requester


class DataForgeClient:
    def __init__(self, https: bool = False):
        self.config = Config(ServiceName.DATAFORGE, ServiceModule.DATAFORGE)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX

        if https:
            self.url = f"https://{self.host}:{self.port}{self.prefix}"
        else:
            self.url = f"http://{self.host}:{self.port}{self.prefix}"

    def get_datasets(self, project_id: str, access_token: str) -> DatasetsResponse:
        url = f"{self.url}/dataset/{project_id}"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(access_token), verify=False)

        return DatasetsResponse(**response.json())

    def get_dataset(self, project_id: str, dataset_uuid: str, access_token: str) -> DatasetResponse:
        url = f"{self.url}/dataset/{project_id}/{dataset_uuid}"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(access_token), verify=False)

        return DatasetResponse(**response.json())

    def get_dataset_versions(self, dataset_uuid: str, split: str, access_token: str) -> DatasetVersionsResponse:
        url = f"{self.url}/dataset/version/{dataset_uuid}/{split}/all"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(access_token), verify=False)

        return DatasetVersionsResponse(**response.json())

    def get_latest_dataset_version(self, dataset_uuid: str, split: str, access_token: str, include_info: bool = True) -> DatasetVersionResponse:
        include_info_value = "true" if include_info else "false"
        url = f"{self.url}/dataset/version/{dataset_uuid}/{split}/latest?include_info={include_info_value}"

        # TODO: Remove verify=False in production. This is only for testing purposes.
        response = Requester.get(url=url, headers=get_headers(access_token), verify=False)

        return DatasetVersionResponse(**response.json())
