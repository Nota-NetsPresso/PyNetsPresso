import os
from enum import Enum
from typing import Any, Dict, Final, Optional
from urllib.parse import urljoin

import requests

from netspresso.clients.dataforge.schemas.response_body import (
    DatasetPayload,
    DatasetResponse,
    DatasetsPayload,
    DatasetsResponse,
)

BASE_URL: Final[str] = os.getenv("DATAFORGE_BASE_URL")

# API 버전을 상수로 관리
API_VERSION: Final[str] = "v1"
BASE_PATH: Final[str] = f"/api/{API_VERSION}"

class DataForgeEndpoint:
    """Class for managing DataForge API endpoints"""

    @staticmethod
    def get_path(endpoint_value: str, **kwargs) -> str:
        """
        Returns the endpoint path with actual values filled in.

        Args:
            endpoint_value (str): The endpoint path template
            **kwargs: Parameters to fill in the path (project_id, dataset_uuid, etc.)

        Returns:
            str: The endpoint path with actual values
        """
        return endpoint_value.format(**kwargs)

    class Dataset(str, Enum):
        """
        API endpoints for dataset operations

        Attributes:
            GET: Endpoint for retrieving details of a specific dataset
            LIST: Endpoint for retrieving all datasets in a project
        """
        GET = f"{BASE_PATH}/dataset/{{project_id}}/{{dataset_uuid}}"
        LIST = f"{BASE_PATH}/dataset/{{project_id}}"


class HTTPMethod(str, Enum):
    """HTTP Methods"""
    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"
    DELETE = "DELETE"


class DataForgeClient:
    """
    Client class for communicating with DataForge API

    This class handles all requests to the DataForge API endpoints.
    """

    def __init__(self, base_url: str = BASE_URL):
        """
        Initialize the DataForge client.

        Args:
            base_url (str): Base URL for the DataForge API
        """
        self.base_url = base_url.rstrip('/')

    def _make_request(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API.

        Args:
            method (HTTPMethod): HTTP method to use for the request
            endpoint (str): API endpoint path
            params (Optional[Dict[str, Any]]): Query parameters for the request
            json_data (Optional[Dict[str, Any]]): JSON data to send in request body

        Returns:
            Dict[str, Any]: Raw JSON response from the API
        """
        url = urljoin(self.base_url, endpoint)
        response = requests.request(
            method=method.value,
            url=url,
            params=params,
            json=json_data
        )
        response.raise_for_status()
        return response.json()

    def get_dataset(self, project_id: str, dataset_uuid: str) -> DatasetPayload:
        """
        Retrieve information about a specific dataset.

        Args:
            project_id (str): Project ID
            dataset_uuid (str): Dataset UUID

        Returns:
            DatasetPayload: Validated dataset information extracted from response
        """
        endpoint = DataForgeEndpoint.get_path(
            DataForgeEndpoint.Dataset.GET,
            project_id=project_id,
            dataset_uuid=dataset_uuid
        )
        response = self._make_request(HTTPMethod.GET, endpoint)
        return DatasetResponse(**response).data

    def get_datasets(self, project_id: str) -> DatasetsPayload:
        """
        Retrieve all datasets for a project.

        Args:
            project_id (str): Project ID

        Returns:
            DatasetsPayload: Validated list of datasets extracted from response
        """
        endpoint = DataForgeEndpoint.get_path(
            DataForgeEndpoint.Dataset.LIST,
            project_id=project_id
        )
        response = self._make_request(HTTPMethod.GET, endpoint)
        return DatasetsResponse(**response).data
