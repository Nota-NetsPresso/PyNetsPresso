from dataclasses import asdict

from netspresso.clients.launcher.v2.enums import LauncherTask
from netspresso.clients.launcher.v2.interfaces import ModelInterface
from netspresso.clients.launcher.v2.schemas import (
    AuthorizationHeader,
    UploadFile,
    RequestPagination,
    RequestModelUploadUrl,
    ResponseModelUploadUrl,
    RequestUploadModel,
    RequestValidateModel,
    ResponseModelItem,
    ResponseModelItems,
    ResponseModelStatus,
)
from netspresso.clients.utils.requester import Requester


class ModelAPI(ModelInterface):
    def __init__(self, url: str, task_type: LauncherTask):
        self.task_type = task_type
        self.base_url = url
        self.model_base_url = f"{self.base_url}/models"

    def get_upload_url(
        self, request_params: RequestModelUploadUrl, headers: AuthorizationHeader
    ) -> ResponseModelUploadUrl:
        endpoint = f"{self.model_base_url}/presigned_url"
        response = Requester().get(
            url=endpoint, headers=asdict(headers), params=asdict(request_params)
        )
        return ResponseModelUploadUrl(**response.json())

    def get_download_url(self, headers: AuthorizationHeader, ai_model_id: str) -> str:
        endpoint = f"{self.model_base_url}/{ai_model_id}/download_url"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return response.text

    def upload(
        self,
        request_body: RequestUploadModel,
        file: UploadFile,
        headers: AuthorizationHeader,
    ) -> str:
        endpoint = f"{self.model_base_url}/upload"
        response = Requester().post_as_form(
            url=endpoint,
            headers=asdict(headers),
            binary=file.files,
            request_body=asdict(request_body),
        )
        return response.text

    def validate(
        self, request_body: RequestValidateModel, headers: AuthorizationHeader
    ) -> ResponseModelItem:
        endpoint = f"{self.model_base_url}/validate"
        response = Requester().post_as_json(
            url=endpoint, request_body=asdict(request_body), headers=asdict(headers)
        )
        return ResponseModelItem(**response.json())

    def read(
        self,
        request_params: RequestPagination,
        headers: AuthorizationHeader,
        ai_model_id: str,
    ) -> ResponseModelItem:
        endpoint = f"{self.model_base_url}/{ai_model_id}"
        response = Requester().get(
            url=endpoint, params=asdict(request_params), headers=asdict(headers)
        )
        return ResponseModelItem(**response.json())

    def read_all(
        self, request_params: RequestPagination, headers: AuthorizationHeader
    ) -> ResponseModelItems:
        endpoint = f"{self.model_base_url}"
        response = Requester().get(
            url=endpoint, headers=asdict(headers), params=asdict(request_params)
        )
        return ResponseModelItems(**response.json())

    def delete(
        self, headers: AuthorizationHeader, ai_model_id: str
    ) -> ResponseModelItem:
        endpoint = f"{self.model_base_url}/{ai_model_id}"
        response = Requester().delete(url=endpoint, headers=asdict(headers))
        return ResponseModelItem(**response.json())

    def status(
        self, headers: AuthorizationHeader, ai_model_id: str
    ) -> ResponseModelStatus:
        endpoint = f"{self.model_base_url}/{ai_model_id}"
        response = Requester().get(url=endpoint, headers=asdict(headers))
        return ResponseModelStatus(**response.json())
