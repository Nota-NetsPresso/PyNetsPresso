import json
import requests

from netspresso.schemas.auth import (
    CreditResponse,
    LoginResponse,
    UserResponse,
    RefreshTokenResponse,
)
from netspresso.compressor.client.schemas.compression import (
    CompressionResponse,
    GetAvailableLayersReponse,
    RecommendationResponse,
)
from netspresso.compressor.client.schemas.model import UploadModelRequest, ModelResponse, GetDownloadLinkResponse
from netspresso.client.utils.common import get_files, get_headers
from netspresso.compressor.client.utils.enum import (
    Task,
    Framework,
    Extension,
    OriginFrom,
    CompressionMethod,
    RecommendationMethod,
    Policy,
    LayerNorm,
    GroupPolicy,
)  # noqa
from netspresso.client.config import Config, EndPoint

class ModelCompressorAPIClient:
    def __init__(self):
        self.config = Config(EndPoint.COMPRESSOR)
        self.host = self.config.HOST
        self.port = self.config.PORT
        self.prefix = self.config.URI_PREFIX
        self.url = f"{self.host}:{self.port}{self.prefix}"

    def upload_model(self, data: UploadModelRequest, access_token) -> ModelResponse:
        url = f"{self.url}/models"
        files = get_files(data.file_path)
        response = requests.post(url, data=data.dict(), files=files, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return ModelResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_parent_models(self, is_simple, access_token):
        url = f"{self.url}/models/parents?is_simple={is_simple}"
        response = requests.get(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return [ModelResponse(**r) for r in response_body]
        else:
            raise Exception(response_body["detail"])

    def get_children_models(self, model_id, access_token):
        url = f"{self.url}/models/{model_id}/children"
        response = requests.get(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return [ModelResponse(**r) for r in response_body]
        else:
            raise Exception(response_body["detail"])

    def get_model_info(self, model_id, access_token) -> ModelResponse:
        url = f"{self.url}/models/{model_id}"
        response = requests.get(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return ModelResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_download_model_link(self, model_id, access_token) -> GetDownloadLinkResponse:
        url = f"{self.url}/models/{model_id}/download"
        response = requests.post(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return GetDownloadLinkResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def delete_model(self, model_id, access_token):
        url = f"{self.url}/models/{model_id}"
        response = requests.delete(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return response_body
        else:
            raise Exception(response_body["detail"])

    def get_available_layers(self, data, access_token) -> GetAvailableLayersReponse:
        url = f"{self.url}/models/{data.model_id}/get_available_layers"
        response = requests.post(url, data=data.json(), headers=get_headers(access_token, json_type=True))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return GetAvailableLayersReponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def create_compression(self, data, access_token) -> CompressionResponse:
        url = f"{self.url}/compressions"
        response = requests.post(url, data=data.json(), headers=get_headers(access_token, json_type=True))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return CompressionResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def get_recommendation(self, data, access_token) -> RecommendationResponse:
        url = f"{self.url}/models/{data.model_id}/recommendation"
        response = requests.post(url, data=data.json(), headers=get_headers(access_token, json_type=True))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            response_body = {"recommended_layers": response_body}
            return RecommendationResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def compress_model(self, data, access_token):
        url = f"{self.url}/compressions/{data.compression_id}"
        response = requests.put(url, data=data.json(), headers=get_headers(access_token, json_type=True))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return CompressionResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def auto_compression(self, data, access_token):
        url = f"{self.url}/models/{data.model_id}/auto_compress"
        response = requests.post(url, data=data.json(), headers=get_headers(access_token, json_type=True))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            response = ModelResponse(**response_body)
            return response
        else:
            raise Exception(response_body["detail"])

    def get_compression_info(self, compression_id, access_token):
        url = f"{self.url}/compressions/{compression_id}"

        response = requests.get(url, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return CompressionResponse(**response_body)
        else:
            raise Exception(response_body["detail"])

    def upload_dataset(self, data, access_token):
        url = f"{self.url}/models/{data.model_id}/datasets"
        files = get_files(data.file_path)
        response = requests.post(url, files=files, headers=get_headers(access_token))
        response_body = json.loads(response.text)

        if response.status_code == 200:
            return response_body
        else:
            raise Exception(response_body["detail"])
