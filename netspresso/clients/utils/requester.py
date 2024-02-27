from typing import Optional

import requests
from requests import Response


class Requester:
    @staticmethod
    def __make_response(response: Response) -> Response:
        if response.ok:
            return response
        else:
            raise Exception(response.json())

    @staticmethod
    def get(url: str, params: Optional[dict] = None, headers=None, **kwargs) -> Response:
        response = requests.get(url, headers=headers, params=params, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def post_as_form(url: str, request_body: Optional[dict] = None, binary=None, headers=None, **kwargs) -> Response:
        response = requests.post(url, headers=headers, data=request_body, files=binary, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def post_as_json(url: str, request_body: dict = None, headers=None, **kwargs) -> Response:
        response = requests.post(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def put(url: str, request_body: dict, headers=None, **kwargs) -> Response:
        response = requests.put(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def patch(url: str, request_body: dict, headers=None, **kwargs) -> Response:
        response = requests.patch(url, headers=headers, json=request_body, **kwargs)

        return Requester.__make_response(response=response)

    @staticmethod
    def delete(url: str, headers=None, **kwargs) -> Response:
        response = requests.delete(url, headers=headers, **kwargs)

        return Requester.__make_response(response=response)
