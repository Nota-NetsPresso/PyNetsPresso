import dataclasses
from abc import ABC, abstractmethod

from netspresso.clients.launcher.v2.schemas import ResponseItem, ResponseItems


class ModelInterface(ABC):
    @abstractmethod
    def get_upload_url(self, request_params: dataclasses, headers: dataclasses) -> ResponseItem:
        """
        :param request_params:
        :param headers:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_download_url(self, headers: dataclasses, ai_model_id: str) -> str:
        """
        :param headers:
        :param ai_model_id:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def upload(self, request_body: dataclasses, file: dataclasses, headers: dataclasses) -> str:
        """
        :param request_body:
        :param file:
        :param headers:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, request_params: dataclasses, headers: dataclasses, ai_model_id: str) -> ResponseItem:
        """
        :param request_params:
        :param headers:
        :param ai_model_id:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def read_all(self, request_params: dataclasses, headers: dataclasses) -> ResponseItems:
        """
        :param request_params:
        :param headers:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, headers: dataclasses, ai_model_id: str) -> ResponseItem:
        """
        :param headers:
        :param ai_model_id:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, request_body: dataclasses, headers: dataclasses) -> ResponseItem:
        """
        :param request_body:
        :param headers:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def status(self, headers: dataclasses, ai_model_id: str) -> ResponseItem:
        """
        :param headers:
        :param ai_model_id:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def options(self, headers: dataclasses, ai_model_id: str) -> ResponseItems:
        """
        :param headers:
        :param ai_model_id:
        :return:
        """
        raise NotImplementedError
