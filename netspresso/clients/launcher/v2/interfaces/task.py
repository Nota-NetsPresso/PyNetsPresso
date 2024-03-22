from abc import ABC, abstractmethod
import dataclasses

from netspresso.clients.launcher.v2.schemas import ResponseItem, ResponseItems


class TaskInterface(ABC):
    @abstractmethod
    def start(
        self, request_body: dataclasses, headers: dataclasses, file: dataclasses
    ) -> ResponseItem:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self, headers: dataclasses, task_id: str) -> ResponseItem:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, headers: dataclasses, task_id: str) -> ResponseItem:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def status(self, headers: dataclasses, task_id: str) -> ResponseItem:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, headers: dataclasses, task_id: str) -> ResponseItem:
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def options(self, headers: dataclasses) -> ResponseItems:
        """

        :return:
        """
        raise NotImplementedError
