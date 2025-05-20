from dataclasses import asdict, dataclass, field
from typing import List, Optional

from netspresso.enums.base import StrEnum


class LinkType(StrEnum):
    DOCS = "docs"
    CONTACT = "contact"


@dataclass
class LinkInfo:
    type: LinkType = field(metadata={"description": "Link type"})
    value: str = field(metadata={"description": "Link value"})


@dataclass
class AdditionalData:
    origin: Optional[str] = field(default="pynp", metadata={"description": "Error origin"})
    error_log: Optional[str] = field(default="", metadata={"description": "Error log"})
    link: Optional[LinkInfo] = field(default=None, metadata={"description": "Link info"})


@dataclass
class ExceptionDetail:
    data: Optional[AdditionalData] = field(default_factory=AdditionalData, metadata={"description": "Additional data"})
    error_code: str = field(default="", metadata={"description": "Error code"})
    name: str = field(default="", metadata={"description": "Error name"})
    message: str = field(default="", metadata={"description": "Error message"})


class PyNPException(Exception):
    def __init__(
        self,
        data: Optional[AdditionalData],
        error_code: str,
        name: str,
        message: str,
    ):
        detail = ExceptionDetail(
            data=data,
            error_code=error_code,
            name=name,
            message=message,
        )
        detail_dict = asdict(detail)
        super().__init__(detail_dict)
        self.detail = detail_dict


class FailedFetchPackageException(PyNPException):
    def __init__(self, package_name: str, error_log: str):
        message = f"Failed to fetch {package_name} from PyPI"
        super().__init__(
            data=AdditionalData(
                origin="pynp",
                error_log=error_log,
            ),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )

class NotEnoughCreditException(PyNPException):
    def __init__(self, current_credit: int, service_credit: int, service_task_name: str):
        error_log = (
            f"Your current balance of {current_credit} credits is insufficient to complete the task. \n"
            f"{service_credit} credits are required for one {service_task_name} task. \n"
            f"For additional credit, please contact us at netspresso@nota.ai."
        )
        message = "Not enough credits. Please check your credit."
        super().__init__(
            data=AdditionalData(
                origin="pynp",
                error_log=error_log,
            ),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSupportedFrameworkException(PyNPException):
    def __init__(self, available_frameworks: List, framework: int):
        message = f"The framework supports {available_frameworks}. The entered framework is {framework}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotValidInputModelPath(PyNPException):
    def __init__(self):
        message = "The input_model_path should be a file and cannot be a directory. Ex) ./model/sample_model.pt"
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class GatewayTimeoutException(PyNPException):
    def __init__(self, error_log, status_code):
        message = f"504 Gateway Timeout: The server did not receive a timely response with status code {status_code}"
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class UnexpetedException(PyNPException):
    def __init__(self, error_log, status_code):
        message = f"Unexpected error occurred with status code {status_code}"
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class InternalServerErrorException(PyNPException):
    def __init__(self, error_log, status_code):
        message = f"Internal server error occurred with status code {status_code}"
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
