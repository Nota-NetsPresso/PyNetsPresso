from fastapi import status
from fastapi.responses import JSONResponse

from netspresso.exceptions.common import AdditionalData, PyNPException


class PyNPHTTPException(PyNPException):
    """
    HTTP 상태 코드를 포함하는 PyNPException 확장 클래스
    """
    def __init__(
        self,
        data: AdditionalData,
        error_code: str,
        name: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(
            data=data,
            error_code=error_code,
            name=name,
            message=message,
        )
        self.status_code = status_code

    def to_response(self):
        """
        FastAPI 응답 객체로 변환
        """
        return JSONResponse(
            status_code=self.status_code,
            content=self.detail
        )


class BadRequestHTTPException(PyNPHTTPException):
    def __init__(
        self,
        message: str,
        error_log: str = "",
        error_code: str = "",
    ):
        super().__init__(
            data=AdditionalData(
                origin="pynp",
                error_log=error_log,
            ),
            error_code=error_code,
            name=self.__class__.__name__,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class NotFoundHTTPException(PyNPHTTPException):
    def __init__(
        self,
        message: str,
        error_log: str = "",
        error_code: str = "",
    ):
        super().__init__(
            data=AdditionalData(
                origin="pynp",
                error_log=error_log,
            ),
            error_code=error_code,
            name=self.__class__.__name__,
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
        )


class UnauthorizedHTTPException(PyNPHTTPException):
    def __init__(
        self,
        message: str,
        error_log: str = "",
        error_code: str = "",
    ):
        super().__init__(
            data=AdditionalData(
                origin="pynp",
                error_log=error_log,
            ),
            error_code=error_code,
            name=self.__class__.__name__,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
