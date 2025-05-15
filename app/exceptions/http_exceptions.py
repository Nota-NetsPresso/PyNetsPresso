from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData, Origin


class BadRequestException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code,
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )


class NotFoundException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code,
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )


class UnauthorizedException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code, 
            status_code=status.HTTP_401_UNAUTHORIZED,
            name=self.__class__.__name__,
            message=message,
        )


class ForbiddenException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code,
            status_code=status.HTTP_403_FORBIDDEN,
            name=self.__class__.__name__,
            message=message,
        )


class ConflictException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code,
            status_code=status.HTTP_409_CONFLICT,
            name=self.__class__.__name__,
            message=message,
        )


class InternalServerErrorException(ExceptionBase):
    def __init__(self, message: str, error_code: str = "", error_log: str = ""):
        data = AdditionalData(origin=Origin.ROUTER, error_log=error_log)
        super().__init__(
            data=data,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            name=self.__class__.__name__,
            message=message,
        ) 