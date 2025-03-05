from netspresso.exceptions.common import AdditionalData, PyNPException


class DatabaseOperationError(PyNPException):
    """Base exception for database operations"""

    def __init__(self, error_log: str = "", error_code: str = "DB_OPERATION_ERROR"):
        message = "Database operation failed"
        super().__init__(
            data=AdditionalData(origin="database", error_log=error_log),
            error_code=error_code,
            name=self.__class__.__name__,
            message=message,
        )
