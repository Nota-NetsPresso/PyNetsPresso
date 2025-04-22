from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class ConversionTaskNotFoundException(PyNPException):
    def __init__(self):
        message = "The conversion task does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="CONVERSION40401",
            name=self.__class__.__name__,
            message=message,
        )


class ConversionTaskIsDeletedException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The conversion task with ID '{task_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="CONVERSION40001",
            name=self.__class__.__name__,
            message=message,
        )
