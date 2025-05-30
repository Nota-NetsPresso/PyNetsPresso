from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class CompressionTaskNotFoundException(PyNPException):
    def __init__(self):
        message = "The compression task does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="COMPRESSION40401",
            name=self.__class__.__name__,
            message=message,
        )


class CompressionTaskIsDeletedException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The compression task with ID '{task_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="COMPRESSION40001",
            name=self.__class__.__name__,
            message=message,
        )
