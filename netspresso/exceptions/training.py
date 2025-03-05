from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class TrainingTaskNotFoundException(PyNPException):
    def __init__(self):
        message = "The training task does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="TRAINING40401",
            name=self.__class__.__name__,
            message=message,
        )


class TrainingTaskIsDeletedException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The training task with ID '{task_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="TRAINING40001",
            name=self.__class__.__name__,
            message=message,
        )
