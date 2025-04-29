from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class EvaluationTaskNotFoundException(PyNPException):
    def __init__(self):
        message = "The evaluation task does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="EVALUATION40401",
            name=self.__class__.__name__,
            message=message,
        )


class EvaluationTaskIsDeletedException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The evaluation task with ID '{task_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="EVALUATION40001",
            name=self.__class__.__name__,
            message=message,
        )
