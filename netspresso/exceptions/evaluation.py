from netspresso.enums import Status
from netspresso.enums.conversion import TargetFramework
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


class EvaluationTaskAlreadyExistsException(PyNPException):
    def __init__(self, task_id: str, status: Status):
        if status == Status.COMPLETED:
            message = f"Evaluation with task ID '{task_id}' has already been completed. Please use the existing results."
        elif status == Status.IN_PROGRESS:
            message = f"Evaluation with task ID '{task_id}' is currently in progress. Please wait for completion."
        else:
            message = f"Evaluation with task ID '{task_id}' already exists with status '{status}'."

        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="EVALUATION40900",
            name=self.__class__.__name__,
            message=message,
        )


class UnsupportedEvaluationFrameworkException(PyNPException):
    def __init__(self, framework: TargetFramework):
        message = f"The framework '{framework}' is not supported for evaluation."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="EVALUATION40002",
            name=self.__class__.__name__,
            message=message,
        )


class EvaluationResultFileNotFoundException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The evaluation result file for task '{task_id}' does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, task_id=task_id),
            error_code="EVALUATION40404",
            name=self.__class__.__name__,
            message=message,
        )


class EvaluationDownloadURLGenerationException(PyNPException):
    def __init__(self, task_id: str, error_details: str):
        message = f"Failed to generate download URL for evaluation task '{task_id}': {error_details}"
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY, task_id=task_id, error_details=error_details),
            error_code="EVALUATION50000",
            name=self.__class__.__name__,
            message=message,
        )
