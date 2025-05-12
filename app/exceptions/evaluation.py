from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData, Origin


class EvaluationTaskAlreadyExistsException(ExceptionBase):
    def __init__(self, task_id: str, task_status: str):
        super().__init__(
            data=AdditionalData(origin=Origin.SERVICE),
            error_code="EVALUATION40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=f"Evaluation task already exists with ID {task_id} (status: {task_status})."
        )
