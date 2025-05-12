from typing import Optional

from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData


class EvaluationTaskAlreadyExistsException(ExceptionBase):
    def __init__(
        self,
        task_id: str,
        task_status: str,
        data: Optional[AdditionalData] = None,
    ):
        super().__init__(
            data=data,
            error_code="EVALUATION40001",
            status_code=status.HTTP_400_BAD_REQUEST,
            name="EvaluationTaskAlreadyExistsException",
            message=f"Evaluation task already exists with ID {task_id} (status: {task_status})."
        )
