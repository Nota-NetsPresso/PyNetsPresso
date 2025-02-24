from fastapi import status

from app.exceptions.base import ExceptionBase
from app.exceptions.schema import AdditionalData, Origin


class ModelNotFoundException(ExceptionBase):
    def __init__(self, origin: Origin = Origin.REPOSITORY):
        message = "The model does not exist."
        super().__init__(
            data=AdditionalData(origin=origin),
            error_code="MODEL40401",
            status_code=status.HTTP_404_NOT_FOUND,
            name=self.__class__.__name__,
            message=message,
        )


class ModelIsDeletedException(ExceptionBase):
    def __init__(self, model_id: str, origin: Origin = Origin.REPOSITORY):
        message = f"The model with ID '{model_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=origin),
            error_code="MODEL40002",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )

class ModelCannotBeDeletedException(ExceptionBase):
    def __init__(self, model_id: str, origin: Origin = Origin.REPOSITORY):
        message = f"The model with ID '{model_id}' cannot be deleted. Only trained and compressed models can be deleted."
        super().__init__(
            data=AdditionalData(origin=origin),
            error_code="MODEL40003",
            status_code=status.HTTP_400_BAD_REQUEST,
            name=self.__class__.__name__,
            message=message,
        )
