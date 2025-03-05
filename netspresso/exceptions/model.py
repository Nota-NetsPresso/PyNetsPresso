from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class ModelNotFoundException(PyNPException):
    def __init__(self):
        message = "The model does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="MODEL40401",
            name=self.__class__.__name__,
            message=message,
        )


class ModelIsDeletedException(PyNPException):
    def __init__(self, model_id: str):
        message = f"The model with ID '{model_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="MODEL40001",
            name=self.__class__.__name__,
            message=message,
        )


class ModelCannotBeDeletedException(PyNPException):
    def __init__(self, model_id: str):
        message = f"The model with ID '{model_id}' cannot be deleted. Only trained and compressed models can be deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="MODEL40002",
            name=self.__class__.__name__,
            message=message,
        )
