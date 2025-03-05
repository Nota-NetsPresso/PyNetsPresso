from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class UserNotFoundException(PyNPException):
    def __init__(self):
        message = "The user does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="USER40401",
            name=self.__class__.__name__,
            message=message,
        )


class UserIsDeletedException(PyNPException):
    def __init__(self, user_id: str):
        message = f"The user with ID '{user_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="USER40001",
            name=self.__class__.__name__,
            message=message,
        )
