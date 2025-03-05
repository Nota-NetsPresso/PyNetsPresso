from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class ProjectNameTooLongException(PyNPException):
    def __init__(self, max_length: int, actual_length: int):
        message = f"The project_name exceeds maximum length. Max: {max_length}, Actual: {actual_length}"
        super().__init__(
            data=AdditionalData(origin=Origin.CORE),
            error_code="PROJECT40001",
            name=self.__class__.__name__,
            message=message,
        )


class ProjectAlreadyExistsException(PyNPException):
    def __init__(self, project_name: str, project_path: str):
        message = f"The project_name '{project_name}' already exists at '{project_path}'."
        super().__init__(
            data=AdditionalData(origin=Origin.CORE),
            error_code="PROJECT40901",
            name=self.__class__.__name__,
            message=message,
        )


class ProjectSaveException(PyNPException):
    def __init__(self, error: Exception, project_name: str):
        message = f"Failed to save project '{project_name}' to the database: {str(error)}"
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="PROJECT50001",
            name=self.__class__.__name__,
            message=message,
        )


class ProjectNotFoundException(PyNPException):
    def __init__(self):
        message = "The project does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="PROJECT40401",
            name=self.__class__.__name__,
            message=message,
        )


class ProjectIsDeletedException(PyNPException):
    def __init__(self, project_id: str):
        message = f"The project with ID '{project_id}' has been deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="PROJECT40002",
            name=self.__class__.__name__,
            message=message,
        )
