from netspresso.exceptions.common import AdditionalData, Origin, PyNPException


class BenchmarkTaskNotFoundException(PyNPException):
    def __init__(self):
        message = "The benchmark task does not exist."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="BENCHMARK40401",
            name=self.__class__.__name__,
            message=message,
        )


class BenchmarkTaskIsDeletedException(PyNPException):
    def __init__(self, task_id: str):
        message = f"The benchmark task with ID '{task_id}' has been already deleted."
        super().__init__(
            data=AdditionalData(origin=Origin.REPOSITORY),
            error_code="BENCHMARK40001",
            name=self.__class__.__name__,
            message=message,
        )
