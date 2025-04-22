from typing import List

from netspresso.exceptions.common import AdditionalData, PyNPException


class NotSupportedTaskException(PyNPException):
    def __init__(self, available_tasks: List, task: int):
        message = f"The task supports {available_tasks}. The entered task is {task}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSetDatasetException(PyNPException):
    def __init__(self):
        message = "The dataset is not set. Use `set_dataset_config` or `set_dataset_config_with_yaml` to set the dataset configuration."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSetModelException(PyNPException):
    def __init__(self):
        message = "The model is not set. Use `set_model_config` or `set_model_config_with_yaml` to set the model configuration."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class TaskOrYamlPathException(PyNPException):
    def __init__(self):
        message = "Either 'task' or 'yaml_path' must be provided, but not both."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotSupportedModelException(PyNPException):
    def __init__(self, available_models: List, model_name: str, task: str):
        message = f"The '{model_name}' model is not supported for the '{task}' task. The available models are {available_models}."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class RetrainingFunctionException(PyNPException):
    def __init__(self):
        message = "This function is intended for retraining. Please use 'set_model_config' for model setup."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class FileNotFoundErrorException(PyNPException):
    def __init__(self, relative_path: str):
        message = f"The required file '{relative_path}' does not exist. Please check and make sure it is in the correct location."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class DirectoryNotFoundException(PyNPException):
    def __init__(self, relative_path: str):
        message = f"The required directory '{relative_path}' does not exist. Please check and make sure it is in the correct location."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class BaseDirectoryNotFoundException(PyNPException):
    def __init__(self, base_path: str):
        message = f"The directory '{base_path}' does not exist."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class FailedTrainingException(PyNPException):
    def __init__(self, error_log: str):
        message = "An error occurred during the training process."
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
