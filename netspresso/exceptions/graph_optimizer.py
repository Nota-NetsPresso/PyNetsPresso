from netspresso.exceptions.common import AdditionalData, PyNPException


class LoadONNXModelException(PyNPException):
    def __init__(self, error_log: str):
        message = "This onnx model is not supported.\nPlease try again with a PyTorch or TensorFlow model."
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class UpdateOnnxException(PyNPException):
    def __init__(self, error_log: str):
        message = "Failed to update onnx model.\n\nPlease try again with a PyTorch or TensorFlow model."
        super().__init__(
            data=AdditionalData(origin="pynp", error_log=error_log),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
