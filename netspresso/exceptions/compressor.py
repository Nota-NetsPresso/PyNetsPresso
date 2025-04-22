from netspresso.exceptions.common import AdditionalData, PyNPException


class NotValidChannelAxisRangeException(PyNPException):
    def __init__(self, reshape_channel_axis: int):
        message = f"The reshape_channel_axis value is in the range [0, 1, -1, -2], but got {reshape_channel_axis}"
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class EmptyCompressionParamsException(PyNPException):
    def __init__(self):
        message = "The available_layer.values all empty. please put in the available_layer.values to compress."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotValidSlampRatioException(PyNPException):
    def __init__(self, ratio: float):
        message = (f"The ratio range for SLAMP is 0 < ratio < 1, but got `{ratio}`.",)
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotValidVbmfRatioException(PyNPException):
    def __init__(self, ratio: float):
        message = f"The ratio range for VBMF is -1 <= ratio <= 1, but got `{ratio}`."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class NotFillInputLayersException(PyNPException):
    def __init__(self):
        message = "Please fill in Input Layers fields."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )


class FailedUploadModelException(PyNPException):
    def __init__(self):
        message = "Failed to upload model."
        super().__init__(
            data=AdditionalData(origin="pynp"),
            error_code="",
            name=self.__class__.__name__,
            message=message,
        )
