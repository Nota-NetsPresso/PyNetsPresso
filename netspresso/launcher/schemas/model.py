import json

from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator

from netspresso.launcher.schemas import DataType, DeviceName, TaskStatus, ModelFramework, HardwareType


class InputShape(BaseModel):
    """Represents the shape of an input tensor.

    Attributes:
        batch (int): The batch size of the input tensor.
        channel (int): The number of channels in the input tensor.
        dimension (str): The dimensions of the input tensor. ex: height = 416, width = 640, [416, 640].
    """

    batch: Optional[int] = Field(default=1)
    channel: int
    input_size: str

    @validator("batch", always=True, allow_reuse=True)
    def set_batch(cls, v, values, **kwargs):
        return v if v is not None else 1


class TargetDevice(BaseModel):
    """Represents the target device.

    Attributes:
        display_name (str): devices's display name.
        display_brand_name (str): the manufacturer name of the device.
        device_name (str): internal devcie name (using as an internal identifier)
        software_version (Optional[str]): software version of the device (ex: jetpack 5.0.1)
        software_version_display_name (Optional[str]): display name of the devcie's software version
    """

    display_name: str = Field(default=None)
    display_brand_name: str = Field(default=None)
    device_name: DeviceName = Field(default=None)
    software_version: Optional[str] = Field(default=None)
    software_version_display_name: Optional[str] = Field(default=None)
    hardware_type: Optional[HardwareType] = Field(default=None)


class Model(BaseModel):
    """Represents the launcher model object.

    Attributes:
        framework (ModelFramework): the framework of the model.
        filename (str): the file name of the model.
        input_shape (InputShape): the input shape of the model.
        data_type (DataType): the data_type of the model.
        available_devices (List[TargetDevice]): avaliable devices for conversion or benchmark.
        model_uuid (str): the launcher model uuid.
        file_size (float): the size of the model.
    """

    framework: ModelFramework = Field(default=None)
    filename: str = Field(default=None)
    input_shape: InputShape = Field(default=None)
    data_type: DataType = Field(default=None)
    available_devices: List[TargetDevice] = Field(default_factory=list)
    model_uuid: str = Field(default=None)
    file_size: float = Field(default=None)


class BaseRequestModel(BaseModel):
    """Represents the launcher base request object.

    Attributes:
        user_uuid (str): the unique user identifier.
        input_model_uuid (str): the uuid of launcher model object.
        software_version (Optional[str]): target device's software version.
    """

    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    software_version: Optional[str] = Field(default=None)


class ModelBenchmarkRequest(BaseRequestModel):
    """Represents the launcher benchmark request object.

    Attributes:
        target_device (DeviceName): the device name.
        target_framework (Optional[ModelFramework]): the target framework of the model.
        data_type: (Optional[DataType]): data type of the model.
        input_shape: (Optional[InputShape]): input shape of the model.
    """

    target_device: DeviceName = Field(default=None)
    target_framework: Optional[ModelFramework] = Field(default=None)
    data_type: Optional[DataType] = Field(default=None)
    input_shape: Optional[InputShape] = Field(default=None)
    hardware_type: Optional[HardwareType] = Field(default=None)


class ModelConversionRequest(BaseRequestModel):
    """Represents the launcher conversion request object.

    Attributes:
        target_device (DeviceName): the device name.
        target_framework (ModelFramework): the target framework of the model.
        data_type: (DataType): data type of the model.
        input_shape: (InputShape): input shape of the model.
    """

    target_device_name: DeviceName = Field(default=None)
    target_framework: ModelFramework = Field(default=None)
    data_type: DataType = Field(default=DataType.FP16)
    input_shape: InputShape = Field(default=None)

    @validator("input_shape")
    def validate_input_shape(cls, value):
        if value:
            return json.dumps(value.dict())
        else:
            return None


class BaseTaskModel(BaseModel):
    """Represents the launcher base task response object.

    Attributes:
        user_uuid (str): the unique user identifier.
        input_model_uuid (str): the uuid of launcher model object.
        status (TaskStatus): current status of the task.
        input_shape: (InputShape): input shape of the model.
        data_type: (DataType): data type of the model.
        software_version (str): target device's software version.
        framework (ModelFramework): the target framework of the model.
    """

    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    status: TaskStatus = Field(default=None)
    input_shape: InputShape = Field(default=None)
    data_type: DataType = Field(default=DataType.FP16)
    input_shape: InputShape = Field(default=None)
    software_version: str = Field(default=None)
    framework: ModelFramework = Field(default=None)


class BenchmarkTask(BaseTaskModel):
    """Represents the launcher benchmark task response object.

    Attributes:
        benchmark_task_uuid (str): the unique benchamrk task identifier.
        target_device (DeviceName): the uuid of launcher model object.
        file_name (str): the model file name.
        data_type: (Optional[DataType]): data type of the model
        memory_footprint_gpu (Optional[float]): memory usage of the model while inferencing the model.
        memory_footprint_cpu (Optional[float]): cpu usage of the model while inferencing the model.
        latency (Optional[float]): average inference latency of the model.
        ram_size (Optional[float]): RAM size of the target device.
        processor (Optional[str]): processor information of the target device.
        benchmark_result (Optional[dict]): original benchmark result in dictionary.
    """

    benchmark_task_uuid: str = Field(default=None)
    target_device: DeviceName = Field(default=None)
    file_name: str = Field(default=None)
    data_type: Optional[DataType] = Field(default=DataType.FP16)
    memory_footprint_gpu: Optional[float] = Field(default=None)
    memory_footprint_cpu: Optional[float] = Field(default=None)
    latency: Optional[float] = Field(default=None)
    ram_size: Optional[float] = Field(default=None)
    processor: Optional[str] = Field(default=None)
    benchmark_result: Optional[dict] = Field(default=None)


class ConversionTask(BaseTaskModel):
    """Represents the launcher benchmark task response object.

    Attributes:
        convert_task_uuid (str): the unique convert task identifier.
        output_model_uuid (str): the uuid of converted model.
        model_file_name (str): the model file name.
        target_device_name (DeviceName): the device name.
    """

    convert_task_uuid: str = Field(default=None)
    output_model_uuid: str = Field(default=None)
    model_file_name: str = Field(default=None)
    target_device_name: DeviceName = Field(default=None)
