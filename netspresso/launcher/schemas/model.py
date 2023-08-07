from typing import Optional
from pydantic import BaseModel, Field, validator

from netspresso.launcher.schemas import DataType, DeviceName, TaskStatus, ModelFramework

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

    @validator("batch", always=True)
    def set_batch(cls, v, values, **kwargs):
        """Set the eggs field based upon a spam value."""
        return v if v is not None else 1

class TargetDevice(BaseModel):
    display_name: str = Field(default=None)
    display_brand_name: str = Field(default=None)
    device_name: DeviceName = Field(default=None)
    software_version: Optional[str] = Field(default=None)
    software_version_display_name: Optional[str] = Field(default=None)

class Model(BaseModel):
    framework: ModelFramework = Field(default=None)
    filename: str = Field(default=None)
    input_shape: InputShape = Field(default=None)
    data_type: DataType = Field(default=None)
    available_devices: list[TargetDevice] = Field(default_factory=list)
    model_uuid: str = Field(default=None)
    file_size: float = Field(default=None)

class BaseRequestModel(BaseModel):
    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    software_version: Optional[str] = Field(default=None)

class ModelBenchmarkRequest(BaseRequestModel):
    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    target_device: DeviceName = Field(default=None)
    target_framework: Optional[ModelFramework] = Field(default=None)
    data_type: Optional[DataType] = Field(default=None)
    input_shape: Optional[InputShape] = Field(default=None)

class ModelConversionRequest(BaseRequestModel):
    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    target_device_name: DeviceName = Field(default=None)
    target_framework: ModelFramework = Field(default=None)
    data_type: DataType = Field(default=DataType.FP16)
    input_shape: InputShape = Field(default=None)

class BaseTaskModel(BaseModel):
    user_uuid: str = Field(default=None)
    input_model_uuid: str = Field(default=None)
    status: TaskStatus = Field(default=None)
    input_shape: InputShape = Field(default=None)
    data_type: DataType = Field(default=DataType.FP16)
    input_shape: InputShape = Field(default=None)
    software_version: str = Field(default=None)
    framework: ModelFramework = Field(default=None)

class BenchmarkTask(BaseTaskModel):
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
    convert_task_uuid: str = Field(default=None)
    output_model_uuid: str = Field(default=None)
    model_file_name: str = Field(default=None)
    target_device_name: DeviceName = Field(default=None)
    
    

