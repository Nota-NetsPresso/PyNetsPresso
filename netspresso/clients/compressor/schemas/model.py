import json
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field, HttpUrl, root_validator, validator

from netspresso.enums.model import Extension, Framework, compressor_framework_literal, originfrom_literal


class InputLayer(BaseModel):
    batch: Any = Field(1, description="Input Batch")
    channel: int = Field(0, description="Input Channel")
    dimension: List[int] = Field([0], description="Input Diemension")


class UploadModelRequest(BaseModel):
    model_name: str = Field(..., description="Model Name")
    description: str = Field("", description="Description")
    task: str = Field("other", description="Task")
    framework: compressor_framework_literal = Field(..., description="Framework")
    input_layers: List[InputLayer] = Field(None, description="Input Layers")
    file_path: str = Field(..., description="Model Path")
    # metric_unit: str = Field("", description="Metric Unit")
    # metric_value: float = Field(0, description="Metric Value")

    @validator("input_layers")
    def validate_input_layers(cls, value):
        if value:
            input_layers = []

            if len(value) != 1:
                raise Exception("Currently, only single input models are supported.")

            for v in value:
                if any((v.batch == 0, v.channel == 0, v.dimension == [0])):
                    return None
                input_layers.append(v.dict())
            return json.dumps(input_layers)
        else:
            return None

    @root_validator(pre=True, skip_on_failure=True)
    def validate_request_params(cls, values):
        framework = values.get("framework")
        input_layers = values.get("input_layers")
        file_path = Path(values.get("file_path"))
        file_extension = file_path.suffix[1:]

        if framework not in [
            Framework.TENSORFLOW_KERAS,
            Framework.PYTORCH,
            Framework.ONNX,
        ]:
            raise Exception("Invalid framework. Supported frameworks are TensorFlow/Keras, PyTorch, and ONNX.")

        if file_extension in [Extension.H5, Extension.ZIP] and not framework == Framework.TENSORFLOW_KERAS:
            raise Exception(
                "Invalid model framework. Models with .h5 or .zip extensions must use TensorFlow/Keras framework."
            )
        elif file_extension == Extension.PT and not framework == Framework.PYTORCH:
            raise Exception("Invalid model framework. Models with .pt extensions must use PyTorch framework.")
        elif file_extension == Extension.ONNX and not framework == Framework.ONNX:
            raise Exception("Invalid model framework. Models with .onnx extensions must use ONNX framework.")

        if framework == Framework.PYTORCH and input_layers is None:
            raise Exception("Invalid input shape. Input shape is required for PyTorch models.")

        return values


class Spec(BaseModel):
    input_layers: List[InputLayer] = Field([], description="Input Layers")
    model_size: float = Field(0, description="Model Size")
    flops: float = Field(0, description="FLOPs")
    trainable_parameters: float = Field(0, description="Trainable Parameters")
    non_trainable_parameters: float = Field(0, description="Non Trainable Parameters")
    number_of_layers: int = Field(0, description="Number of Layers")


class Status(BaseModel):
    is_convertible: bool = Field(False, description="Convertible Status")
    is_packageable: bool = Field(False, description="Packageable Status")
    is_compressible: bool = Field(False, description="Compressible Status")
    is_visible: bool = Field(False, description="Visible Status")
    is_compressed: bool = Field(False, description="Compressed Status")
    is_trained: bool = Field(False, description="Trained Status")
    is_downloadable: bool = Field(False, description="Downloadable Status")
    is_retrainable: bool = Field(False, description="Retrainable Status")


class Metric(BaseModel):
    metric_unit: str = Field("", description="Metric Unit")
    metric_value: float = Field(None, description="Metric Value")


class Device(BaseModel):
    name: str = Field("", description="Device Name")
    total_latency: float = Field(0, description="Total Latency of Model")
    performance: List = Field([], description="Metric Value")
    spec: List = Field([], description="Metric Value")
    layers: List = Field([], description="Metric Value")


class ModelResponse(BaseModel):
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model Name")
    description: str = Field("", description="Description")
    original_model_id: str = Field(..., description="Original Model ID")
    original_compression_id: str = Field("", description="Compression ID")
    task: str = Field(..., description="Task")
    framework: compressor_framework_literal = Field(..., description="Framework")
    origin_from: originfrom_literal = Field(..., description="Origin From(Model Source)")
    target_device: str = Field("", description="Target Device")
    metric: Metric = Field(..., description="Metric")
    spec: Spec = Field(..., description="Spec")
    status: Status = Field(..., description="Status")
    devices: List[Device] = Field([], description="Devices")
    # edges: List = Field([], description="Edges")
    # nodes: List = Field([], description="Nodes")


class GetDownloadLinkResponse(BaseModel):
    url: HttpUrl = Field(..., description="Model Path")
