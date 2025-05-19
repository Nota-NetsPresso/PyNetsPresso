from dataclasses import dataclass, field
from typing import List, Optional

from netspresso.metadata.common import InputShape, ModelInfo


@dataclass
class InputLayer:
    name: Optional[str] = None
    batch: Optional[int] = None
    channel: Optional[int] = None
    dimension: Optional[list] = None

    def __post_init__(self):
        if self.batch is None:
            self.batch = 1

    def to(self) -> InputShape:
        input_shape = InputShape()
        input_shape.batch = self.batch
        input_shape.channel = self.channel
        input_shape.dimension = self.dimension
        return input_shape


@dataclass
class ModelDetail:
    data_type: Optional[str] = None
    ai_model_format: Optional[str] = None
    framework: Optional[str] = None
    flops: Optional[float] = 0
    trainable_parameters: Optional[int] = 0
    non_trainable_parameters: Optional[int] = 0
    number_of_layers: Optional[int] = None
    graph_info: Optional[dict] = None
    input_layers: Optional[List[InputLayer]] = field(default_factory=list)

    def __post_init__(self):
        if self.input_layers:
            self.input_layers = [InputLayer(**input_layer) for input_layer in self.input_layers]

        if self.trainable_parameters is None:
            self.trainable_parameters = 0
        if self.non_trainable_parameters is None:
            self.non_trainable_parameters = 0


@dataclass
class ModelStatus:
    is_deleted: Optional[bool] = False
    is_convertible: Optional[bool] = False
    is_quantizable: Optional[bool] = False
    is_compressible: Optional[bool] = False
    is_benchmarkable: Optional[bool] = False
    is_uploaded: Optional[bool] = False

    def __init__(self, **kwargs):
        default_fields = {
            "is_deleted": False,
            "is_convertible": False,
            "is_quantizable": False,
            "is_compressible": False,
            "is_benchmarkable": False,
            "is_uploaded": False,
        }

        for field_name, default_value in default_fields.items():
            setattr(self, field_name, kwargs.get(field_name, default_value))

        for key, value in kwargs.items():
            if key not in default_fields:
                setattr(self, key, value)


@dataclass
class ModelBase:
    user_id: str
    ai_model_id: str
    uploaded_file_name: Optional[str]
    file_size_in_mb: Optional[float]
    md5_checksum: Optional[str]
    bucket_name: Optional[str]
    object_path: Optional[str]
    task: str
    display_name: Optional[str]
    error_log: Optional[str]
    status: Optional[ModelStatus]
    detail: Optional[ModelDetail]

    def __post_init__(self):
        self.status = ModelStatus(**self.status)
        self.detail = ModelDetail(**self.detail)

    def to(self) -> ModelInfo:
        model_info = ModelInfo()
        model_info.data_type = self.detail.data_type
        model_info.framework = self.detail.framework
        model_info.input_shapes = [
            InputShape(batch=input_layer.batch, channel=input_layer.channel, dimension=input_layer.dimension)
            for input_layer in self.detail.input_layers
        ]
        return model_info
