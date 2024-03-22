from dataclasses import dataclass
from typing import Optional


@dataclass
class InputLayer:
    name: Optional[str] = None
    batch: Optional[int] = None
    channel: Optional[int] = None
    dimension: Optional[list] = None


@dataclass
class ModelDetail:
    data_type: Optional[str] = None
    ai_model_format: Optional[str] = None
    framework: Optional[str] = None
    flops: Optional[float] = None
    non_trainable_parameter: Optional[int] = None
    trainable_parameter: Optional[int] = None
    number_of_layers: Optional[int] = None
    graph_info: Optional[dict] = None
    input_layer: Optional[InputLayer] = None


@dataclass
class ModelStatus:
    is_deleted: Optional[bool] = False
    is_convertible: Optional[bool] = False
    is_compressible: Optional[bool] = False
    is_benchmarkable: Optional[bool] = False
    is_uploaded: Optional[bool] = False


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
