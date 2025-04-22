import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Union

from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseItem, ResponseItems
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import QuantizationMode, QuantizationPrecision, SimilarityMetric, TaskStatusForDisplay
from netspresso.metadata.quantizer import QuantizeInfo


@dataclass
class QuantizeOption:
    threshold: Union[float, int]
    metric: SimilarityMetric = field(default=SimilarityMetric.SNR)
    weight_precision: QuantizationPrecision = field(default=QuantizationPrecision.INT8)
    activation_precision: QuantizationPrecision = field(default=QuantizationPrecision.INT8)


@dataclass
class QuantizeTask:
    quantize_task_id: str
    input_model_id: str
    output_model_id: str

    quantization_mode: QuantizationMode = field(default=QuantizationMode.UNIFORM_PRECISION_QUANTIZATION)
    options: Optional[QuantizeOption] = field(default_factory=QuantizeOption)
    task_input_layers: List[InputLayer] = field(default_factory=[])

    status: TaskStatusForDisplay = ""
    error_log: Optional[dict] = None

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.task_input_layers = [InputLayer(**task_input_layers) for task_input_layers in self.task_input_layers]
        self.options = QuantizeOption(**self.options)

    def to(self, model_file_name: str) -> QuantizeInfo:
        quantize_info = QuantizeInfo()
        quantize_info.quantize_task_uuid = self.quantize_task_id
        quantize_info.input_model_uuid = self.input_model_id
        quantize_info.output_model_uuid = self.output_model_id
        quantize_info.model_file_name = model_file_name
        quantize_info.quantization_mode = self.quantization_mode
        quantize_info.metric = self.options.metric
        quantize_info.threshold = self.options.threshold
        quantize_info.weight_precision = self.options.weight_precision
        quantize_info.activation_precision = self.options.activation_precision

        return quantize_info


@dataclass
class ResponseQuantizeTaskItem(ResponseItem):
    data: Optional[QuantizeTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = QuantizeTask(**self.data)


@dataclass
class ResponseQuantizeOptionItems(ResponseItems):
    data: List[Optional[QuantizeOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [QuantizeOption(**item) for item in self.data]


@dataclass
class ResponseQuantizeStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)


@dataclass
class DownloadModelUrl:
    ai_model_id: str
    presigned_download_url: str


@dataclass
class ResponseQuantizeDownloadModelUrlItem(ResponseItem):
    data: Optional[DownloadModelUrl] = field(default_factory=dict)

    def __post_init__(self):
        self.data = DownloadModelUrl(**self.data)
