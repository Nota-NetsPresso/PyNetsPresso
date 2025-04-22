from dataclasses import dataclass, field
from typing import Dict, Union

from netspresso.enums import QuantizationMode, QuantizationPrecision, SimilarityMetric, TaskType
from netspresso.metadata.common import BaseMetadata, ModelInfo


@dataclass
class QuantizeInfo:
    quantize_task_uuid: str = ""
    model_file_name: str = ""
    quantization_mode: QuantizationMode = QuantizationMode.UNIFORM_PRECISION_QUANTIZATION
    metric: SimilarityMetric = SimilarityMetric.SNR
    threshold: Union[float, int] = 0
    weight_precision: QuantizationPrecision = QuantizationPrecision.INT8
    activation_precision: QuantizationPrecision = QuantizationPrecision.INT8
    input_model_uuid: str = ""
    output_model_uuid: str = ""


@dataclass
class QuantizerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.QUANTIZE
    input_model_path: str = ""
    quantized_model_path: str = ""
    recommendation_result_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    quantize_info: QuantizeInfo = field(default_factory=QuantizeInfo)
    compare_result: Dict = field(default_factory=dict)
