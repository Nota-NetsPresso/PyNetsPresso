import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.enums.metadata import Status, TaskType

from .common import TargetDevice


@dataclass
class InputShape:
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class Model:
    size: int = 0
    flops: int = 0
    number_of_parameters: int = 0
    trainable_parameters: int = 0
    non_trainable_parameters: int = 0
    number_of_layers: Optional[int] = None
    model_id: str = ""


@dataclass
class ModelInfo:
    task: str = ""
    framework: str = ""
    input_shapes: List[InputShape] = field(default_factory=lambda: [InputShape()])


@dataclass
class CompressionInfo:
    method: str = ""
    ratio: float = 0.0
    options: Dict[str, Any] = None
    layers: List[Dict] = field(default_factory=list)


@dataclass
class Results:
    original_model: Model = field(default_factory=Model)
    compressed_model: Model = field(default_factory=Model)


@dataclass
class CompressorMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.COMPRESS
    compressed_model_path: str = ""
    compressed_onnx_model_path: str = ""
    results: Results = field(default_factory=Results)
    model_info: ModelInfo = field(default_factory=ModelInfo)
    compression_info: CompressionInfo = field(default_factory=CompressionInfo)
    available_devices: List[TargetDevice] = field(default_factory=list)

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_status(self, status: Status):
        self.status = status

    def update_model_info(self, task, framework, input_shapes):
        self.model_info.task = task
        self.model_info.framework = framework
        self.model_info.input_shapes = input_shapes

    def update_compression_info(self, method, options, layers, ratio=0.0):
        self.compression_info.method = method
        self.compression_info.ratio = ratio
        self.compression_info.options = options
        self.compression_info.layers = layers

    def update_compressed_model_path(self, compressed_model_path):
        self.compressed_model_path = compressed_model_path

    def update_compressed_onnx_model_path(self, compressed_onnx_model_path):
        self.compressed_onnx_model_path = compressed_onnx_model_path

    def update_results(self, model, compressed_model):
        def update_model_fields(target, source):
            target.size = source.model_size
            target.flops = source.flops
            target.number_of_parameters = source.trainable_parameters + source.non_trainable_parameters
            target.trainable_parameters = source.trainable_parameters
            target.non_trainable_parameters = source.non_trainable_parameters
            target.number_of_layers = source.number_of_layers if source.number_of_layers != 0 else None
            target.model_id = source.model_id

        update_model_fields(self.results.original_model, model)
        update_model_fields(self.results.compressed_model, compressed_model)

    def update_available_devices(self, available_devices):
        self.available_devices = [
            TargetDevice(
                display_name=device.display_name,
                display_brand_name=device.display_brand_name,
                device_name=device.device_name,
                software_version=device.software_version,
                software_version_display_name=device.software_version_display_name,
                hardware_type=device.hardware_type,
            )
            for device in available_devices
        ]
