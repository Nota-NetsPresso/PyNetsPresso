import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from netspresso.enums.metadata import Status, TaskType


@dataclass
class InputShape:
    batch: int = 1
    channel: int = 3
    dimension: List[int] = field(default_factory=list)


@dataclass
class ModelInfo:
    task: str = ""
    model: str = ""
    dataset: str = ""
    input_shapes: List[InputShape] = field(default_factory=lambda: [InputShape()])


@dataclass
class TrainingInfo:
    epoch: int = 0
    batch_size: int = 0


@dataclass
class TrainerMetadata:
    status: Status = Status.IN_PROGRESS
    task_type: TaskType = TaskType.TRAIN
    model_info: ModelInfo = field(default_factory=ModelInfo)
    training_info: TrainingInfo = field(default_factory=TrainingInfo)
    logging_dir: str = ""
    best_fx_model_path: str = ""
    best_onnx_model_path: str = ""
    hparams: str = ""
    traning_result: Dict = field(default_factory=dict)

    def asdict(self) -> Dict:
        _dict = json.loads(json.dumps(asdict(self)))
        return _dict

    def update_status(self, status: Status):
        self.status = status

    def update_model_info(self, task, model, dataset, input_shapes):
        self.model_info.task = task
        self.model_info.model = model
        self.model_info.dataset = dataset
        self.model_info.input_shapes = input_shapes

    def update_training_info(self, epoch, batch_size):
        self.training_info.epoch = epoch
        self.training_info.batch_size = batch_size

    def update_training_result(self, training_summary):
        self.traning_result = training_summary

    def update_logging_dir(self, logging_dir):
        self.logging_dir = logging_dir

    def update_best_fx_model_path(self, best_fx_model_path):
        self.best_fx_model_path = best_fx_model_path

    def update_best_onnx_model_path(self, best_onnx_model_path):
        self.best_onnx_model_path = best_onnx_model_path

    def update_hparams(self, hparams):
        self.hparams = hparams
