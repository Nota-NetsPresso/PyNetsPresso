from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class ModelSaveOptions:
    save_optimizer_state: bool = True
    save_best_only: bool = False
    best_model_criterion: str = "loss"  # metric or loss
    sample_input_size: List[int] = field(default_factory=lambda: [512, 512])
    onnx_export_opset: int = 13  # Recommend in range [13, 17]
    validation_epoch: int = 10
    save_checkpoint_epoch: Optional[int] = None

    def __post_init__(self):
        if self.save_checkpoint_epoch is None:
            self.save_checkpoint_epoch = self.validation_epoch


@dataclass
class Metrics:
    classwise_analysis: bool = False
    metric_names: Optional[List[str]] = None


@dataclass
class LoggingConfig:
    project_id: Optional[str] = None
    output_dir: Union[Path, str] = "./outputs"
    tensorboard: bool = True
    mlflow: bool = False
    image: bool = True
    stdout: bool = True
    model_save_options: ModelSaveOptions = field(default_factory=ModelSaveOptions)
    metrics: Metrics = field(default_factory=Metrics)
