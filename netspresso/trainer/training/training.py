from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ScheduleConfig:
    epochs: int = 3
    ema: Optional[Dict] = field(default=None)
    optimizer: Dict = field(
        default_factory=lambda: {
            "name": "adamw",
            "lr": 6e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0005,
            "no_bias_decay": False,
            "no_norm_weight_decay": False,
            "overwrite": None,
        }
    )
    scheduler: Dict = field(
        default_factory=lambda: {
            "name": "cosine_no_sgdr",
            "warmup_epochs": 5,
            "warmup_bias_lr": 1e-5,
            "min_lr": 0.0,
        }
    )


@dataclass
class ClassificationScheduleConfig(ScheduleConfig):
    pass


@dataclass
class SegmentationScheduleConfig(ScheduleConfig):
    pass


@dataclass
class DetectionScheduleConfig(ScheduleConfig):
    pass
