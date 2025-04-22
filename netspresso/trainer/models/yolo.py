from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class ShufflenetV2ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "shufflenetv2",
            "params": {"model_size": "0.5x"},
            "stage_params": None,
        }
    )


@dataclass
class DetectionYoloFastestV2ModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolo_fastest_v2"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ShufflenetV2ArchitectureConfig(
            neck={
                "name": "lightfpn",
                "params": {
                    "out_channels": 72,
                },
            },
            head={
                "name": "yolo_fastest_head_v2",
                "params": {
                    "anchors": [
                        [12, 18, 37, 49, 52, 132],  # P2
                        [115, 73, 119, 199, 242, 238],  # P3
                    ]
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "score_thresh": 0.01,
                # postprocessor - nms
                "nms_thresh": 0.65,
                "anchors": [
                    [12, 18, 37, 49, 52, 132],  # P2
                    [115, 73, 119, 199, 242, 238],  # P3
                ],
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "criterion": "yolofastest_loss",
                "anchors": [
                    [12, 18, 37, 49, 52, 132],  # P2
                    [115, 73, 119, 199, 242, 238],  # P3
                ],
                "l1_activate_epoch": None,
                "weight": None,
            }
        ]
    )
