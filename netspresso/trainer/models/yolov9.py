from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class GelanCArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "gelan",
            "params": {
                "stem_out_channels": 64,
                "stem_kernel_size": 3,
                "stem_stride": 2,
                "return_stage_idx": [1, 2, 3],
                "act_type": "silu",
            },
            "stage_params": [
                [
                    ['conv', 128, 3, 2],
                    ['repncspelan', 256, 128, False, 1],
                ],
                [
                    ['adown', 256],
                    ['repncspelan', 512, 256, False, 1],
                ],
                [
                    ['adown', 512],
                    ['repncspelan', 512, 512, False, 1],
                ],
                [
                    ['adown', 512],
                    ['repncspelan', 512, 512, False, 1],
                ],
            ],
        }
    )

@dataclass
class GelanMArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "gelan",
            "params": {
                "stem_out_channels": 32,
                "stem_kernel_size": 3,
                "stem_stride": 2,
                "return_stage_idx": [1, 2, 3],
                "act_type": "silu",
            },
            "stage_params": [
                [
                    ['conv', 64, 3, 2],
                    ['repncspelan', 128, 128, False, 1],
                ],
                [
                    ['aconv', 240],
                    ['repncspelan', 240, 240, False, 1],
                ],
                [
                    ['aconv', 360],
                    ['repncspelan', 360, 360, False, 1],
                ],
                [
                    ['aconv', 480],
                    ['repncspelan', 480, 480, False, 1],
                ],
            ],
        }
    )


@dataclass
class GelanSArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "gelan",
            "params": {
                "stem_out_channels": 32,
                "stem_kernel_size": 3,
                "stem_stride": 2,
                "return_stage_idx": [1, 2, 3],
                "act_type": "silu",
            },
            "stage_params": [
                [
                    ['conv', 64, 3, 2],
                    ['elan', 64, 64, False],
                ],
                [
                    ['aconv', 128],
                    ['repncspelan', 128, 128, False, 3],
                ],
                [
                    ['aconv', 192],
                    ['repncspelan', 192, 192, False, 3],
                ],
                [
                    ['aconv', 256],
                    ['repncspelan', 256, 256, False, 3],
                ],
            ],
        }
    )


@dataclass
class DetectionYoloV9CModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolov9_c"
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(use_pretrained=False))
    architecture: ArchitectureConfig = field(
        default_factory=lambda: GelanCArchitectureConfig(
            neck={
                "name": "yolov9fpn",
                "params": {
                    "repeat_num": 1,
                    "act_type": "silu",
                    "use_aux_loss": False,
                    "bu_type": "aconv",
                    "spp_channels": 480,
                    "n4_channels": 360,
                    "p3_channels": 240,
                    "p3_to_p4_channels": 184,
                    "p4_channels": 360,
                    "p4_to_p5_channels": 240,
                    "p5_channels": 480,
                },
            },
            head={
                "name": "yolo_detection_head",
                "params": {
                    "version": "v9",
                    "num_anchors": None,
                    "use_group": True,
                    "reg_max": 16,
                    "act_type": "silu",
                    "use_aux_loss": False,
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "reg_max": 16,
                "score_thresh": 0.01,
                # postprocessor - nms
                "nms_thresh": 0.65,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "criterion": "yolov9_loss",
                "reg_max": 16,
                "weight": None,
                "l1_activate_epoch": None,
            }
        ]
    )


@dataclass
class DetectionYoloV9MModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolov9_m"
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(use_pretrained=False))
    architecture: ArchitectureConfig = field(
        default_factory=lambda: GelanMArchitectureConfig(
            neck={
                "name": "yolov9fpn",
                "params": {
                    "repeat_num": 1,
                    "act_type": "silu",
                    "use_aux_loss": False,
                    "bu_type": "aconv",
                    "spp_channels": 480,
                    "n4_channels": 360,
                    "p3_channels": 240,
                    "p3_to_p4_channels": 184,
                    "p4_channels": 360,
                    "p4_to_p5_channels": 240,
                    "p5_channels": 480,
                },
            },
            head={
                "name": "yolo_detection_head",
                "params": {
                    "version": "v9",
                    "num_anchors": None,
                    "use_group": True,
                    "reg_max": 16,
                    "act_type": "silu",
                    "use_aux_loss": False,
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "reg_max": 16,
                "score_thresh": 0.01,
                # postprocessor - nms
                "nms_thresh": 0.65,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "criterion": "yolov9_loss",
                "reg_max": 16,
                "weight": None,
                "l1_activate_epoch": None,
            }
        ]
    )


@dataclass
class DetectionYoloV9SModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "yolov9_s"
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(use_pretrained=False))
    architecture: ArchitectureConfig = field(
        default_factory=lambda: GelanSArchitectureConfig(
            neck={
                "name": "yolov9fpn",
                "params": {
                    "repeat_num": 3,
                    "act_type": "silu",
                    "use_aux_loss": False,
                    "bu_type": "aconv",
                    "spp_channels": 256,
                    "n4_channels": 192,
                    "p3_channels": 128,
                    "p3_to_p4_channels": 96,
                    "p4_channels": 192,
                    "p4_to_p5_channels": 128,
                    "p5_channels": 256,
                },
            },
            head={
                "name": "yolo_detection_head",
                "params": {
                    "version": "v9",
                    "num_anchors": None,
                    "use_group": True,
                    "reg_max": 16,
                    "act_type": "silu",
                    "use_aux_loss": False,
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "reg_max": 16,
                "score_thresh": 0.01,
                # postprocessor - nms
                "nms_thresh": 0.65,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "criterion": "yolov9_loss",
                "reg_max": 16,
                "weight": None,
                "l1_activate_epoch": None,
            }
        ]
    )
