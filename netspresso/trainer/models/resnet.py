from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netspresso.trainer.models.base import ArchitectureConfig, CheckpointConfig, ModelConfig


@dataclass
class ResNet18ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "resnet",
            "params": {
                "block_type": "basicblock",
                "norm_type": "batch_norm",
                "return_stage_idx": None,
                "split_stem_conv": False,
                "first_stage_shortcut_conv": False,
            },
            "stage_params": [
                {
                    "channels": 64,
                    "num_blocks": 2,
                },
                {
                    "channels": 128,
                    "num_blocks": 2,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 256,
                    "num_blocks": 2,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 512,
                    "num_blocks": 2,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
            ],
        }
    )


@dataclass
class ResNet34ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "resnet",
            "params": {
                "block_type": "basicblock",
                "norm_type": "batch_norm",
                "return_stage_idx": None,
                "split_stem_conv": False,
                "first_stage_shortcut_conv": False,
            },
            "stage_params": [
                {
                    "channels": 64,
                    "num_blocks": 3,
                },
                {
                    "channels": 128,
                    "num_blocks": 4,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 256,
                    "num_blocks": 6,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 512,
                    "num_blocks": 3,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
            ],
        }
    )


@dataclass
class ResNet50ArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "resnet",
            "params": {
                "block_type": "bottleneck",
                "norm_type": "batch_norm",
                "return_stage_idx": None,
                "split_stem_conv": False,
                "first_stage_shortcut_conv": False,
            },
            "stage_params": [
                {
                    "channels": 64,
                    "num_blocks": 3,
                },
                {
                    "channels": 128,
                    "num_blocks": 4,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 256,
                    "num_blocks": 6,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
                {
                    "channels": 512,
                    "num_blocks": 3,
                    "replace_stride_with_dilation": False,
                    "replace_stride_with_pooling": False,
                },
            ],
        }
    )


@dataclass
class ClassificationResNet18ModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "resnet18"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ResNet18ArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class ClassificationResNet34ModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "resnet34"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ResNet34ArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class ClassificationResNet50ModelConfig(ModelConfig):
    task: str = "classification"
    name: str = "resnet50"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ResNet50ArchitectureConfig(
            head={
                "name": "fc",
                "params": {
                    "num_layers": 1,
                    "intermediate_channels": None,
                    "act_type": None,
                    "dropout_prob": 0.0,
                },
            }
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "cross_entropy", "label_smoothing": 0.1, "weight": None}]
    )


@dataclass
class SegmentationResNet50ModelConfig(ModelConfig):
    task: str = "segmentation"
    name: str = "resnet50"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ResNet50ArchitectureConfig(
            backbone={
                "name": "resnet",
                "params": {
                    "block_type": "bottleneck",
                    "norm_type": "batch_norm",
                    "return_stage_idx": [0, 1, 2, 3],
                    "split_stem_conv": False,
                    "first_stage_shortcut_conv": False,
                },
                "stage_params": [
                    {
                        "channels": 64,
                        "num_blocks": 3,
                    },
                    {
                        "channels": 128,
                        "num_blocks": 4,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                    {
                        "channels": 256,
                        "num_blocks": 6,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                    {
                        "channels": 512,
                        "num_blocks": 3,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                ],
            },
            head={
                "name": "all_mlp_decoder",
                "params": {
                    "intermediate_channels": 256,
                    "classifier_dropout_prob": 0.0,
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = None
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [{"criterion": "seg_cross_entropy", "ignore_index": 255, "weight": None}]
    )


@dataclass
class DetectionResNet50ModelConfig(ModelConfig):
    task: str = "detection"
    name: str = "resnet50"
    architecture: ArchitectureConfig = field(
        default_factory=lambda: ResNet50ArchitectureConfig(
            backbone={
                "name": "resnet",
                "params": {
                    "block_type": "bottleneck",
                    "norm_type": "batch_norm",
                    "return_stage_idx": [0, 1, 2, 3],
                    "split_stem_conv": False,
                    "first_stage_shortcut_conv": False,
                },
                "stage_params": [
                    {
                        "channels": 64,
                        "num_blocks": 3,
                    },
                    {
                        "channels": 128,
                        "num_blocks": 4,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                    {
                        "channels": 256,
                        "num_blocks": 6,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                    {
                        "channels": 512,
                        "num_blocks": 3,
                        "replace_stride_with_dilation": False,
                        "replace_stride_with_pooling": False,
                    },
                ],
            },
            neck={
                "name": "fpn",
                "params": {
                    "num_outs": 4,
                    "start_level": 0,
                    "end_level": -1,
                    "add_extra_convs": False,
                    "relu_before_extra_convs": False,
                },
            },
            head={
                "name": "anchor_decoupled_head",
                "params": {
                    # Anchor parameters
                    "anchor_sizes": [
                        [
                            32,
                        ],
                        [
                            64,
                        ],
                        [
                            128,
                        ],
                        [
                            256,
                        ],
                    ],
                    "aspect_ratios": [0.5, 1.0, 2.0],
                    "num_layers": 1,
                    "norm_type": "batch_norm",
                },
            },
        )
    )
    postprocessor: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "params": {
                # postprocessor - decode
                "topk_candidates": 1000,
                "score_thresh": 0.05,
                # postprocessor - nms
                "nms_thresh": 0.45,
                "class_agnostic": False,
            },
        }
    )
    losses: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"criterion": "retinanet_loss", "weight": None},
        ]
    )
