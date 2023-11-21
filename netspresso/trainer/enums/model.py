from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from netspresso_trainer.cfg.model import *
from netspresso_trainer.cfg.model import ArchitectureConfig


class Backbone(Enum):
    EfficientFormer = "efficientformer"
    MobileNetV3 = "mobilenetv3"
    MobileViT = "mobilevit"
    PIDNet = "pidnet"
    ResNet = "resnet"
    SegFormer = "segformer"
    ViT = "vit"
    YOLOX = "yolox"


class Head(Enum):
    FC = "fc"
    Faster_R_CNN = "faster_rcnn"
    FPN = "fpn"
    YOLOX_Head = "yolox_head"
    MLP_Decoder = "all_mlp_decoder"


@dataclass
class YOLOXArchitectureConfig(ArchitectureConfig):
    backbone: Dict[str, Any] = field(default_factory=lambda: {
        "name": "cspdarknet",
        "params": {
            "dep_mul": 0.33,
            "wid_mul": 0.5,
            "act_type": "silu",
        },
        "stage_params": None,
    })


@dataclass
class DetectionYOLOXModelConfig(ModelConfig):
    task: str = "detection"
    checkpoint: Optional[Union[Path, str]] = "./weights/yolox/yolox_s.pth"
    architecture: ArchitectureConfig = field(default_factory=lambda: YOLOXArchitectureConfig(
        head={"name": "yolox_head"}
    ))
    losses: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"criterion": "yolox_loss", "weight": None},
    ])


SUPPORTED_MODELS = {
    (Backbone.EfficientFormer, Head.FC): ClassificationEfficientFormerModelConfig(),
    (Backbone.EfficientFormer, Head.Faster_R_CNN): DetectionEfficientFormerModelConfig(),
    (Backbone.EfficientFormer, Head.MLP_Decoder): SegmentationEfficientFormerModelConfig(),
    (Backbone.MobileNetV3, Head.FC): ClassificationMobileNetV3ModelConfig(),
    (Backbone.MobileNetV3, Head.MLP_Decoder): SegmentationMobileNetV3ModelConfig(),
    (Backbone.MobileViT, Head.FC): ClassificationMobileViTModelConfig(),
    (Backbone.PIDNet, None): PIDNetModelConfig(),
    (Backbone.ResNet, Head.FC): ClassificationResNetModelConfig(),
    (Backbone.ResNet, Head.MLP_Decoder): SegmentationResNetModelConfig(),
    (Backbone.SegFormer, Head.FC): ClassificationSegFormerModelConfig(),
    (Backbone.SegFormer, Head.MLP_Decoder): SegmentationSegFormerModelConfig(),
    (Backbone.ViT, Head.FC): ClassificationViTModelConfig(),
    (Backbone.YOLOX, Head.YOLOX_Head): DetectionYOLOXModelConfig(),
}
