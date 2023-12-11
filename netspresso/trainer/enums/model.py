from enum import Enum

from netspresso_trainer.cfg.model import *


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"


class Backbone(Enum):
    EfficientFormer = "efficientformer"
    MobileNetV3 = "mobilenetv3"
    MobileViT = "mobilevit"
    PIDNet = "pidnet"
    ResNet = "resnet"
    SegFormer = "segformer"
    ViT = "vit"
    CSPDarkNet = "cspdarknet"


class Head(Enum):
    FC = "fc"
    Faster_R_CNN = "faster_rcnn"
    FPN = "fpn"
    YOLOX_Head = "yolox_head"
    MLP_Decoder = "all_mlp_decoder"


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
    (Backbone.CSPDarkNet, Head.YOLOX_Head): DetectionYoloXModelConfig(),
}
