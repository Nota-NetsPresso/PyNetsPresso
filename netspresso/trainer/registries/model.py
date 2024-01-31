from netspresso_trainer.cfg.model import (
    ClassificationEfficientFormerModelConfig,
    ClassificationMixNetLargeModelConfig,
    ClassificationMixNetMediumModelConfig,
    ClassificationMixNetSmallModelConfig,
    ClassificationMobileNetV3ModelConfig,
    ClassificationMobileViTModelConfig,
    ClassificationResNetModelConfig,
    ClassificationViTModelConfig,
    DetectionEfficientFormerModelConfig,
    DetectionMixNetLargeModelConfig,
    DetectionMixNetMediumModelConfig,
    DetectionMixNetSmallModelConfig,
    DetectionMobileNetV3ModelConfig,
    DetectionResNetModelConfig,
    DetectionYoloXModelConfig,
    PIDNetModelConfig,
    SegmentationEfficientFormerModelConfig,
    SegmentationMixNetLargeModelConfig,
    SegmentationMixNetMediumModelConfig,
    SegmentationMixNetSmallModelConfig,
    SegmentationMobileNetV3ModelConfig,
    SegmentationResNetModelConfig,
    SegmentationSegFormerModelConfig,
)

CLASSIFICATION_MODELS = {
    "EfficientFormer": ClassificationEfficientFormerModelConfig,
    "MobileNetV3": ClassificationMobileNetV3ModelConfig,
    "MobileViT": ClassificationMobileViTModelConfig,
    "ResNet": ClassificationResNetModelConfig,
    "ViT": ClassificationViTModelConfig,
    "MixNetS": ClassificationMixNetSmallModelConfig,
    "MixNetM": ClassificationMixNetMediumModelConfig,
    "MixNetL": ClassificationMixNetLargeModelConfig,
}

DETECTION_MODELS = {
    "EfficientFormer": DetectionEfficientFormerModelConfig,
    "YOLOX-S": DetectionYoloXModelConfig,
    "ResNet": DetectionResNetModelConfig,
    "MobileNetV3": DetectionMobileNetV3ModelConfig,
    "MixNetL": DetectionMixNetLargeModelConfig,
    "MixNetM": DetectionMixNetMediumModelConfig,
    "MixNetS": DetectionMixNetSmallModelConfig,
}

SEGMENTATION_MODELS = {
    "EfficientFormer": SegmentationEfficientFormerModelConfig,
    "MobileNetV3": SegmentationMobileNetV3ModelConfig,
    "ResNet": SegmentationResNetModelConfig,
    "SegFormer": SegmentationSegFormerModelConfig,
    "MixNetS": SegmentationMixNetSmallModelConfig,
    "MixNetM": SegmentationMixNetMediumModelConfig,
    "MixNetL": SegmentationMixNetLargeModelConfig,
    "PIDNet": PIDNetModelConfig,
}
