from netspresso_trainer.cfg.model import *


CLASSIFICATION_MODELS = {
    "EfficientFormer": ClassificationEfficientFormerModelConfig,
    "MobileNetV3": ClassificationMobileNetV3ModelConfig,
    "MobileViT": ClassificationMobileViTModelConfig,
    "ResNet": ClassificationResNetModelConfig,
    "SegFormer": ClassificationSegFormerModelConfig,
    "ViT": ClassificationViTModelConfig,
    "MixNetS": ClassificationMixNetSmallModelConfig,
    "MixNetM": ClassificationMixNetMediumModelConfig,
    "MixNetL": ClassificationMixNetLargeModelConfig,
}

DETECTION_MODELS = {
    "EfficientFormer": DetectionEfficientFormerModelConfig,
    "YOLOX": DetectionYoloXModelConfig,
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
