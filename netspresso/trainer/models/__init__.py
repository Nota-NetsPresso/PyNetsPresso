from typing import Dict, List

from netspresso.trainer.models.base import CheckpointConfig, ModelConfig
from netspresso.trainer.models.efficientformer import (
    ClassificationEfficientFormerModelConfig,
    DetectionEfficientFormerModelConfig,
    SegmentationEfficientFormerModelConfig,
)
from netspresso.trainer.models.mixnet import (
    ClassificationMixNetLargeModelConfig,
    ClassificationMixNetMediumModelConfig,
    ClassificationMixNetSmallModelConfig,
    DetectionMixNetLargeModelConfig,
    DetectionMixNetMediumModelConfig,
    DetectionMixNetSmallModelConfig,
    SegmentationMixNetLargeModelConfig,
    SegmentationMixNetMediumModelConfig,
    SegmentationMixNetSmallModelConfig,
)
from netspresso.trainer.models.mobilenetv3 import (
    ClassificationMobileNetV3LargeModelConfig,
    ClassificationMobileNetV3SmallModelConfig,
    DetectionMobileNetV3SmallModelConfig,
    SegmentationMobileNetV3SmallModelConfig,
)
from netspresso.trainer.models.mobilevit import ClassificationMobileViTModelConfig
from netspresso.trainer.models.pidnet import PIDNetModelConfig
from netspresso.trainer.models.resnet import (
    ClassificationResNet18ModelConfig,
    ClassificationResNet34ModelConfig,
    ClassificationResNet50ModelConfig,
    DetectionResNet50ModelConfig,
    SegmentationResNet50ModelConfig,
)
from netspresso.trainer.models.rtmpose import PoseEstimationMobileNetV3SmallModelConfig
from netspresso.trainer.models.segformer import SegmentationSegFormerB0ModelConfig
from netspresso.trainer.models.vit import ClassificationViTTinyModelConfig
from netspresso.trainer.models.yolo import DetectionYoloFastestV2ModelConfig
from netspresso.trainer.models.yolov9 import (
    DetectionYoloV9CModelConfig,
    DetectionYoloV9MModelConfig,
    DetectionYoloV9SModelConfig,
)
from netspresso.trainer.models.yolox import (
    DetectionYoloXLModelConfig,
    DetectionYoloXMModelConfig,
    DetectionYoloXSModelConfig,
    DetectionYoloXXModelConfig,
)

CLASSIFICATION_MODELS = {
    "efficientformer_l1": ClassificationEfficientFormerModelConfig,
    "mobilenet_v3_small": ClassificationMobileNetV3SmallModelConfig,
    "mobilenet_v3_large": ClassificationMobileNetV3LargeModelConfig,
    "mobilevit_s": ClassificationMobileViTModelConfig,
    "resnet18": ClassificationResNet18ModelConfig,
    "resnet34": ClassificationResNet34ModelConfig,
    "resnet50": ClassificationResNet50ModelConfig,
    "vit_tiny": ClassificationViTTinyModelConfig,
    "mixnet_s": ClassificationMixNetSmallModelConfig,
    "mixnet_m": ClassificationMixNetMediumModelConfig,
    "mixnet_l": ClassificationMixNetLargeModelConfig,
}

DETECTION_MODELS = {
    "yolox_s": DetectionYoloXSModelConfig,
    "yolox_m": DetectionYoloXMModelConfig,
    "yolox_l": DetectionYoloXLModelConfig,
    "yolox_x": DetectionYoloXXModelConfig,
}

SEGMENTATION_MODELS = {
    "efficientformer_l1": SegmentationEfficientFormerModelConfig,
    "mobilenet_v3_small": SegmentationMobileNetV3SmallModelConfig,
    "resnet50": SegmentationResNet50ModelConfig,
    "segformer_b0": SegmentationSegFormerB0ModelConfig,
    "mixnet_s": SegmentationMixNetSmallModelConfig,
    "mixnet_m": SegmentationMixNetMediumModelConfig,
    "mixnet_l": SegmentationMixNetLargeModelConfig,
    "pidnet_s": PIDNetModelConfig,
}

POSEESTIMATION_MODELS = {
    "mobilenet_v3_small": PoseEstimationMobileNetV3SmallModelConfig,
}

# NOT_SUPPORTED_PRETRAINED_MODELS = ["YOLO-Fastest"]


__all__ = [
    "CLASSIFICATION_MODELS",
    "DETECTION_MODELS",
    "SEGMENTATION_MODELS",
    "POSEESTIMATION_MODELS",
    "MODEL_NAME_DISPLAY_MAP",
    "CheckpointConfig",
    "ModelConfig",
]


def get_all_available_models() -> Dict[str, List[str]]:
    """Get all available models for each task, excluding deprecated names.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each task to its available models.
    """
    all_models = {
        "classification": list(CLASSIFICATION_MODELS),
        "detection": list(DETECTION_MODELS),
        "segmentation": list(SEGMENTATION_MODELS),
    }
    return all_models
