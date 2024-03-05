from netspresso_trainer.cfg import (
    ClassificationAugmentationConfig,
    DetectionAugmentationConfig,
    SegmentationAugmentationConfig,
)

AUGMENTATION_CONFIG_TYPE = {
    "classification": ClassificationAugmentationConfig,
    "detection": DetectionAugmentationConfig,
    "segmentation": SegmentationAugmentationConfig,
}
