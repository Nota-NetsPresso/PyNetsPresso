from netspresso_trainer.cfg import (
    LocalClassificationDatasetConfig,
    LocalDetectionDatasetConfig,
    LocalSegmentationDatasetConfig,
)

DATA_CONFIG_TYPE = {
    "classification": LocalClassificationDatasetConfig,
    "detection": LocalDetectionDatasetConfig,
    "segmentation": LocalSegmentationDatasetConfig,
}
