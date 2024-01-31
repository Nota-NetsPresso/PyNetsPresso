from netspresso_trainer.cfg import ClassificationScheduleConfig, DetectionScheduleConfig, SegmentationScheduleConfig

TRAINING_CONFIG_TYPE = {
    "classification": ClassificationScheduleConfig,
    "detection": DetectionScheduleConfig,
    "segmentation": SegmentationScheduleConfig,
}
