from netspresso.trainer.training.environment import EnvironmentConfig
from netspresso.trainer.training.logging import LoggingConfig
from netspresso.trainer.training.training import (
    ClassificationScheduleConfig,
    DetectionScheduleConfig,
    ScheduleConfig,
    SegmentationScheduleConfig,
)

TRAINING_CONFIG_TYPE = {
    "classification": ClassificationScheduleConfig,
    "detection": DetectionScheduleConfig,
    "segmentation": SegmentationScheduleConfig,
}


__all__ = ["ScheduleConfig", "TRAINING_CONFIG_TYPE", "EnvironmentConfig", "LoggingConfig"]
