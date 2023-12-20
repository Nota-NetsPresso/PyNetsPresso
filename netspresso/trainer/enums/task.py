from enum import Enum


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"
