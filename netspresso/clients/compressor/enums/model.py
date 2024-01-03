from enum import Enum
from typing import Literal


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    OTHER = "other"

    @classmethod
    def create_literal(cls):
        return Literal[
            "image_classification",
            "object_detection",
            "image_segmentation",
            "semantic_segmentation",
            "instance_segmentation",
            "panoptic_segmentation",
            "other",
        ]


class Framework(str, Enum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    PYTORCH = "pytorch"
    ONNX = "onnx"

    @classmethod
    def create_literal(cls):
        return Literal["tensorflow_keras", "pytorch", "onnx"]


class Extension(str, Enum):
    H5 = "h5"
    ZIP = "zip"
    PT = "pt"
    ONNX = "onnx"

    @classmethod
    def create_literal(cls):
        return Literal["h5", "zip", "pt", "onnx"]


class OriginFrom(str, Enum):
    CUSTOM = "custom"
    NPMS = "npms"

    @classmethod
    def create_literal(cls):
        return Literal["custom", "npms"]


task_literal = Task.create_literal()
framework_literal = Framework.create_literal()
extension_literal = Extension.create_literal()
originfrom_literal = OriginFrom.create_literal()
