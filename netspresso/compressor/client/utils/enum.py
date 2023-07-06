from enum import Enum
from typing import Literal


class ExtendedEnum(str, Enum):
    @classmethod
    def create_literal(cls):
        return Literal[tuple(value.value for key, value in cls.__members__.items())]


class Task(ExtendedEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    OTHER = "other"


class Framework(ExtendedEnum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class Extension(ExtendedEnum):
    H5 = "h5"
    ZIP = "zip"
    PT = "pt"
    ONNX = "onnx"


class CompressionMethod(ExtendedEnum):
    PR_L2 = "PR_L2"
    PR_GM = "PR_GM"
    PR_NN = "PR_NN"
    PR_ID = "PR_ID"
    FD_TK = "FD_TK"
    FD_CP = "FD_CP"
    FD_SVD = "FD_SVD"


class RecommendationMethod(ExtendedEnum):
    SLAMP = "slamp"
    VBMF = "vbmf"


class OriginFrom(ExtendedEnum):
    CUSTOM = "custom"
    NPMS = "npms"


task_literal = Task.create_literal()
framework_literal = Framework.create_literal()
extension_literal = Extension.create_literal()
compression_literal = CompressionMethod.create_literal()
recommendation_literal = RecommendationMethod.create_literal()
originfrom_literal = OriginFrom.create_literal()
