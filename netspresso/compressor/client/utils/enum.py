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


class CompressionMethod(str, Enum):
    PR_L2 = "PR_L2"
    PR_GM = "PR_GM"
    PR_NN = "PR_NN"
    PR_ID = "PR_ID"
    FD_TK = "FD_TK"
    FD_CP = "FD_CP"
    FD_SVD = "FD_SVD"

    @classmethod
    def create_literal(cls):
        return Literal["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"]


class RecommendationMethod(str, Enum):
    SLAMP = "slamp"
    VBMF = "vbmf"

    @classmethod
    def create_literal(cls):
        return Literal["slamp", "vbmf"]


class OriginFrom(str, Enum):
    CUSTOM = "custom"
    NPMS = "npms"

    @classmethod
    def create_literal(cls):
        return Literal["custom", "npms"]


class Policy(str, Enum):
    SUM = "sum"
    AVERAGE = "average"
    BACKWARD = "backward"

    @classmethod
    def create_literal(cls):
        return Literal["sum", "average", "backward"]


class GroupPolicy(str, Enum):
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    NONE = "none"

    @classmethod
    def create_literal(cls):
        return Literal["sum", "average", "count", "none"]


class LayerNorm(str, Enum):
    NONE = "none"
    STANDARD_SCORE = "standard_score"
    TSS_NORM = "tss_norm"
    LINEAR_SCALING = "linear_scaling"
    SOFTMAX_NORM = "softmax_norm"

    @classmethod
    def create_literal(cls):
        return Literal["none", "standard_score", "tss_norm", "linear_scaling", "softmax_norm"]


task_literal = Task.create_literal()
framework_literal = Framework.create_literal()
extension_literal = Extension.create_literal()
compression_literal = CompressionMethod.create_literal()
recommendation_literal = RecommendationMethod.create_literal()
originfrom_literal = OriginFrom.create_literal()
policy_literal = Policy.create_literal()
grouppolicy_literal = GroupPolicy.create_literal()
layernorm_literal = LayerNorm.create_literal()
