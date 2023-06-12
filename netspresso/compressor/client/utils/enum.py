from enum import Enum


class Task(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    OTHER = "other"


class Framework(Enum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class Extension(Enum):
    H5 = "h5"
    ZIP = "zip"
    PT = "pt"
    ONNX = "onnx"


class CompressionMethod(Enum):
    PR_L2 = "PR_L2"
    PR_GM = "PR_GM"
    PR_NN = "PR_NN"
    PR_ID = "PR_ID"
    FD_TK = "FD_TK"
    FD_CP = "FD_CP"
    FD_SVD = "FD_SVD"


class RecommendationMethod(Enum):
    SLAMP = "slamp"
    VBMF = "vbmf"
