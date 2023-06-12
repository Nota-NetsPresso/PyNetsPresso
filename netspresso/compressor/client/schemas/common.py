from typing import Literal


COMPRESSION_METHOD = Literal["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"]
RECOMMENDATION_METHOD = Literal["slamp", "vbmf"]
TASK = Literal[
    "image_classification",
    "object_detection",
    "image_segmentation",
    "semantic_segmentation",
    "instance_segmentation",
    "panoptic_segmentation",
    "other",
]
FRAMEWORK = Literal["tensorflow_keras", "onnx", "pytorch"]
ORIGIN_FROM = Literal["custom", "npms"]
