from enum import Enum


class EncryptionKey(str, Enum):
    NVIDIA_TAO = "nvidia_tao"
    NVIDIA_TLT = "nvidia_tlt"
    TLT_ENCODE = "tlt_encode"


class CheckpointChooseMethod(str, Enum):
    LATEST_MODEL = "latest_model"
    BEST_MODEL = "best_model"
    FROM_EPOCH_NUMBER = "from_epoch_number"


class NetworkArch(str, Enum):
    CLASSIFICATION_TF1 = "classification_tf1"
    CLASSIFICATION_TF2 = "classification_tf2"
    CLASSIFICATION_PYT = "classification_pyt"
    DETECTNET_V2 = "detectnet_v2"
    FASTER_RCNN = "faster_rcnn"
    YOLO_V3 = "yolo_v3"
    YOLO_V4 = "yolo_v4"
    YOLO_V4_TINY = "yolo_v4_tiny"
    SSD = "ssd"
    DSSD = "dssd"
    RETINANET = "retinanet"
    DEFORMABLE_DETR = "deformable_detr"
    DINO = "dino"
    EFFICIENTDET_TF1 = "efficientdet_tf1"
    EFFICIENTDET_TF2 = "efficientdet_tf2"
    OCDNET = "ocdnet"
    MASK_RCNN = "mask_rcnn"
    UNET = "unet"
    SEGFORMER = "segformer"
    MULTITASK_CLASSIFICATION = "multitask_classification"
    LPRNET = "lprnet"
    BPNET = "bpnet"
    FPENET = "fpenet"
    ACTION_RECOGNITION = "action_recognition"
    MAL = "mal"
    ML_RECOG = "ml_recog"
    OCRNET = "ocrnet"
    OPTICAL_INSPECTION = "optical_inspection"
    POINTPILLARS = "pointpillars"
    POSE_CLASSIFICATION = "pose_classification"
    RE_IDENTIFICATION = "re_identification"
    VISUAL_CHANGENET = "visual_changenet"
    CENTERPOSE = "centerpose"
    ANNOTATIONS = "annotations"
    ANALYTICS = "analytics"
    AUGMENTATION = "augmentation"
    AUTO_LABEL = "auto_label"
