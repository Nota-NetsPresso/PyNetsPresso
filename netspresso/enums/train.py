from enum import Enum


class Task(str, Enum):
    IMAGE_CLASSIFICATION = "classification"
    OBJECT_DETECTION = "detection"
    SEMANTIC_SEGMENTATION = "segmentation"


class TaskDisplay(str, Enum):
    IMAGE_CLASSIFICATION = "Classification"
    OBJECT_DETECTION = "Object Detection"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"


TASK_DISPLAY_MAP = {
    Task.IMAGE_CLASSIFICATION: TaskDisplay.IMAGE_CLASSIFICATION,
    Task.OBJECT_DETECTION: TaskDisplay.OBJECT_DETECTION,
    Task.SEMANTIC_SEGMENTATION: TaskDisplay.SEMANTIC_SEGMENTATION,
}


class Framework(str, Enum):
    PYTORCH = "pytorch"


class FrameworkDisplay(str, Enum):
    PYTORCH = "PyTorch"


FRAMEWORK_DISPLAY_MAP = {
    Framework.PYTORCH: FrameworkDisplay.PYTORCH,
}


class Optimizer(str, Enum):
    ADADELTA = "adadelta"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMAX = "adamax"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    SGD = "sgd"


class OptimizerDisplay(str, Enum):
    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    ADAMAX = "Adamax"
    ADAMW = "AdamW"
    RMSPROP = "RMSprop"
    SGD = "SGD"


OPTIMIZER_DISPLAY_MAP = {
    Optimizer.ADADELTA: OptimizerDisplay.ADADELTA,
    Optimizer.ADAGRAD: OptimizerDisplay.ADAGRAD,
    Optimizer.ADAM: OptimizerDisplay.ADAM,
    Optimizer.ADAMAX: OptimizerDisplay.ADAMAX,
    Optimizer.ADAMW: OptimizerDisplay.ADAMW,
    Optimizer.RMSPROP: OptimizerDisplay.RMSPROP,
    Optimizer.SGD: OptimizerDisplay.SGD,
}


class Scheduler(str, Enum):
    COSINE_ANNEALING_WARM_RESTARTS = "cosine"
    STEP_LR = "step"
    POLYNOMIAL_LR = "poly"
    COSINE_ANNEALING_LR = "cosine_no_sgdr"
    MULTI_STEP_LR = "multi_step"


class SchedulerDisplay(str, Enum):
    COSINE_ANNEALING_WARM_RESTARTS = "CosineAnnealingWarmRestartsWithCustomWarmUp"
    STEP_LR = "StepLR"
    POLYNOMIAL_LR = "PolynomialLRWithWarmUp"
    COSINE_ANNEALING_LR = "CosineAnnealingLRWithCustomWarmUp"
    MULTI_STEP_LR = "MultiStepLR"


SCHEDULER_DISPLAY_MAP = {
    Scheduler.COSINE_ANNEALING_WARM_RESTARTS: SchedulerDisplay.COSINE_ANNEALING_WARM_RESTARTS,
    Scheduler.STEP_LR: SchedulerDisplay.STEP_LR,
    Scheduler.POLYNOMIAL_LR: SchedulerDisplay.POLYNOMIAL_LR,
    Scheduler.COSINE_ANNEALING_LR: SchedulerDisplay.COSINE_ANNEALING_LR,
    Scheduler.MULTI_STEP_LR: SchedulerDisplay.MULTI_STEP_LR,
}


class StorageLocation(str, Enum):
    LOCAL = "local"
    STORAGE = "storage"


class AugmentationType(str, Enum):
    TRAIN = "train"
    INFERENCE = "inference"


class PretrainedModel(str, Enum):
    EFFICIENTFORMER_L1 = "efficientformer_l1"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILEVIT_S = "mobilevit_s"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    VIT_TINY = "vit_tiny"
    MIXNET_S = "mixnet_s"
    MIXNET_M = "mixnet_m"
    MIXNET_L = "mixnet_l"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    YOLOX_L = "yolox_l"
    YOLOX_X = "yolox_x"
    YOLO_FASTEST_V2 = "yolo_fastest_v2"
    SEGFORMER_B0 = "segformer_b0"
    PIDNET_S = "pidnet_s"
    YOLOV9_S = "yolov9_s"
    YOLOV9_M = "yolov9_m"
    YOLOV9_C = "yolov9_c"


class PretrainedModelDisplay(str, Enum):
    EFFICIENTFORMER_L1 = "EfficientFormer-L1"
    MOBILENET_V3_SMALL = "MobileNetV3-S"
    MOBILENET_V3_LARGE = "MobileNetV3-L"
    MOBILEVIT_S = "MobileViT-S"
    RESNET18 = "ResNet18"
    RESNET34 = "ResNet34"
    RESNET50 = "ResNet50"
    VIT_TINY = "ViT-T"
    MIXNET_S = "MixNet-S"
    MIXNET_M = "MixNet-M"
    MIXNET_L = "MixNet-L"
    YOLOX_S = "YOLOX-S"
    YOLOX_M = "YOLOX-M"
    YOLOX_L = "YOLOX-L"
    YOLOX_X = "YOLOX-X"
    YOLO_FASTEST_V2 = "YOLO-FastestV2"
    SEGFORMER_B0 = "SegFormer-B0"
    PIDNET_S = "PIDNet-S"
    YOLOV9_S = "YOLOv9-S"
    YOLOV9_M = "YOLOv9-M"
    YOLOV9_C = "YOLOv9-C"


class PretrainedModelGroup(str, Enum):
    RESNET = "ResNet"
    MOBILENET = "MobileNet"
    MOBILEVIT = "MobileViT"
    EFFICIENTFORMER = "EfficientFormer"
    VIT = "ViT"
    MIXNET = "MixNet"
    YOLOX = "YOLOX"
    YOLO = "YOLO"
    SEGFORMER = "SegFormer"
    PIDNET = "PIDNet"
    YOLOV9 = "YOLOv9"

MODEL_DISPLAY_MAP = {
    PretrainedModel.EFFICIENTFORMER_L1: PretrainedModelDisplay.EFFICIENTFORMER_L1,
    PretrainedModel.MOBILENET_V3_SMALL: PretrainedModelDisplay.MOBILENET_V3_SMALL,
    PretrainedModel.MOBILENET_V3_LARGE: PretrainedModelDisplay.MOBILENET_V3_LARGE,
    PretrainedModel.MOBILEVIT_S: PretrainedModelDisplay.MOBILEVIT_S,
    PretrainedModel.RESNET18: PretrainedModelDisplay.RESNET18,
    PretrainedModel.RESNET34: PretrainedModelDisplay.RESNET34,
    PretrainedModel.RESNET50: PretrainedModelDisplay.RESNET50,
    PretrainedModel.VIT_TINY: PretrainedModelDisplay.VIT_TINY,
    PretrainedModel.MIXNET_S: PretrainedModelDisplay.MIXNET_S,
    PretrainedModel.MIXNET_M: PretrainedModelDisplay.MIXNET_M,
    PretrainedModel.MIXNET_L: PretrainedModelDisplay.MIXNET_L,
    PretrainedModel.YOLOX_S: PretrainedModelDisplay.YOLOX_S,
    PretrainedModel.YOLOX_M: PretrainedModelDisplay.YOLOX_M,
    PretrainedModel.YOLOX_L: PretrainedModelDisplay.YOLOX_L,
    PretrainedModel.YOLOX_X: PretrainedModelDisplay.YOLOX_X,
    PretrainedModel.YOLO_FASTEST_V2: PretrainedModelDisplay.YOLO_FASTEST_V2,
    PretrainedModel.SEGFORMER_B0: PretrainedModelDisplay.SEGFORMER_B0,
    PretrainedModel.PIDNET_S: PretrainedModelDisplay.PIDNET_S,
    PretrainedModel.YOLOV9_S: PretrainedModelDisplay.YOLOV9_S,
    PretrainedModel.YOLOV9_M: PretrainedModelDisplay.YOLOV9_M,
    PretrainedModel.YOLOV9_C: PretrainedModelDisplay.YOLOV9_C,
}


MODEL_GROUP_MAP = {
    PretrainedModel.RESNET18: PretrainedModelGroup.RESNET,
    PretrainedModel.RESNET34: PretrainedModelGroup.RESNET,
    PretrainedModel.RESNET50: PretrainedModelGroup.RESNET,
    PretrainedModel.MOBILENET_V3_SMALL: PretrainedModelGroup.MOBILENET,
    PretrainedModel.MOBILENET_V3_LARGE: PretrainedModelGroup.MOBILENET,
    PretrainedModel.MOBILEVIT_S: PretrainedModelGroup.MOBILEVIT,
    PretrainedModel.EFFICIENTFORMER_L1: PretrainedModelGroup.EFFICIENTFORMER,
    PretrainedModel.VIT_TINY: PretrainedModelGroup.VIT,
    PretrainedModel.MIXNET_S: PretrainedModelGroup.MIXNET,
    PretrainedModel.MIXNET_M: PretrainedModelGroup.MIXNET,
    PretrainedModel.MIXNET_L: PretrainedModelGroup.MIXNET,
    PretrainedModel.YOLOX_S: PretrainedModelGroup.YOLOX,
    PretrainedModel.YOLOX_M: PretrainedModelGroup.YOLOX,
    PretrainedModel.YOLOX_L: PretrainedModelGroup.YOLOX,
    PretrainedModel.YOLOX_X: PretrainedModelGroup.YOLOX,
    PretrainedModel.YOLO_FASTEST_V2: PretrainedModelGroup.YOLO,
    PretrainedModel.SEGFORMER_B0: PretrainedModelGroup.SEGFORMER,
    PretrainedModel.PIDNET_S: PretrainedModelGroup.PIDNET,
    PretrainedModel.YOLOV9_S: PretrainedModelGroup.YOLOV9,
    PretrainedModel.YOLOV9_M: PretrainedModelGroup.YOLOV9,
    PretrainedModel.YOLOV9_C: PretrainedModelGroup.YOLOV9,
}
