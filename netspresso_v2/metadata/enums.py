from enum import Enum


class TaskType(str, Enum):
    TRAIN = "train"
    COMPRESS = "compress"
    CONVERT = "convert"
    BENCHMARK = "benchmark"


class Status(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class DataType(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


class Framework(str, Enum):
    TENSORFLOW_KERAS = "tensorflow_keras"
    TENSORFLOW = "saved_model"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TENSORFLOW_LITE = "tensorflow_lite"
    DRPAI = "drpai"


class DeviceName(str, Enum):
    RASPBERRY_PI_5 = "RaspberryPi5"
    RASPBERRY_PI_4B = "RaspberryPi4B"
    RASPBERRY_PI_3B_PLUS = "RaspberryPi3BPlus"
    RASPBERRY_PI_ZERO_W = "RaspberryPi-ZeroW"
    RASPBERRY_PI_ZERO_2W = "RaspberryPi-Zero2W"
    RENESAS_RZ_V2L = "rzv2l_avnet"
    RENESAS_RZ_V2M = "rzv2m"
    JETSON_NANO = "Jetson-Nano"
    JETSON_TX2 = "Jetson-Tx2"
    JETSON_XAVIER = "Jetson-Xavier"
    JETSON_NX = "Jetson-Nx"
    JETSON_AGX_ORIN = "Jetson-AGX-Orin"
    JETSON_ORIN_NANO = "Jetson-Orin-Nano"
    AWS_T4 = "AWS-T4"
    INTEL_XEON_W_2233 = "Intel-Xeon"
    ALIF_ENSEMBLE_E7_DEVKIT_GEN2 = "Ensemble-E7-DevKit-Gen2"
    RENESAS_RA8D1 = "Renesas-RA8D1"
    ARM_ETHOS_U_SERIES = "Arm Virtual Hardware Ethos-U Series"


class SoftwareVersion(str, Enum):
    JETPACK_4_4_1 = "4.4.1-b50"
    JETPACK_4_6 = "4.6-b199"
    JETPACK_5_0_1 = "5.0.1-b118"
    JETPACK_5_0_2 = "5.0.2-b231"
    JETPACK_6_0 = "6.0-b52"


class HardwareType(str, Enum):
    HELIUM = "helium"
