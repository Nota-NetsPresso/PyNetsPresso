from enum import Enum
from typing import Literal


class StrEnumBase(str, Enum):
    def __repr__(self) -> str:
        return str.__repr__(self.value)


class LauncherFunction(StrEnumBase):
    GENERAL = "GENERAL"
    CONVERT = "CONVERT"
    BENCHMARK = "BENCHMARK"

    @classmethod
    def create_literal(cls):
        return Literal[
            "GENERAL",
            "CONVERT",
            "BENCHMARK",
        ]


class DataType(StrEnumBase):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    NONE = ""

    @classmethod
    def create_literal(cls):
        return Literal[
            "FP32",
            "FP16",
            "INT8",
            "",
        ]


class ModelFramework(StrEnumBase):
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TENSORFLOW_LITE = "tensorflow_lite"
    DRPAI = "drpai"
    TENSORFLOW_KERAS = "keras"
    TENSORFLOW = "saved_model"

    @classmethod
    def create_literal(cls):
        return Literal[
            "onnx",
            "tensorrt",
            "openvino",
            "tensorflow_lite",
            "drpai",
            "keras",
            "saved_model",
        ]


class DeviceName(StrEnumBase):
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
    AWS_T4 = "AWS-T4"
    Intel_XEON_W_2233 = "Intel-Xeon"
    ALIF_ENSEMBLE_E7_DEVKIT_GEN2 = "Ensemble-E7-DevKit-Gen2"
    RENESAS_RA8D1 = "Renesas-RA8D1"

    @classmethod
    def create_literal(cls):
        return Literal[
            "RaspberryPi4B",
            "RaspberryPi3BPlus",
            "RaspberryPi-ZeroW",
            "RaspberryPi-Zero2W",
            "rzv2l_avnet",
            "rzv2m",
            "Jetson-Nano",
            "Jetson-Tx2",
            "Jetson-Xavier",
            "Jetson-Nx",
            "Jetson-AGX-Orin",
            "AWS-T4",
            "Intel-Xeon",
            "Ensemble-E7-DevKit-Gen2",
            "Renesas-RA8D1",
        ]


JETSON_DEVICES = [
    DeviceName.JETSON_NANO,
    DeviceName.JETSON_TX2,
    DeviceName.JETSON_XAVIER,
    DeviceName.JETSON_NX,
    DeviceName.JETSON_AGX_ORIN,
]
RASPBERRY_PI_DEVICES = [
    DeviceName.RASPBERRY_PI_4B,
    DeviceName.RASPBERRY_PI_3B_PLUS,
    DeviceName.RASPBERRY_PI_ZERO_W,
    DeviceName.RASPBERRY_PI_ZERO_2W,
]
RENESAS_DEVICES = [DeviceName.RENESAS_RZ_V2L, DeviceName.RENESAS_RZ_V2M]
NVIDIA_GRAPHIC_CARDS = [DeviceName.AWS_T4]
INTEL_DEVICES = [DeviceName.Intel_XEON_W_2233]
ONLY_INT8_DEVICES = [DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2, DeviceName.RENESAS_RA8D1]


class SoftwareVersion(StrEnumBase):
    JETPACK_4_4_1 = "4.4.1-b50"
    JETPACK_4_6 = "4.6-b199"
    JETPACK_5_0_1 = "5.0.1-b118"
    JETPACK_5_0_2 = "5.0.2-b231"

    @classmethod
    def create_literal(cls):
        return Literal[
            "4.4.1-b50",
            "4.6-b199",
            "5.0.1-b118",
            "5.0.2-b231",
        ]


class HardwareType(StrEnumBase):
    HELIUM = "helium"

    @classmethod
    def create_literal(cls):
        return Literal["helium"]


class TaskStatus(StrEnumBase):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    USER_CANCEL = "USER_CANCEL"

    @classmethod
    def create_literal(cls):
        return Literal["IN_QUEUE", "IN_PROGRESS", "FINISHED", "ERROR", "USER_CANCEL"]
