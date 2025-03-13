from enum import Enum
from typing import Literal


class DeviceName(str, Enum):
    RASPBERRY_PI_5 = "RaspberryPi5"
    RASPBERRY_PI_4B = "RaspberryPi4B"
    RASPBERRY_PI_3B_PLUS = "RaspberryPi3BPlus"
    RASPBERRY_PI_3B = "RaspberryPi3B"
    RASPBERRY_PI_2B = "RaspberryPi2B"
    RASPBERRY_PI_ZERO_W = "RaspberryPi-ZeroW"
    RASPBERRY_PI_ZERO_2W = "RaspberryPi-Zero2W"
    RENESAS_RZ_V2L = "rzv2l_avnet"
    RENESAS_RZ_V2M = "rzv2m"
    RENESAS_RA8D1 = "Renesas-RA8D1"

    JETSON_NANO = "Jetson-Nano"
    JETSON_TX2 = "Jetson-Tx2"
    JETSON_XAVIER = "Jetson-Xavier"
    JETSON_NX = "Jetson-Nx"
    JETSON_AGX_ORIN = "Jetson-AGX-Orin"
    JETSON_ORIN_NANO = "Jetson-Orin-Nano"
    AWS_T4 = "AWS-T4"
    INTEL_XEON_W_2233 = "Intel-Xeon"
    ALIF_ENSEMBLE_E7_DEVKIT_GEN2 = "Ensemble-E7-DevKit-Gen2"

    ARM_ETHOS_U_SERIES = "Arm Virtual Hardware Ethos-U Series"
    NXP_iMX93 = "nxp_imx93_ethos_u65"
    ARDUINO_NICLA_VISION = "arduino_nicla_vision"

    @classmethod
    def create_literal(cls):
        return Literal[
            "RaspberryPi5",
            "RaspberryPi4B",
            "RaspberryPi3BPlus",
            "RaspberryPi3B",
            "RaspberryPi2B",
            "RaspberryPi-ZeroW",
            "RaspberryPi-Zero2W",
            "rzv2l_avnet",
            "rzv2m",
            "Renesas-RA8D1",
            "Jetson-Nano",
            "Jetson-Tx2",
            "Jetson-Xavier",
            "Jetson-Nx",
            "Jetson-AGX-Orin",
            "Jetson-Orin-Nano",
            "AWS-T4",
            "Intel-Xeon",
            "Ensemble-E7-DevKit-Gen2",
            "Arm Virtual Hardware Ethos-U Series",
            "nxp_imx93_ethos_u65",
            "arduino_nicla_vision",
        ]

    JETSON_DEVICES = [
        JETSON_NANO,
        JETSON_TX2,
        JETSON_XAVIER,
        JETSON_NX,
        JETSON_AGX_ORIN,
        JETSON_ORIN_NANO,
    ]
    RASPBERRY_PI_DEVICES = [
        RASPBERRY_PI_5,
        RASPBERRY_PI_4B,
        RASPBERRY_PI_3B,
        RASPBERRY_PI_2B,
        RASPBERRY_PI_3B_PLUS,
        RASPBERRY_PI_ZERO_W,
        RASPBERRY_PI_ZERO_2W,
    ]
    RENESAS_DEVICES = [RENESAS_RZ_V2L, RENESAS_RZ_V2M]
    NVIDIA_GRAPHIC_CARDS = [AWS_T4]
    INTEL_DEVICES = [INTEL_XEON_W_2233]
    AVAILABLE_INT8_DEVICES = [
        ALIF_ENSEMBLE_E7_DEVKIT_GEN2,
        RENESAS_RA8D1,
        RASPBERRY_PI_5,
        RASPBERRY_PI_4B,
        RASPBERRY_PI_3B_PLUS,
        RASPBERRY_PI_3B,
        RASPBERRY_PI_2B,
        RASPBERRY_PI_ZERO_W,
        RASPBERRY_PI_ZERO_2W,
        ARM_ETHOS_U_SERIES,
        NXP_iMX93,
        ARDUINO_NICLA_VISION,
    ]
    ONLY_INT8_DEVICES = [
        ALIF_ENSEMBLE_E7_DEVKIT_GEN2,
        RENESAS_RA8D1,
        ARM_ETHOS_U_SERIES,
        NXP_iMX93,
        ARDUINO_NICLA_VISION,
    ]


class SoftwareVersion(str, Enum):
    JETPACK_4_4_1 = "4.4.1-b50"
    JETPACK_4_6 = "4.6-b199"
    JETPACK_5_0_1 = "5.0.1-b118"
    JETPACK_5_0_2 = "5.0.2-b231"
    JETPACK_6_1 = "6.1+b123"

    @classmethod
    def create_literal(cls):
        return Literal["4.4.1-b50", "4.6-b199", "5.0.1-b118", "5.0.2-b231", "6.1+b123"]


class DisplaySoftwareVersion(str, Enum):
    JETPACK_4_4_1 = "Jetpack 4.4.1"
    JETPACK_4_6 = "Jetpack 4.6"
    JETPACK_5_0_1 = "Jetpack 5.0.1"
    JETPACK_5_0_2 = "Jetpack 5.0.2"
    JETPACK_6_1 = "Jetpack 6.1"


class HardwareType(str, Enum):
    HELIUM = "helium"

    @classmethod
    def create_literal(cls):
        return Literal["helium"]


class TaskStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    USER_CANCEL = "USER_CANCEL"

    @classmethod
    def create_literal(cls):
        return Literal["IN_QUEUE", "IN_PROGRESS", "FINISHED", "ERROR", "USER_CANCEL", "TIMEOUT"]


device_name_literal = DeviceName.create_literal()
