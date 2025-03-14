from enum import Enum
from typing import Literal


class DeviceBrand(str, Enum):
    NVIDIA = "NVIDIA"
    ARM = "Arm"
    RASPBERRY_PI = "RaspberryPi"
    SAMSUNG = "Samsung"
    ST_MICROELECTRONICS = "STMicroelectronics"
    INTEL = "Intel"
    RENESAS = "Renesas"
    NXP = "NXP"


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


class DeviceDisplay(str, Enum):
    RASPBERRY_PI_5 = "Raspberry Pi 5 (Arm Cortex-A76)"
    RASPBERRY_PI_4B = "Raspberry Pi 4B"
    RASPBERRY_PI_3B_PLUS = "Raspberry Pi 3B+"
    RASPBERRY_PI_3B = "Raspberry Pi 3B"
    RASPBERRY_PI_2B = "Raspberry Pi 2B"
    RASPBERRY_PI_ZERO_W = "Raspberry Pi Zero W"
    RASPBERRY_PI_ZERO_2W = "Raspberry Pi Zero 2 W"
    RENESAS_RZ_V2L = "Renesas RZ/V2L"
    RENESAS_RZ_V2M = "Renesas RZ/V2M"
    RENESAS_RA8D1 = "Renesas RA8D1 (Arm Cortex-M85)"

    JETSON_NANO = "NVIDIA Jetson Nano"
    JETSON_TX2 = "NVIDIA Jetson TX2"
    JETSON_XAVIER = "NVIDIA Jetson Xavier"
    JETSON_NX = "NVIDIA Jetson Xavier NX"
    JETSON_AGX_ORIN = "NVIDIA Jetson AGX Orin"
    JETSON_ORIN_NANO = "NVIDIA Jetson Orin Nano"
    AWS_T4 = "NVIDIA AWS T4"
    INTEL_XEON_W_2233 = "Intel Xeon W-2233"
    ALIF_ENSEMBLE_E7_DEVKIT_GEN2 = "Alif Ensemble DevKit-E7 Gen2 (Arm Cortex-M55+Ethos-U55)"

    ARM_ETHOS_U_SERIES = "Arm Virtual Hardware Corstone-300 (Ethos-U55/U65)"
    NXP_iMX93 = "NXP i.MX 93(Arm Cortex-A55/M33+Ethos-U65)"
    ARDUINO_NICLA_VISION = "Arduino Nicla Vision(Arm Cortex-M7/M4)"


DEVICE_DISPLAY_MAP = {
    DeviceName.RASPBERRY_PI_5: DeviceDisplay.RASPBERRY_PI_5,
    DeviceName.RASPBERRY_PI_4B: DeviceDisplay.RASPBERRY_PI_4B,
    DeviceName.RASPBERRY_PI_3B_PLUS: DeviceDisplay.RASPBERRY_PI_3B_PLUS,
    DeviceName.RASPBERRY_PI_3B: DeviceDisplay.RASPBERRY_PI_3B,
    DeviceName.RASPBERRY_PI_2B: DeviceDisplay.RASPBERRY_PI_2B,
    DeviceName.RASPBERRY_PI_ZERO_W: DeviceDisplay.RASPBERRY_PI_ZERO_W,
    DeviceName.RASPBERRY_PI_ZERO_2W: DeviceDisplay.RASPBERRY_PI_ZERO_2W,
    DeviceName.RENESAS_RZ_V2L: DeviceDisplay.RENESAS_RZ_V2L,
    DeviceName.RENESAS_RZ_V2M: DeviceDisplay.RENESAS_RZ_V2M,
    DeviceName.RENESAS_RA8D1: DeviceDisplay.RENESAS_RA8D1,
    DeviceName.JETSON_NANO: DeviceDisplay.JETSON_NANO,
    DeviceName.JETSON_TX2: DeviceDisplay.JETSON_TX2,
    DeviceName.JETSON_XAVIER: DeviceDisplay.JETSON_XAVIER,
    DeviceName.JETSON_NX: DeviceDisplay.JETSON_NX,
    DeviceName.JETSON_AGX_ORIN: DeviceDisplay.JETSON_AGX_ORIN,
    DeviceName.JETSON_ORIN_NANO: DeviceDisplay.JETSON_ORIN_NANO,
    DeviceName.AWS_T4: DeviceDisplay.AWS_T4,
    DeviceName.INTEL_XEON_W_2233: DeviceDisplay.INTEL_XEON_W_2233,
    DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2: DeviceDisplay.ALIF_ENSEMBLE_E7_DEVKIT_GEN2,
    DeviceName.ARM_ETHOS_U_SERIES: DeviceDisplay.ARM_ETHOS_U_SERIES,
    DeviceName.NXP_iMX93: DeviceDisplay.NXP_iMX93,
    DeviceName.ARDUINO_NICLA_VISION: DeviceDisplay.ARDUINO_NICLA_VISION,
}

DEVICE_BRAND_MAP = {
    DeviceName.RASPBERRY_PI_5: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_4B: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_3B_PLUS: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_3B: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_2B: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_ZERO_W: DeviceBrand.RASPBERRY_PI,
    DeviceName.RASPBERRY_PI_ZERO_2W: DeviceBrand.RASPBERRY_PI,
    DeviceName.RENESAS_RZ_V2L: DeviceBrand.RENESAS,
    DeviceName.RENESAS_RZ_V2M: DeviceBrand.RENESAS,
    DeviceName.JETSON_NANO: DeviceBrand.NVIDIA,
    DeviceName.JETSON_TX2: DeviceBrand.NVIDIA,
    DeviceName.JETSON_XAVIER: DeviceBrand.NVIDIA,
    DeviceName.JETSON_NX: DeviceBrand.NVIDIA,
    DeviceName.JETSON_AGX_ORIN: DeviceBrand.NVIDIA,
    DeviceName.JETSON_ORIN_NANO: DeviceBrand.NVIDIA,
    DeviceName.AWS_T4: DeviceBrand.NVIDIA,
    DeviceName.INTEL_XEON_W_2233: DeviceBrand.INTEL,
    DeviceName.RENESAS_RA8D1: DeviceBrand.ARM,
    DeviceName.ARDUINO_NICLA_VISION: DeviceBrand.ARM,
    DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2: DeviceBrand.ARM,
    DeviceName.ARM_ETHOS_U_SERIES: DeviceBrand.ARM,
    DeviceName.NXP_iMX93: DeviceBrand.NXP,
}


class SoftwareVersion(str, Enum):
    JETPACK_4_4_1 = "4.4.1-b50"
    JETPACK_4_6 = "4.6-b199"
    JETPACK_5_0_1 = "5.0.1-b118"
    JETPACK_5_0_2 = "5.0.2-b231"
    JETPACK_6_1 = "6.1+b123"

    @classmethod
    def create_literal(cls):
        return Literal["4.4.1-b50", "4.6-b199", "5.0.1-b118", "5.0.2-b231", "6.1+b123"]


class SoftwareVersionDisplay(str, Enum):
    JETPACK_4_4_1 = "Jetpack 4.4.1"
    JETPACK_4_6 = "Jetpack 4.6"
    JETPACK_5_0_1 = "Jetpack 5.0.1"
    JETPACK_5_0_2 = "Jetpack 5.0.2"
    JETPACK_6_1= "Jetpack 6.1"


SOFTWARE_VERSION_DISPLAY_MAP = {
    SoftwareVersion.JETPACK_4_4_1: SoftwareVersionDisplay.JETPACK_4_4_1,
    SoftwareVersion.JETPACK_4_6: SoftwareVersionDisplay.JETPACK_4_6,
    SoftwareVersion.JETPACK_5_0_1: SoftwareVersionDisplay.JETPACK_5_0_1,
    SoftwareVersion.JETPACK_5_0_2: SoftwareVersionDisplay.JETPACK_5_0_2,
    SoftwareVersion.JETPACK_6_1: SoftwareVersionDisplay.JETPACK_6_1,
}


class HardwareType(str, Enum):
    HELIUM = "helium"

    @classmethod
    def create_literal(cls):
        return Literal["helium"]


class HardwareTypeDisplay(str, Enum):
    HELIUM = "Helium"


HARDWARE_TYPE_DISPLAY_MAP = {
    HardwareType.HELIUM: HardwareTypeDisplay.HELIUM,
}


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
