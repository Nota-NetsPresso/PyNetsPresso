from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.api.v1.schemas.base import ResponseListItems
from netspresso.enums.conversion import (
    PRECISION_FOR_BENCHMARK_DISPLAY_MAP,
    PRECISION_FOR_CONVERSION_DISPLAY_MAP,
    TARGET_FRAMEWORK_DISPLAY_MAP,
    PrecisionForBenchmark,
    PrecisionForBenchmarkDisplay,
    PrecisionForConversion,
    PrecisionForConversionDisplay,
    TargetFramework,
    TargetFrameworkDisplay,
)
from netspresso.enums.device import (
    DEVICE_BRAND_MAP,
    DEVICE_DISPLAY_MAP,
    HARDWARE_TYPE_DISPLAY_MAP,
    SOFTWARE_VERSION_DISPLAY_MAP,
    DeviceBrand,
    DeviceDisplay,
    DeviceName,
    HardwareType,
    HardwareTypeDisplay,
    SoftwareVersion,
    SoftwareVersionDisplay,
)


class SoftwareVersionPayload(BaseModel):
    name: SoftwareVersion
    display_name: Optional[SoftwareVersionDisplay] = Field(default=None, description="Software version display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = SOFTWARE_VERSION_DISPLAY_MAP.get(self.name)

        return self


class PrecisionForConversionPayload(BaseModel):
    name: PrecisionForConversion
    display_name: Optional[PrecisionForConversionDisplay] = Field(default=None, description="Precision display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = PRECISION_FOR_CONVERSION_DISPLAY_MAP.get(self.name)

        return self


class PrecisionForBenchmarkPayload(BaseModel):
    name: PrecisionForBenchmark
    display_name: Optional[PrecisionForBenchmarkDisplay] = Field(default=None, description="Precision display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = PRECISION_FOR_BENCHMARK_DISPLAY_MAP.get(self.name)

        return self


class HardwareTypePayload(BaseModel):
    name: HardwareType
    display_name: Optional[HardwareTypeDisplay] = Field(default=None, description="Hardware type display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = HARDWARE_TYPE_DISPLAY_MAP.get(self.name)

        return self


class SupportedDevicePayload(BaseModel):
    name: DeviceName
    display_name: Optional[DeviceDisplay] = Field(default=None, description="Device display name")
    brand_name: Optional[DeviceBrand] = Field(default=None, description="Device brand name")
    software_versions: List[SoftwareVersionPayload]
    precisions: List[PrecisionForConversionPayload]
    hardware_types: List[HardwareTypePayload]

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = DEVICE_DISPLAY_MAP.get(self.name)
        self.brand_name = DEVICE_BRAND_MAP.get(self.name)

        return self


class TargetDevicePayload(BaseModel):
    name: DeviceName
    display_name: Optional[DeviceDisplay] = Field(default=None, description="Device display name")
    brand_name: Optional[DeviceBrand] = Field(default=None, description="Device brand name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = DEVICE_DISPLAY_MAP.get(self.name)
        self.brand_name = DEVICE_BRAND_MAP.get(self.name)

        return self


class TargetFrameworkPayload(BaseModel):
    name: TargetFramework = Field(description="Framework name")
    display_name: Optional[TargetFrameworkDisplay] = Field(default=None, description="Framework display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = TARGET_FRAMEWORK_DISPLAY_MAP.get(self.name)

        return self


class SupportedDeviceResponse(BaseModel):
    framework: TargetFrameworkPayload
    devices: List[SupportedDevicePayload]


class SupportedDevicesResponse(ResponseListItems):
    data: List[SupportedDeviceResponse]
