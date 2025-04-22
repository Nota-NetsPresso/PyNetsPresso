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


class BenchmarkResultPayload(BaseModel):
    memory_footprint_gpu: float = Field(default=0, description="Memory footprint of the device in GPU")
    memory_footprint_cpu: float = Field(default=0, description="Memory footprint of the device in CPU")
    power_consumption: float = Field(default=0, description="Power consumption of the device")
    ram_size: float = Field(default=0, description="RAM size of the device")
    latency: float = Field(default=0, description="Latency of the device")
    file_size: float = Field(default=0, description="File size of the device")


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


class SupportedDeviceForBenchmarkPayload(BaseModel):
    input_model_id: str
    display_name: Optional[str] = Field(default=None, description="Device display name")
    name: DeviceName
    brand_name: Optional[DeviceBrand] = Field(default=None, description="Device brand name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version of the device")
    data_type: Optional[PrecisionForBenchmark] = Field(default=None, description="Data type supported by the device")
    hardware_type: Optional[HardwareType] = Field(default=None, description="Hardware type of the device")

    @model_validator(mode="after")
    def set_display_name(self):
        # Get base device name from map
        base_name = DEVICE_DISPLAY_MAP.get(self.name, str(self.name))

        # Build display name parts
        display_parts = [base_name]

        # Add software version if exists
        if self.software_version:
            display_parts.append(str(self.software_version))

        # Add data type if exists
        if self.data_type:
            display_parts.append(f"({self.data_type})")

        # Add "with helium" if hardware type exists
        if self.hardware_type:
            display_hardware_type = HARDWARE_TYPE_DISPLAY_MAP.get(self.hardware_type)
            display_parts.append(f"with {display_hardware_type}")

        # Join all parts with spaces
        self.display_name = " ".join(display_parts)

        return self

    @model_validator(mode="after")
    def set_brand_name(self):
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


class SupportedDeviceForBenchmarkResponse(BaseModel):
    data: SupportedDeviceForBenchmarkPayload


class SupportedDevicesForBenchmarkResponse(ResponseListItems):
    data: List[SupportedDeviceForBenchmarkPayload]
