from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem
from app.api.v1.schemas.device import (
    PrecisionForConversionPayload,
    SoftwareVersionPayload,
    TargetDevicePayload,
    TargetFrameworkPayload,
)
from netspresso.enums.conversion import Precision, TargetFramework
from netspresso.enums.device import DeviceName, SoftwareVersion


class ConversionCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    framework: TargetFramework = Field(description="Framework name")
    device_name: DeviceName = Field(description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    precision: Precision = Field(description="Precision")
    calibration_dataset_path: Optional[str] = Field(default=None, description="Path to the calibration dataset")


class ConversionPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    framework: TargetFrameworkPayload
    device: TargetDevicePayload
    software_version: Optional[SoftwareVersionPayload] = None
    precision: PrecisionForConversionPayload
    status: str
    is_deleted: bool
    error_detail: Optional[Dict] = None
    input_model_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversionCreatePayload(BaseModel):
    task_id: str


class ConversionCreateResponse(ResponseItem):
    data: ConversionCreatePayload


class ConversionResponse(ResponseItem):
    data: ConversionPayload
