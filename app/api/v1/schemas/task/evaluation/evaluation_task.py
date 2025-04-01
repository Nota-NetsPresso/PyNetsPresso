from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem
from netspresso.enums.conversion import PrecisionForConversion, TargetFramework
from netspresso.enums.device import DeviceName, SoftwareVersion


class EvaluationCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    dataset_id: str = Field(description="Dataset ID")

    framework: TargetFramework = Field(description="Framework name")
    device_name: DeviceName = Field(description="Device name")
    software_version: Optional[SoftwareVersion] = Field(default=None, description="Software version")
    precision: PrecisionForConversion = Field(description="Precision")

    training_task_id: str = Field(description="Training task ID")


class EvaluationCreatePayload(BaseModel):
    task_id: str


class EvaluationPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    dataset_id: str
    is_dataset_deleted: bool
    metric_unit: str
    metric_value: float
    results_path: str

    input_model_id: str
    training_task_id: str
    conversion_task_id: str

    status: str
    error_detail: Optional[Dict] = None
    is_deleted: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class EvaluationCreateResponse(ResponseItem):
    data: EvaluationCreatePayload


class EvaluationResponse(ResponseItem):
    data: EvaluationPayload
