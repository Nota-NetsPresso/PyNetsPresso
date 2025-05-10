from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
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
    dataset_name: str = "Traffic Sign"
    is_dataset_deleted: bool

    confidence_score: float
    metrics: Dict
    metrics_names: List[str]
    primary_metric: str
    results_path: str

    input_model_id: str
    training_task_id: str
    conversion_task_id: str
    user_id: str

    status: str
    error_detail: Optional[Dict] = None
    is_deleted: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BoundingBoxCoordinates(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    x1: int
    y1: int
    x2: int
    y2: int


class BoundingBox(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    class_id: int = Field(alias="class")
    name: str
    confidence_score: float
    coords: BoundingBoxCoordinates


class PredictionForThreshold(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    threshold: float
    bboxes: List[BoundingBox]


class ImagePrediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    image_id: str  # 이미지 식별자 (파일명 또는 고유 ID)
    image_url: str  # 이미지의 URL
    predictions: List[PredictionForThreshold]  # 여러 threshold에 대한 예측 결과


class EvaluationResultsPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: str
    dataset_id: str
    results: List[ImagePrediction]


class EvaluationCreateResponse(ResponseItem):
    data: EvaluationCreatePayload


class EvaluationResponse(ResponseItem):
    data: EvaluationPayload


class EvaluationsResponse(ResponsePaginationItems):
    data: List[EvaluationPayload]


class EvaluationResultsResponse(ResponseItem):
    data: EvaluationResultsPayload


class EvaluationDatasetsPayload(BaseModel):
    model_id: str
    dataset_ids: List[str]


class EvaluationDatasetsResponse(ResponseItem):
    data: EvaluationDatasetsPayload
