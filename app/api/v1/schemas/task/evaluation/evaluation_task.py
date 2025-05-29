from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from app.api.v1.schemas.task.conversion.conversion_task import ConversionForEvaluationCreate
from app.api.v1.schemas.task.train.dataset import EvaluationDatasetPayload
from netspresso.enums.conversion import EvaluationTargetFramework
from netspresso.enums.train import StorageLocation


class EvaluationCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")

    dataset_path: str = Field(description="Dataset path")
    storage_location: StorageLocation = Field(description="Storage location")

    conversion: ConversionForEvaluationCreate

    training_task_id: str = Field(description="Training task ID")

    @model_validator(mode='after')
    def validate_device_fields(self) -> 'EvaluationCreate':
        framework = self.conversion.framework
        device_name = self.conversion.device_name
        precision = self.conversion.precision

        if framework != EvaluationTargetFramework.ONNX:
            if device_name is None:
                raise ValueError("device_name is required for non-ONNX frameworks")
            if precision is None:
                raise ValueError("precision is required for non-ONNX frameworks")

        return self


class EvaluationCreatePayload(BaseModel):
    task_id: str


class EvaluationPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    dataset_id: str
    is_dataset_deleted: bool

    confidence_score: float
    metrics: Optional[Dict] = None
    metrics_names: Optional[List[str]] = None
    primary_metric: Optional[str] = None
    results_path: Optional[str] = None

    input_model_id: str
    training_task_id: str
    conversion_task_id: Optional[str] = None
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
    result_count: int = 0
    total_count: int = 0


class EvaluationCreateResponse(ResponseItem):
    data: EvaluationCreatePayload


class EvaluationResponse(ResponseItem):
    data: EvaluationPayload


class EvaluationsResponse(ResponsePaginationItems):
    data: List[EvaluationPayload]


class EvaluationResultsResponse(ResponseItem):
    data: EvaluationResultsPayload


class EvaluationDatasetPayload2(BaseModel):
    dataset_id: str
    dataset_name: str = "Traffic Sign"
    dataset_type: str = "detection"


class EvaluationDatasetsPayload(BaseModel):
    model_id: str
    datasets: List[EvaluationDatasetPayload]


class EvaluationDatasetsResponse(ResponseItem):
    data: EvaluationDatasetsPayload
