from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from app.api.v1.schemas.base import ResponseItem, ResponseListItems
from netspresso.enums.train import StorageLocation


class DatasetCreate(BaseModel):
    train_path: str
    test_path: Optional[str] = None
    storage_location: StorageLocation


class DatasetBasePayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset_id: Optional[str] = None
    name: Optional[str] = None
    path: Optional[str] = None
    id_mapping: Optional[List] = []
    palette: Optional[Dict] = {}
    task_type: Optional[str] = None
    mime_type: Optional[str] = "image"
    class_count: Optional[int] = None
    count: Optional[int] = None

    storage_location: StorageLocation
    storage_info: Optional[Dict] = {}


class TrainingDatasetPayload(DatasetBasePayload):
    model_config = ConfigDict(from_attributes=True)

    valid_split_ratio: Optional[float] = 0.1
    random_seed: Optional[int] = 0


class EvaluationDatasetPayload(DatasetBasePayload):
    pass


class LocalTrainingDatasetPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    path: str


class LocalTrainingDatasetsResponse(ResponseListItems):
    data: List[LocalTrainingDatasetPayload]
