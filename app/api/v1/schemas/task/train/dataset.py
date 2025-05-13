from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from netspresso.enums.train import StorageLocation


class DatasetCreate(BaseModel):
    train_path: str
    valid_path: Optional[str] = None
    test_path: Optional[str] = None


class DatasetSplitPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    name: str
    path: str
    storage_location: StorageLocation
    split_type: str
    count: int
    dataset_id: Optional[int] = None


class DatasetPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    name: Optional[str] = None
    task_type: Optional[str] = None
    mime_type: Optional[str] = "image"
    class_count: Optional[int] = None
    id_mapping: Optional[List] = []
    palette: Optional[Dict] = {}
    valid_split_ratio: Optional[float] = 0.1
    random_seed: Optional[int] = 0

    train_split: Optional[DatasetSplitPayload] = None
    test_split: Optional[DatasetSplitPayload] = None
