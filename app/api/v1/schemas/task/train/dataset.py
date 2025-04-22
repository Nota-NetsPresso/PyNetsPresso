from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from netspresso.enums.train import StorageLocation


class DatasetCreate(BaseModel):
    train_path: str
    valid_path: Optional[str] = None
    test_path: Optional[str] = None


class DatasetPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    train_path: str
    valid_path: Optional[str]
    test_path: Optional[str]
    storage_location: StorageLocation
    id_mapping: Optional[List] = []
    palette: Optional[dict] = {}
