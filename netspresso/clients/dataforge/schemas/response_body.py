from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class S3Path(BaseModel):
    """S3 path information schema"""
    bucket_name: str
    data_path: str
    id_mapping_path: str
    metadata_path: str


class DatasetInfo(BaseModel):
    """Dataset information schema"""
    dataset_id: str
    dataset_created: datetime
    dataset_hash: str
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    project_id: str
    dataset_type: str
    dataset_class_count: str
    dataset_data_count: str
    dataset_title: str
    dataset_uuid: str
    dataset_creator_id: str
    mime_type: str


class DatasetPayload(BaseModel):
    dataset: DatasetInfo = Field(..., description="Dataset information")
    s3_path: S3Path = Field(None, description="S3 path information")


class DatasetResponse(BaseModel):
    """Dataset API response schema"""
    data: DatasetPayload


class DatasetsPayload(BaseModel):
    datasets: List[DatasetInfo] = Field(..., description="Dataset information")


class DatasetsResponse(BaseModel):
    """Datasets API response schema"""
    data: DatasetsPayload
