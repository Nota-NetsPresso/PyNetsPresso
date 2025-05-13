import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class S3Path:
    """S3 경로 정보 스키마"""
    bucket_name: str
    data_path: str
    id_mapping_path: str
    metadata_path: str


@dataclass
class ProjectSummary:
    """프로젝트 요약 정보 스키마"""
    project_name: str
    storage_type: Optional[str] = None
    project_created: Optional[str] = None
    project_updated: Optional[str] = None
    project_desc: Optional[str] = None
    thumbnail: Optional[str] = None
    ext_project_id: Optional[str] = None
    project_id: Optional[str] = None


@dataclass
class DatasetInfo:
    """데이터셋 정보 스키마"""
    dataset_class_count: Optional[int] = None
    dataset_title: Optional[str] = None
    mime_type: Optional[str] = None
    task_type: Optional[str] = None
    dataset_uuid: Optional[str] = None
    project_summary: Optional[ProjectSummary] = None

    def __post_init__(self):
        if hasattr(self, 'project_summary') and isinstance(self.project_summary, dict):
            self.project_summary = ProjectSummary(**self.project_summary)


@dataclass
class DatasetMetadata:
    """데이터셋 메타데이터 스키마"""
    dataset_bucket_name: Optional[str] = None
    dataset_uuid: Optional[str] = None
    dataset_type: Optional[str] = None
    dataset_hash: Optional[str] = None
    id_mapping: List[str] = field(default_factory=list)
    csv_s3_path: Optional[str] = None
    id_mapping_s3_path: Optional[str] = None

    def __post_init__(self):
        # id_mapping이 문자열 형태로 들어온 경우 처리
        if isinstance(self.id_mapping, str):
            try:
                import json
                self.id_mapping = json.loads(self.id_mapping)
            except Exception:
                self.id_mapping = []


@dataclass
class DatasetVersionInfo:
    """데이터셋 버전 정보 스키마"""
    dataset_uuid: str
    dataset_hash: str
    dataset_type: str
    dataset_data_count: int
    dataset_created: str
    dataset_metadata: Union[Dict[str, Any], DatasetMetadata] = field(default_factory=dict)
    user_uuid: Optional[str] = None
    origin_dataset_hash: Optional[str] = None
    project_id: Optional[str] = None

    def __post_init__(self):
        # 정수형 필드 타입 변환
        if isinstance(self.dataset_data_count, str):
            try:
                self.dataset_data_count = int(self.dataset_data_count)
            except (ValueError, TypeError):
                self.dataset_data_count = 0

        # 메타데이터가 Dict로 들어온 경우 DatasetMetadata 객체로 변환
        if isinstance(self.dataset_metadata, dict):
            self.dataset_metadata = DatasetMetadata(**self.dataset_metadata)


@dataclass
class DatasetVersionsPayload:
    """데이터셋 버전 목록 페이로드 스키마"""
    dataset_versions: List[DatasetVersionInfo] = field(default_factory=list)

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        if hasattr(self, 'dataset_versions'):
            self.dataset_versions = [DatasetVersionInfo(**version) for version in self.dataset_versions]


@dataclass
class DatasetPayload:
    """데이터셋 페이로드 스키마"""
    dataset: DatasetInfo

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        if hasattr(self, 'dataset') and isinstance(self.dataset, dict):
            self.dataset = DatasetInfo(**self.dataset)


@dataclass
class DatasetsPayload:
    """데이터셋 목록 페이로드 스키마"""
    datasets: List[DatasetInfo] = field(default_factory=list)

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        if hasattr(self, 'datasets'):
            self.datasets = [DatasetInfo(**dataset) for dataset in self.datasets]


@dataclass
class ApiResponse:
    """API 기본 응답 스키마"""
    code: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Any] = None

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass
class DatasetResponse(ApiResponse):
    """단일 데이터셋 API 응답 스키마"""
    success: bool = True
    data: Optional[DatasetPayload] = None

    def __post_init__(self):
        if isinstance(self.data, dict) and 'dataset' in self.data:
            self.data = DatasetPayload(**self.data)

        # 예전 방식 호환성 유지 - success 기준 검증
        if self.success and self.data is None and self.code is None:
            raise ValueError("success가 True인 경우 data는 필수입니다")


@dataclass
class DatasetsResponse(ApiResponse):
    """데이터셋 목록 API 응답 스키마"""
    success: bool = True
    data: Optional[DatasetsPayload] = None

    def __post_init__(self):
        if isinstance(self.data, dict) and 'datasets' in self.data:
            self.data = DatasetsPayload(**self.data)

        # 예전 방식 호환성 유지 - success 기준 검증
        if self.success and self.data is None and self.code is None:
            raise ValueError("success가 True인 경우 data는 필수입니다")


@dataclass
class DatasetVersionResponse(ApiResponse):
    """데이터셋 버전 API 응답 스키마"""
    data: Optional[DatasetVersionInfo] = None

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DatasetVersionInfo(**self.data)


@dataclass
class DatasetVersionsResponse(ApiResponse):
    """데이터셋 버전 목록 API 응답 스키마"""
    data: Optional[DatasetVersionsPayload] = None

    def __post_init__(self):
        if isinstance(self.data, dict) and 'dataset_versions' in self.data:
            self.data = DatasetVersionsPayload(**self.data)
