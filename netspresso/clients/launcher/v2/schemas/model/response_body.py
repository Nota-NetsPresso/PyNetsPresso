from dataclasses import dataclass, field
from typing import List, Optional

from netspresso.clients.launcher.v2.schemas.common import (
    ModelOption,
    ResponseItem,
    ResponseItems,
    ResponsePaginationItems,
)
from netspresso.clients.launcher.v2.schemas.model import ModelBase, ModelStatus


@dataclass
class ModelUploadUrlData:
    """ """

    ai_model_id: str
    presigned_upload_url: str


@dataclass
class ResponseModelUploadUrl(ResponseItem):
    """ """

    data: Optional[ModelUploadUrlData] = field(default_factory=dict)

    def __post_init__(self):
        self.data = ModelUploadUrlData(**self.data)


@dataclass
class ResponseModelStatus(ResponseItem):
    """ """

    data: ModelStatus = field(default_factory=ModelStatus)

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = ModelStatus(**self.data)


@dataclass
class ResponseModelItem(ResponseItem):
    """ """

    data: ModelBase = field(default_factory=ModelBase)

    def __post_init__(self):
        self.data = ModelBase(**self.data)


@dataclass
class ResponseModelOptions(ResponseItems):
    """ """

    data: Optional[List[ModelOption]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ModelOption(**option) for option in self.data]


@dataclass
class ResponseModelItems(ResponsePaginationItems):
    """ """

    data: Optional[List[ModelBase]] = field(default_factory=list)

    def __post_init__(self):
        self.data = [ModelBase(**model) for model in self.data]
