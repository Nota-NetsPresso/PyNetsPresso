import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseItem
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import TaskStatusForDisplay
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler


@dataclass
class GraphOptimizeTask:
    graph_optimize_task_id: str
    status: TaskStatusForDisplay = ""
    input_model_id: str
    user_id: str
    output_model_id: str
    model_file_name: str
    error_log: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    pattern_handlers: List[GraphOptimizePatternHandler]
    input_layers: List[InputLayer] = field(default_factory=[])

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.input_layers = [InputLayer(**input_layers) for input_layers in self.input_layers]


@dataclass
class ResponseGraphOptimizeTaskItem(ResponseItem):
    data: Optional[GraphOptimizeTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = GraphOptimizeTask(**self.data)


@dataclass
class ResponseGraphOptimizeStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)


@dataclass
class DownloadModelUrl:
    ai_model_id: str
    presigned_download_url: str


@dataclass
class ResponseGraphOptimizeDownloadModelUrlItem(ResponseItem):
    data: Optional[DownloadModelUrl] = field(default_factory=dict)

    def __post_init__(self):
        self.data = DownloadModelUrl(**self.data)
