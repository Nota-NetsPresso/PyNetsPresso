import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from netspresso.clients.launcher.v2.schemas import InputLayer, ResponseItem
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import TaskStatusForDisplay
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler
from netspresso.metadata.graph_optimizer import GraphOptimizeInfo


@dataclass
class GraphOptimizeTask:
    graph_optimize_task_id: str
    input_model_id: str
    user_id: str
    output_model_id: str
    model_file_name: str
    pattern_handlers: List[GraphOptimizePatternHandler]
    status: TaskStatusForDisplay = ""
    error_log: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)
    input_layer: InputLayer = field(default_factory=InputLayer)

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

        self.input_layer = InputLayer(**self.input_layer)

    def to(self, model_file_name: str) -> GraphOptimizeInfo:
        graph_optimize_info = GraphOptimizeInfo()
        graph_optimize_info.graph_optimize_task_id = self.graph_optimize_task_id
        graph_optimize_info.input_model_id = self.input_model_id
        graph_optimize_info.user_id = self.user_id
        graph_optimize_info.output_model_id = self.output_model_id
        graph_optimize_info.model_file_name = model_file_name
        graph_optimize_info.pattern_handlers = self.pattern_handlers
        graph_optimize_info.status = self.status
        graph_optimize_info.error_log = self.error_log
        graph_optimize_info.input_layer = self.input_layer

        return graph_optimize_info


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
