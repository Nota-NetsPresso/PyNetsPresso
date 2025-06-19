from dataclasses import dataclass, field
from typing import List

from netspresso.clients.launcher.v2.schemas import InputLayer
from netspresso.enums.graph_optimize import GraphOptimizePatternHandler
from netspresso.enums.metadata import TaskType
from netspresso.enums.task import TaskStatusForDisplay
from netspresso.metadata.common import BaseMetadata, ModelInfo


@dataclass
class GraphOptimizeInfo:
    graph_optimize_task_id: str = ""
    input_model_id: str = ""
    user_id: str = ""
    output_model_id: str = ""
    model_file_name: str = ""
    pattern_handlers: List[GraphOptimizePatternHandler] = field(default_factory=list)
    status: TaskStatusForDisplay = ""
    error_log: str = ""
    input_layer: List[InputLayer] = field(default_factory=list)


@dataclass
class GraphOptimizerMetadata(BaseMetadata):
    task_type: TaskType = TaskType.GRAPH_OPTIMIZE
    input_model_path: str = ""
    optimized_model_path: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    graph_optimize_task_info: GraphOptimizeInfo = field(default_factory=GraphOptimizeInfo)
