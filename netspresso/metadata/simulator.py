from dataclasses import dataclass, field
from typing import List

from netspresso.enums.metadata import TaskType
from netspresso.enums.simulate import SimulateTaskType
from netspresso.enums.task import TaskStatusForDisplay
from netspresso.metadata.common import BaseMetadata, ModelInfo


@dataclass
class SimulatorInfo:
    simulate_task_id: str = ""
    user_id: str = ""
    base_model_id: str = ""
    target_model_id: str = ""
    snr_scores: List[dict] = field(default_factory=list)
    type: SimulateTaskType = ""
    status: TaskStatusForDisplay = ""
    error_log: str = ""


@dataclass
class SimulatorMetadata(BaseMetadata):
    task_type: TaskType = TaskType.SIMULATE
    base_model_path: str = ""
    target_model_path: str = ""
    base_model_info: ModelInfo = field(default_factory=ModelInfo)
    target_model_info: ModelInfo = field(default_factory=ModelInfo)
    simulate_task_info: SimulatorInfo = field(default_factory=SimulatorInfo)
