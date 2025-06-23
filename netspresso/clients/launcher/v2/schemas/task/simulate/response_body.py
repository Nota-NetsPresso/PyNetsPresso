import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

from netspresso.clients.launcher.v2.schemas import ResponseItem
from netspresso.clients.launcher.v2.schemas.task.common import TaskStatusInfo
from netspresso.enums import TaskStatusForDisplay
from netspresso.enums.simulate import SimulateTaskType
from netspresso.metadata.simulator import SimulatorInfo


@dataclass
class SnrScore:
    layer_name: str
    snr_score: Union[float, str]
    sensitivity_score: float
    sensitivity_rank: Optional[int]


@dataclass
class SimulateTask:
    simulate_task_id: str
    user_id: str
    base_model_id: str
    target_model_id: str
    snr_scores: List[SnrScore]
    type: SimulateTaskType
    status: TaskStatusForDisplay = ""
    error_log: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    def to(self) -> SimulatorInfo:
        simulator_info = SimulatorInfo()
        simulator_info.simulate_task_id = self.simulate_task_id
        simulator_info.user_id = self.user_id
        simulator_info.base_model_id = self.base_model_id
        simulator_info.target_model_id = self.target_model_id
        simulator_info.snr_scores = self.snr_scores
        simulator_info.type = self.type
        simulator_info.status = self.status
        simulator_info.error_log = self.error_log

        return simulator_info


@dataclass
class ResponseSimulateTaskItem(ResponseItem):
    data: Optional[SimulateTask] = field(default_factory=dict)

    def __post_init__(self):
        self.data = SimulateTask(**self.data)


@dataclass
class ResponseSimulateStatusItem(ResponseItem):
    data: TaskStatusInfo = field(default_factory=TaskStatusInfo)

    def __post_init__(self):
        self.data = TaskStatusInfo(**self.data)
