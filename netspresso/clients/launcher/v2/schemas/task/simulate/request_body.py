from dataclasses import dataclass


@dataclass
class RequestCreateSimulateTask:
    base_model_id: str
    target_model_id: str
    type: str
