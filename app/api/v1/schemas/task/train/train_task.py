from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.api.v1.schemas.base import ResponseItem
from netspresso.enums.train import (
    FRAMEWORK_DISPLAY_MAP,
    MODEL_DISPLAY_MAP,
    MODEL_GROUP_MAP,
    TASK_DISPLAY_MAP,
    Framework,
    FrameworkDisplay,
    PretrainedModel,
    PretrainedModelDisplay,
    PretrainedModelGroup,
    Task,
    TaskDisplay,
)

from .dataset import DatasetCreate, DatasetPayload
from .environment import EnvironmentCreate, EnvironmentPayload
from .hyperparameter import HyperparameterCreate, HyperparameterPayload
from .performance import PerformancePayload


class InputShape(BaseModel):
    batch: int = Field(default=1, description="Batch size")
    channel: int = Field(default=3, description="Number of channels")
    dimension: List[int] = Field(default=[224, 224], description="Input shape")


class TrainingCreate(BaseModel):
    project_id: str
    name: str
    pretrained_model: str
    task: Task = Field(default=Task.OBJECT_DETECTION, description="Task")
    input_shapes: List[InputShape] = Field(default_factory=list, description="List of input shapes")
    dataset: Optional[DatasetCreate]
    hyperparameter: Optional[HyperparameterCreate]
    environment: Optional[EnvironmentCreate]


class PretrainedModelPayload(BaseModel):
    name: PretrainedModel = Field(description="Pretrained model name")
    display_name: Optional[PretrainedModelDisplay] = Field(default=None, description="Pretrained model display name")
    group_name: Optional[PretrainedModelGroup] = Field(default=None, description="Pretrained model group name")

    @model_validator(mode="after")
    def set_display_and_group_name(self) -> str:
        self.display_name = MODEL_DISPLAY_MAP.get(self.name)
        self.group_name = MODEL_GROUP_MAP.get(self.name)

        return self


class TaskPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: Task = Field(description="Task name")
    display_name: Optional[TaskDisplay] = Field(default=None, description="Task display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = TASK_DISPLAY_MAP.get(self.name)

        return self


class FrameworkPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: Framework = Field(description="Framework name")
    display_name: Optional[FrameworkDisplay] = Field(default=None, description="Framework display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = FRAMEWORK_DISPLAY_MAP.get(self.name)

        return self


class TrainingPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    pretrained_model: PretrainedModelPayload
    task: TaskPayload
    framework: FrameworkPayload
    input_shapes: List[Dict]
    status: str
    error_detail: Optional[Dict] = None
    dataset: Optional[DatasetPayload]
    hyperparameter: Optional[HyperparameterPayload]
    performance: Optional[PerformancePayload]
    environment: Optional[EnvironmentPayload]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TrainingResponse(ResponseItem):
    data: TrainingPayload
