from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from netspresso.enums.train import (
    OPTIMIZER_DISPLAY_MAP,
    SCHEDULER_DISPLAY_MAP,
    Optimizer,
    OptimizerDisplay,
    Scheduler,
    SchedulerDisplay,
)


class TrainerModel(BaseModel):
    name: str = Field(..., description="Name of the model")
    display_name: Optional[str] = Field(..., description="Display name of the model")
    group_name: Optional[str] = Field(..., description="Group name of the model")


class SupportedModel(BaseModel):
    classification: List[TrainerModel] = Field(..., description="Supported models for classification tasks")
    detection: List[TrainerModel] = Field(..., description="Supported models for object detection tasks")
    segmentation: List[TrainerModel] = Field(..., description="Supported models for semantic segmentation tasks")


class SupportedModelResponse(BaseModel):
    data: SupportedModel = Field(..., description="Supported models for classification tasks")


class OptimizerPayload(BaseModel):
    name: Optimizer = Field(description="Optimizer name")
    display_name: Optional[OptimizerDisplay] = Field(default=None, description="Optimizer display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = OPTIMIZER_DISPLAY_MAP.get(self.name)
        return self


class SupportedOptimizersResponse(BaseModel):
    data: List[OptimizerPayload] = Field(..., description="Supported optimizers for training tasks")


class SchedulerPayload(BaseModel):
    name: Scheduler = Field(description="Scheduler name")
    display_name: Optional[SchedulerDisplay] = Field(default=None, description="Scheduler display name")

    @model_validator(mode="after")
    def set_display_name(self) -> str:
        self.display_name = SCHEDULER_DISPLAY_MAP.get(self.name)
        return self


class SupportedSchedulersResponse(BaseModel):
    data: List[SchedulerPayload] = Field(..., description="Supported schedulers for training tasks")


class HyperparameterCreate(BaseModel):
    epochs: int = Field(default=10, description="Number of epochs to train for")
    batch_size: int = Field(default=32, description="Batch size to use")
    learning_rate: Optional[float] = Field(default=0.001, description="Learning rate to use")
    optimizer: Optimizer = Field(..., description="Optimizer to use")
    scheduler: Scheduler = Field(..., description="Scheduler to use")


class HyperparameterPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerPayload = Field(..., description="Optimizer to use")
    scheduler: SchedulerPayload = Field(..., description="Scheduler to use")
