from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.task.train.hyperparameter import (
    SupportedModelResponse,
    SupportedOptimizersResponse,
    SupportedSchedulersResponse,
)
from app.api.v1.schemas.task.train.train_task import TrainingCreate, TrainingResponse
from app.services.training_task import train_task_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get(
    "/trainings/configuration/models",
    response_model=SupportedModelResponse,
    description="Get supported models for training tasks.",
)
def get_supported_models() -> SupportedModelResponse:
    supported_models = train_task_service.get_supported_models()

    return SupportedModelResponse(data=supported_models)


@router.get(
    "/trainings/configuration/optimizers",
    response_model=SupportedOptimizersResponse,
    description="Get supported optimizers for training tasks.",
)
def get_supported_optimizers() -> SupportedOptimizersResponse:
    supported_optimizers = train_task_service.get_supported_optimizers()

    return SupportedOptimizersResponse(data=supported_optimizers)


@router.get(
    "/trainings/configuration/schedulers",
    response_model=SupportedSchedulersResponse,
    description="Get supported schedulers for training tasks.",
)
def get_supported_schedulers() -> SupportedSchedulersResponse:
    supported_schedulers = train_task_service.get_supported_schedulers()

    return SupportedSchedulersResponse(data=supported_schedulers)


@router.post("/trainings", response_model=TrainingResponse)
def create_training_task(
    request_body: TrainingCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainingResponse:
    training_task = train_task_service.create_training_task(db=db, training_in=request_body, api_key=api_key)

    return TrainingResponse(data=training_task)


@router.get("/trainings/{task_id}", response_model=TrainingResponse)
def get_training_task(
    *,
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> TrainingResponse:
    training_task = train_task_service.get_training_task(db=db, task_id=task_id, api_key=api_key)

    return TrainingResponse(data=training_task)
