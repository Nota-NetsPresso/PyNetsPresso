from typing import Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.model import ModelDetailResponse, ModelsResponse, ModelUrlResponse
from app.api.v1.schemas.task.compression.compression_task import (
    CompressionsResponse,
)
from app.api.v1.schemas.task.evaluation.evaluation_task import EvaluationsResponse
from app.services.compression_task import compression_task_service
from app.services.evaluation_task import evaluation_task_service
from app.services.model import model_service
from netspresso.enums.task import TaskType
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get("", response_model=ModelsResponse)
def get_models(
    *,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
    task_type: Optional[TaskType] = Query(None, description="Filter models by task type"),
    project_id: Optional[str] = Query(None, description="Filter models by project ID"),
) -> ModelsResponse:
    models = model_service.get_models(db=db, api_key=api_key, task_type=task_type, project_id=project_id)

    return ModelsResponse(data=models, total_count=len(models))


@router.get("/{model_id}", response_model=ModelDetailResponse)
def get_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelDetailResponse:
    model = model_service.get_model(db=db, model_id=model_id, api_key=api_key)

    return ModelDetailResponse(data=model)


@router.delete("/{model_id}", response_model=ModelDetailResponse)
def delete_model(
    *,
    model_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelDetailResponse:
    model = model_service.delete_model(db=db, model_id=model_id, api_key=api_key)

    return ModelDetailResponse(data=model)


@router.get("/{model_id}/download")
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ModelUrlResponse:
    presigned_url = model_service.download_model(db=db, model_id=model_id, api_key=api_key)

    return ModelUrlResponse(data=presigned_url)


@router.get("/{model_id}/compressions", response_model=CompressionsResponse)
def get_model_compression_tasks(
    model_id: str = Path(..., description="Model ID to get all related compression tasks"),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionsResponse:
    compression_tasks = compression_task_service.get_compression_tasks(
        db=db,
        model_id=model_id,
        api_key=api_key,
    )

    return CompressionsResponse(data=compression_tasks, result_count=len(compression_tasks), total_count=len(compression_tasks))


@router.get("/{model_id}/evaluations", response_model=EvaluationsResponse)
def get_model_evaluation_tasks(
    model_id: str = Path(..., description="Model ID to get all related evaluation tasks"),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationsResponse:
    # 1. Get all converted model IDs derived from the trained model
    _, _, converted_model_ids = model_service._get_conversion_info(db, model_id)

    # 2. Get all evaluation tasks using original model ID and converted model IDs
    evaluation_tasks = []

    # Add evaluation tasks directly linked to the original model ID
    original_model_tasks = evaluation_task_service.get_evaluation_tasks(
        db=db,
        api_key=api_key,
        model_id=model_id
    )
    evaluation_tasks.extend(original_model_tasks)

    # Add evaluation tasks for each converted model
    for converted_id in converted_model_ids:
        converted_model_tasks = evaluation_task_service.get_evaluation_tasks(
            db=db,
            api_key=api_key,
            model_id=converted_id
        )
        evaluation_tasks.extend(converted_model_tasks)

    # Remove duplicates (based on task_id)
    unique_tasks = {}
    for task in evaluation_tasks:
        unique_tasks[task.task_id] = task

    evaluation_tasks = list(unique_tasks.values())

    return EvaluationsResponse(data=evaluation_tasks, result_count=len(evaluation_tasks), total_count=len(evaluation_tasks))
