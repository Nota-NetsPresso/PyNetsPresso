from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesResponse
from app.api.v1.schemas.task.evaluation.evaluation_task import (
    EvaluationCreate,
    EvaluationCreatePayload,
    EvaluationCreateResponse,
    EvaluationsResponse,
)
from app.services.evaluation_task import evaluation_task_service
from netspresso.enums.conversion import SourceFramework
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get(
    "/evaluations/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices and frameworks for model evaluation based on the source framework.",
)
def get_supported_evaluation_devices(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    framework = SourceFramework.ONNX
    supported_devices = evaluation_task_service.get_supported_devices(db=db, framework=framework, api_key=api_key)

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/evaluations", response_model=EvaluationCreateResponse, status_code=201)
def create_evaluations_task(
    request_body: EvaluationCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationCreateResponse:
    evaluation_task_id = evaluation_task_service.create_evaluation_task(db=db, evaluation_in=request_body, api_key=api_key)

    response_data = EvaluationCreatePayload(task_id=evaluation_task_id)
    return EvaluationCreateResponse(data=response_data)


@router.get("/evaluations", response_model=EvaluationsResponse, status_code=200)
def get_evaluation_tasks(
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> EvaluationsResponse:
    evaluation_tasks = evaluation_task_service.get_evaluation_tasks(db=db, api_key=api_key)
    total_count = evaluation_task_service.count_evaluation_task_by_user_id(db=db, api_key=api_key)

    return EvaluationsResponse(data=evaluation_tasks, result_count=len(evaluation_tasks), total_count=total_count)
