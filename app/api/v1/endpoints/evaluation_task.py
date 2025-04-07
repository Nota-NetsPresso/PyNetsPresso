from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesResponse
from app.api.v1.schemas.task.evaluation.evaluation_task import (
    EvaluationCreate,
    EvaluationCreateResponse,
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
