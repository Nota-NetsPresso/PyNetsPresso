from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesResponse
from app.api.v1.schemas.task.conversion.conversion_task import (
    ConversionCreate,
    ConversionCreateResponse,
    ConversionResponse,
)
from app.services.conversion_task import conversion_task_service
from netspresso.enums.conversion import SourceFramework
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get(
    "/conversions/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices and frameworks for model conversion based on the source framework.",
)
def get_supported_conversion_devices(
    framework: SourceFramework = Query(..., description="Source framework of the model to be converted."),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    supported_devices = conversion_task_service.get_supported_devices(db=db, framework=framework, api_key=api_key)

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/conversions", response_model=ConversionCreateResponse, status_code=201)
def create_conversions_task(
    request_body: ConversionCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionCreateResponse:
    conversion_task = conversion_task_service.create_conversion_task(db=db, conversion_in=request_body, api_key=api_key)

    return ConversionCreateResponse(data=conversion_task)


@router.get("/conversions/{task_id}", response_model=ConversionResponse)
def get_conversions_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionResponse:
    conversion_task = conversion_task_service.get_conversion_task(db=db, task_id=task_id, api_key=api_key)

    return ConversionResponse(data=conversion_task)


@router.post("/conversions/{task_id}/cancel", response_model=ConversionResponse)
def cancel_conversion_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> ConversionResponse:
    conversion_task = conversion_task_service.cancel_conversion_task(db=db, task_id=task_id, api_key=api_key)

    return ConversionResponse(data=conversion_task)
