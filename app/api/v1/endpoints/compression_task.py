from fastapi import APIRouter, Depends

from app.api.deps import api_key_header
from app.api.v1.schemas.task.compression.compression_task import (
    CompressionCreate,
    CompressionCreateResponse,
)
from app.services.compression_task import compression_task_service

router = APIRouter()


@router.post("/compressions", response_model=CompressionCreateResponse, status_code=201)
def create_compressions_task(
    request_body: CompressionCreate,
    api_key: str = Depends(api_key_header),
) -> CompressionCreateResponse:
    compression_task = compression_task_service.create_compression_task(compression_in=request_body, api_key=api_key)

    return CompressionCreateResponse(data=compression_task)
