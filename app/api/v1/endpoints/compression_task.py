from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.task.compression.compression_task import (
    CompressionCreate,
    CompressionCreateResponse,
    CompressionResponse,
)
from app.services.compression_task import compression_task_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.post("/compressions", response_model=CompressionCreateResponse, status_code=201)
def create_compressions_task(
    request_body: CompressionCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionCreateResponse:
    compression_task = compression_task_service.create_compression_task(
        db=db, compression_in=request_body, api_key=api_key
    )

    return CompressionCreateResponse(data=compression_task)


@router.get("/compressions/{task_id}", response_model=CompressionResponse)
def get_compression_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionResponse:
    compression_task = compression_task_service.get_compression_task(
        db=db, compression_task_id=task_id, api_key=api_key
    )

    return CompressionResponse(data=compression_task)


@router.delete("/compressions/{task_id}", response_model=CompressionResponse)
def delete_compression_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> CompressionResponse:
    compression_task = compression_task_service.delete_compression_task(db=db, task_id=task_id, api_key=api_key)

    return CompressionResponse(data=compression_task)
