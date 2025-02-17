from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesResponse
from app.api.v1.schemas.task.benchmark.benchmark_task import (
    BenchmarkCreate,
    BenchmarkCreateResponse,
    BenchmarkResponse,
)
from app.services.benchmark_task import benchmark_task_service
from netspresso.utils.db.session import get_db

router = APIRouter()


@router.get(
    "/benchmarks/configuration/devices",
    response_model=SupportedDevicesResponse,
    description="Get supported devices for model benchmark based on the conversion task.",
)
def get_supported_benchmark_devices(
    conversion_task_id: str = Query(..., description="Conversion task id of the model to be benchmarked."),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesResponse:
    supported_devices = benchmark_task_service.get_supported_devices(
        db=db,
        conversion_task_id=conversion_task_id,
        api_key=api_key,
    )

    return SupportedDevicesResponse(data=supported_devices)


@router.post("/benchmarks", response_model=BenchmarkCreateResponse, status_code=201)
def create_benchmark_task(
    request_body: BenchmarkCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> BenchmarkCreateResponse:
    benchmark_task = benchmark_task_service.create_benchmark_task(db=db, benchmark_in=request_body, api_key=api_key)

    return BenchmarkCreateResponse(data=benchmark_task)


@router.get("/benchmarks/{task_id}", response_model=BenchmarkResponse)
def get_benchmark_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> BenchmarkResponse:
    benchmark_task = benchmark_task_service.get_benchmark_task(db=db, task_id=task_id, api_key=api_key)

    return BenchmarkResponse(data=benchmark_task)
