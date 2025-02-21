from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import api_key_header
from app.api.v1.schemas.device import SupportedDevicesForBenchmarkResponse
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
    response_model=SupportedDevicesForBenchmarkResponse,
    description="Get supported devices for model benchmark based on the conversion task.",
)
def get_supported_benchmark_devices(
    model_id: str = Query(..., description="Model id of the model to be benchmarked."),
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> SupportedDevicesForBenchmarkResponse:
    supported_devices = benchmark_task_service.get_supported_devices(
        db=db,
        model_id=model_id,
        api_key=api_key,
    )

    return SupportedDevicesForBenchmarkResponse(data=supported_devices, total_count=len(supported_devices))


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


@router.post("/benchmarks/{task_id}/cancel", response_model=BenchmarkResponse)
def cancel_benchmark_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(api_key_header),
) -> BenchmarkResponse:
    benchmark_task = benchmark_task_service.cancel_benchmark_task(db=db, task_id=task_id, api_key=api_key)

    return BenchmarkResponse(data=benchmark_task)
