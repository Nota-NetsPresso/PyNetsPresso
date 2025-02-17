from celery import Celery, chain

from app.services.user import user_service
from netspresso.enums import Status, TaskStatusForDisplay
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.session import get_db

REDIS_URL = "localhost:6379"
REDIS_PASSWORD = ""
POLLING_INTERVAL = 10  # seconds

connection_url = f"redis://:{REDIS_PASSWORD}@{REDIS_URL}" if REDIS_PASSWORD else f"redis://{REDIS_URL}"

app = Celery("netspresso_converter", broker=f"{connection_url}/0", backend=f"{connection_url}/0")


@app.task
def convert_model_task(
    api_key: str,
    input_model_path: str,
    output_dir: str,
    target_framework: str,
    target_device_name: str,
    target_data_type: str,
    target_software_version: str = None,
    input_layer=None,
    dataset_path: str = None,
    input_model_id: str = None,
):
    db = next(get_db())
    netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
    converter = netspresso.converter_v2()
    task_id = converter.convert_model(
        input_model_path=input_model_path,
        output_dir=output_dir,
        target_framework=target_framework,
        target_device_name=target_device_name,
        target_data_type=target_data_type,
        target_software_version=target_software_version,
        input_layer=input_layer,
        dataset_path=dataset_path,
        wait_until_done=False,
        input_model_id=input_model_id,
    )
    # 폴링 태스크 체이닝
    chain(poll_conversion_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
    return task_id


@app.task
def poll_conversion_status(api_key: str, task_id: str):
    db = next(get_db())
    netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
    converter = netspresso.converter_v2()
    conversion_task = conversion_task_repository.get_by_task_id(db, task_id)

    # launcher server에서 상태 확인
    launcher_status = converter.get_conversion_task(conversion_task.convert_task_uuid)

    status_updated = False
    if launcher_status.status == TaskStatusForDisplay.FINISHED:
        conversion_task.status = Status.COMPLETED
        status_updated = True
    elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
        conversion_task.status = Status.ERROR
        conversion_task.error_detail = launcher_status.error_log
        status_updated = True
    elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
        conversion_task.status = Status.STOPPED
        status_updated = True

    if status_updated:
        conversion_task_repository.save(db, conversion_task)
        print(f"Conversion task {task_id} status updated to {conversion_task.status}")
    else:
        # 아직 완료되지 않았으면 다시 폴링 예약
        poll_conversion_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)


@app.task
def benchmark_model_task(
    api_key: str,
    input_model_path: str,
    target_device_name: str,
    target_software_version: str = None,
    target_hardware_type: str = None,
    input_model_id: str = None,
):
    db = next(get_db())
    netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
    benchmarker = netspresso.benchmarker_v2()
    task_id = benchmarker.benchmark_model(
        input_model_path=input_model_path,
        target_device_name=target_device_name,
        target_software_version=target_software_version,
        target_hardware_type=target_hardware_type,
        input_model_id=input_model_id,
    )
    chain(poll_benchmark_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
    return task_id


@app.task
def poll_benchmark_status(api_key: str, task_id: str):
    db = next(get_db())
    netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
    benchmarker = netspresso.benchmarker_v2()
    benchmark_task = benchmark_task_repository.get_by_task_id(db, task_id)

    print(benchmark_task.benchmark_task_id)

    # launcher server에서 상태 확인
    launcher_status = benchmarker.get_benchmark_task(benchmark_task.benchmark_task_id)

    status_updated = False
    if launcher_status.status == TaskStatusForDisplay.FINISHED:
        benchmark_task.status = Status.COMPLETED
        status_updated = True
    elif launcher_status.status in [TaskStatusForDisplay.ERROR, TaskStatusForDisplay.TIMEOUT]:
        benchmark_task.status = Status.ERROR
        benchmark_task.error_detail = launcher_status.error_log
        status_updated = True
    elif launcher_status.status == TaskStatusForDisplay.USER_CANCEL:
        benchmark_task.status = Status.STOPPED
        status_updated = True

    if status_updated:
        benchmark_task_repository.save(db, benchmark_task)
        print(f"Benchmark task {task_id} status updated to {benchmark_task.status}")
    else:
        # 아직 완료되지 않았으면 다시 폴링 예약
        poll_benchmark_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
