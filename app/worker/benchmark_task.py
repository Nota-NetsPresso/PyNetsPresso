from celery import chain

from app.worker.celery_app import celery_app
from netspresso import NetsPresso

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='benchmark_model')
def benchmark_model(
    self,
    api_key: str,
    input_model_path: str,
    target_device_name: str,
    target_software_version: str = None,
    target_hardware_type: str = None,
    input_model_id: str = None,
    benchmark_task_id: str = None,
):
    netspresso = NetsPresso(api_key=api_key)

    benchmarker = netspresso.benchmarker_v2()
    task_id = benchmarker.benchmark_model(
        input_model_path=input_model_path,
        target_device_name=target_device_name,
        target_software_version=target_software_version,
        target_hardware_type=target_hardware_type,
        input_model_id=input_model_id,
        benchmark_task_id=benchmark_task_id,
        wait_until_done=False,
    )

    chain(poll_benchmark_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
    return task_id


@celery_app.task
def poll_benchmark_status(api_key: str, task_id: str):
    netspresso = NetsPresso(api_key=api_key)

    benchmarker = netspresso.benchmarker_v2()
    status_updated = benchmarker.update_benchmark_task_status(task_id)

    if not status_updated:
        poll_benchmark_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
