from celery import chain

from app.worker.celery_app import celery_app
from netspresso import NetsPresso
from netspresso.enums.metadata import Status
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.session import get_db_session

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
    try:
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
    except Exception as e:
        from loguru import logger
        logger.error(f"Error in benchmark_model task: {str(e)}")

        if benchmark_task_id:
            try:
                with get_db_session() as db:
                    task = benchmark_task_repository.get_by_task_id(db, benchmark_task_id)
                    if task:
                        task.status = Status.ERROR
                        task.error_detail = {"error": str(e)}
                        benchmark_task_repository.save(db, task)
            except Exception as db_err:
                logger.error(f"Failed to update benchmark task status: {str(db_err)}")

        raise e  # Re-raise the exception to mark the task as failed


@celery_app.task
def poll_benchmark_status(api_key: str, task_id: str):
    netspresso = NetsPresso(api_key=api_key)

    benchmarker = netspresso.benchmarker_v2()
    status_updated = benchmarker.update_benchmark_task_status(task_id)

    if not status_updated:
        poll_benchmark_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
