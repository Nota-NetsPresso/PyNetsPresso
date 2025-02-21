from celery import Celery, chain
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session

from app.services.user import user_service
from netspresso.utils.db.session import SessionLocal

REDIS_URL = "localhost:6379"
REDIS_PASSWORD = ""
POLLING_INTERVAL = 30  # seconds

connection_url = f"redis://:{REDIS_PASSWORD}@{REDIS_URL}" if REDIS_PASSWORD else f"redis://{REDIS_URL}"

app = Celery("netspresso_converter", broker=f"{connection_url}/0", backend=f"{connection_url}/0")


def create_scoped_session():
    """Create a scoped session that will be removed after each request"""
    return scoped_session(SessionLocal)


def _process_with_session(func, *args, **kwargs):
    """Execute function with a new session and ensure proper cleanup

    Args:
        func: Function to execute
        args: Positional arguments for func
        kwargs: Keyword arguments for func

    Returns:
        Result from func
    """
    session = create_scoped_session()
    try:
        result = func(session, *args, **kwargs)
        session.commit()
        return result
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.remove()


def _process_conversion(session: Session, api_key: str, **kwargs) -> str:
    netspresso = user_service.build_netspresso_with_api_key(db=session, api_key=api_key)
    converter = netspresso.converter_v2()
    return converter.convert_model(**kwargs, wait_until_done=False)


def _process_conversion_status(session: Session, api_key: str, task_id: str) -> bool:
    netspresso = user_service.build_netspresso_with_api_key(db=session, api_key=api_key)
    converter = netspresso.converter_v2()
    return converter.update_conversion_task_status(task_id)


def _process_benchmark(session: Session, api_key: str, **kwargs) -> str:
    netspresso = user_service.build_netspresso_with_api_key(db=session, api_key=api_key)
    benchmarker = netspresso.benchmarker_v2()
    return benchmarker.benchmark_model(**kwargs, wait_until_done=False)


def _process_benchmark_status(session: Session, api_key: str, task_id: str) -> bool:
    netspresso = user_service.build_netspresso_with_api_key(db=session, api_key=api_key)
    benchmarker = netspresso.benchmarker_v2()
    return benchmarker.update_benchmark_task_status(task_id)


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
    task_id = _process_with_session(
        _process_conversion,
        api_key,
        input_model_path=input_model_path,
        output_dir=output_dir,
        target_framework=target_framework,
        target_device_name=target_device_name,
        target_data_type=target_data_type,
        target_software_version=target_software_version,
        input_layer=input_layer,
        dataset_path=dataset_path,
        input_model_id=input_model_id,
    )

    chain(poll_conversion_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
    return task_id


@app.task
def poll_conversion_status(api_key: str, task_id: str):
    status_updated = _process_with_session(_process_conversion_status, api_key, task_id)

    if not status_updated:
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
    task_id = _process_with_session(
        _process_benchmark,
        api_key,
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
    status_updated = _process_with_session(_process_benchmark_status, api_key, task_id)

    if not status_updated:
        poll_benchmark_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
