from celery import chain

from app.worker.celery_app import celery_app
from netspresso import NetsPresso

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='convert_model')
def convert_model(
    self,
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
    conversion_task_id: str = None,
):
    netspresso = NetsPresso(api_key=api_key)

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
        input_model_id=input_model_id,
        conversion_task_id=conversion_task_id,
        wait_until_done=False,
    )

    chain(poll_conversion_status.s(api_key, task_id).set(countdown=POLLING_INTERVAL))()
    return task_id


@celery_app.task
def poll_conversion_status(api_key: str, task_id: str):
    netspresso = NetsPresso(api_key=api_key)

    converter = netspresso.converter_v2()
    status_updated = converter.update_conversion_task_status(task_id)

    if not status_updated:
        poll_conversion_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
