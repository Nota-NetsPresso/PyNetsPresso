from celery import chain
from loguru import logger

from app.worker.celery_app import celery_app
from netspresso import NetsPresso

POLLING_INTERVAL = 30  # seconds


@celery_app.task(bind=True, name='compress_model')
def compress_model(
    self,
    api_key: str,
    method: str,
    recommendation_method: str,
    ratio: float,
    options: dict,
    input_model_id: str,
    compression_task_id: str,
):
    netspresso = NetsPresso(api_key=api_key)

    compressor = netspresso.compressor_v2()

    try:
        task_id = compressor.recommendation_compression_from_id(
            compression_method=method,
            recommendation_method=recommendation_method,
            recommendation_ratio=ratio,
            options=options,
            input_model_id=input_model_id,
            compression_task_id=compression_task_id,
        )
        result = {"task_id": task_id, "status": "completed"}
        return result
    except Exception as e:
        logger.error(f"Compression failed: {str(e)}")
        raise e
