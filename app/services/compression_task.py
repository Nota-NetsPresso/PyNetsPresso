from sqlalchemy.orm import Session

from app.api.v1.schemas.task.compression.compression_task import CompressionCreate, CompressionCreatePayload
from app.worker.compression_task import compress_model
from netspresso.utils.db.models.base import generate_uuid


class CompressionTaskService:
    def create_compression_task(self, compression_in: CompressionCreate, api_key: str) -> CompressionCreatePayload:
        # Get model from trained models repository
        compression_task_id = generate_uuid(entity="task")
        _ = compress_model.apply_async(
            kwargs={
                "api_key": api_key,
                "method": compression_in.method,
                "recommendation_method": compression_in.recommendation_method,
                "ratio": compression_in.ratio,
                "options": compression_in.options.model_dump(),
                "input_model_id": compression_in.input_model_id,
                "compression_task_id": compression_task_id,
            },
            compression_task_id=compression_task_id,
        )
        return CompressionCreatePayload(task_id=compression_task_id)


compression_task_service = CompressionTaskService()

