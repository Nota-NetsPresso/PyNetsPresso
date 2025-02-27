from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload, PresignedUrl
from app.configs.settings import settings
from app.services.training_task import train_task_service
from app.services.user import user_service
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.enums.project import SubFolder
from netspresso.exceptions.model import ModelCannotBeDeletedException
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository


class ModelService:
    def __init__(self):
        self.storage_handler = ObjectStorageHandler()
        self.BUCKET_NAME = settings.MODEL_BUCKET_NAME

    def _get_conversion_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str], List[str]]:
        """Get conversion task information

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            tuple: (latest_status, task_ids, model_ids)
        """
        conversion_tasks = conversion_task_repository.get_all_by_model_id(db=db, model_id=model_id)
        if not conversion_tasks:
            return None, [], []

        latest_status = conversion_tasks[0].status
        task_ids = []
        model_ids = []

        for task in conversion_tasks:
            task_ids.append(task.task_id)
            model_ids.append(task.model_id)

        return latest_status, task_ids, model_ids

    def _get_benchmark_info(self, db: Session, converted_model_ids: List[str]) -> tuple[Optional[str], List[str]]:
        """Get benchmark task information

        Args:
            db: Database session
            converted_model_ids: List of converted model IDs

        Returns:
            tuple: (latest_status, task_ids)
        """
        if not converted_model_ids:
            return None, []

        benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
            db=db, converted_model_ids=converted_model_ids
        )
        if not benchmark_tasks:
            return None, []

        latest_status = benchmark_tasks[0].status
        task_ids = [task.task_id for task in benchmark_tasks]

        return latest_status, task_ids

    def _attach_child_task_info(self, db: Session, model: ModelPayload) -> ModelPayload:
        """Attach child tasks (conversion, benchmark) information to model

        Args:
            db: Database session
            model: Model to attach task information to

        Returns:
            ModelPayload: Model with attached task information
        """
        # Get conversion tasks and their benchmark tasks
        conv_status, conv_task_ids, conv_model_ids = self._get_conversion_info(db, model.model_id)
        if conv_status:
            model.latest_experiments.convert = conv_status
            model.convert_task_ids.extend(conv_task_ids)

            # Get benchmark tasks for converted models
            bench_status, bench_task_ids = self._get_benchmark_info(db, conv_model_ids)
            if bench_status:
                model.latest_experiments.benchmark = bench_status
                model.benchmark_task_ids.extend(bench_task_ids)

        return model

    def get_models(self, db: Session, api_key: str) -> List[ModelPayload]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)
        models = model_repository.get_all_by_user_id(db=db, user_id=netspresso.user_info.user_id)

        new_models = []
        for model in models:
            if model.type in [SubFolder.CONVERTED_MODELS, SubFolder.BENCHMARKED_MODELS]:
                continue

            training_task = training_task_repository.get_by_model_id(db=db, model_id=model.model_id)
            model_payload = ModelPayload.model_validate(model)
            model_payload.train_task_id = training_task.task_id
            model_payload.status = training_task.status
            model_payload = self._attach_child_task_info(db, model_payload)
            new_models.append(model_payload)

        return new_models

    def get_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        _ = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)

        model_payload = ModelPayload.model_validate(model)
        model_payload.train_task_id = training_task.task_id
        model_payload.status = training_task.status

        return self._attach_child_task_info(db, model_payload)

    def delete_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        """Delete model and all related tasks

        Args:
            db: Database session
            model_id: Model ID to delete
            api_key: API key for authentication

        Returns:
            ModelPayload: Deleted model info

        Raises:
            HTTPException: If model not found
        """
        _ = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        # Get model before deletion
        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if model.type not in [SubFolder.TRAINED_MODELS]:
            raise ModelCannotBeDeletedException(model_id=model_id)

        # Delete model and training task
        model = model_repository.delete_by_model_id(db=db, model_id=model_id)
        training_task = train_task_service.delete_training_task_by_model_id(db=db, model_id=model_id)

        # Process and return model info
        model_payload = ModelPayload.model_validate(model)
        model_payload.train_task_id = training_task.task_id
        model_payload.status = training_task.status

        return self._attach_child_task_info(db, model_payload)

    def download_model(self, db: Session, model_id: str, api_key: str) -> PresignedUrl:
        """Download model file from Zenko

        Args:
            db: Database session
            model_id: Model ID to download
            api_key: API key for authentication

        Returns:
            FileResponse: Model file response

        Raises:
            HTTPException: If model not found or file not accessible
        """
        _ = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)

        try:
            # Generate presigned URL for download
            file_name = Path(model.object_path).name
            url = self.storage_handler.get_download_presigned_url(
                bucket_name=self.BUCKET_NAME,
                object_path=str(model.object_path),
                download_name=file_name,
                expires_in=3600  # URL expires in 1 hour
            )

            return PresignedUrl(
                model_id=model.model_id,
                file_name=file_name,
                presigned_url=url
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL: {str(e)}"
            )


model_service = ModelService()
