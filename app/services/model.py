from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload, PresignedUrl
from app.configs.settings import settings
from app.services.training_task import train_task_service
from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.enums.metadata import Status
from netspresso.enums.project import SubFolder
from netspresso.enums.task import TaskType
from netspresso.exceptions.model import ModelCannotBeDeletedException
from netspresso.netspresso import NetsPresso
from netspresso.utils.db.repositories.base import Order, TimeSort
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.compression import compression_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.evaluation import evaluation_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository


class ModelService:
    def __init__(self):
        self.storage_handler = ObjectStorageHandler()
        self.BUCKET_NAME = settings.MODEL_BUCKET_NAME

    def _get_conversion_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str], List[str]]:
        # Get all conversion tasks sorted by creation time (newest first)
        conversion_tasks = conversion_task_repository.get_all_by_model_id(
            db=db, model_id=model_id, order=Order.DESC, time_sort=TimeSort.CREATED_AT,
        )

        task_ids = []
        model_ids = []

        for task in conversion_tasks:
            task_ids.append(task.task_id)
            model_ids.append(task.model_id)

        latest_status = conversion_tasks[0].status if conversion_tasks else None

        return latest_status, task_ids, model_ids

    def _get_evaluation_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str]]:
        """Get evaluation task information

        Args:
            db: Database session
            model_id: Model ID

        Returns:
            tuple: (latest_status, task_ids)
        """
        evaluation_tasks = evaluation_task_repository.get_all_by_model_id(
            db=db,
            model_id=model_id,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        if not evaluation_tasks:
            return None, []

        task_ids = [task.task_id for task in evaluation_tasks]

        evaluation_task = evaluation_task_repository.get_latest_evaluation_task(
            db=db,
            model_id=model_id,
            order=Order.DESC,
            time_sort=TimeSort.UPDATED_AT,
        )
        latest_status = evaluation_task.status

        return latest_status, task_ids

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
            db=db,
            converted_model_ids=converted_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.CREATED_AT,
        )
        if not benchmark_tasks:
            return None, []

        task_ids = [task.task_id for task in benchmark_tasks]

        benchmark_task = benchmark_task_repository.get_latest_benchmark_task(
            db=db,
            converted_model_ids=converted_model_ids,
            order=Order.DESC,
            time_sort=TimeSort.UPDATED_AT,
        )
        latest_status = benchmark_task.status

        return latest_status, task_ids

    def _get_compression_info(self, db: Session, model_id: str) -> tuple[Optional[str], List[str]]:
        """Get compression task information for a model

        Args:
            db: Database session
            model_id: Model ID to get compression tasks for

        Returns:
            tuple: (latest status, task IDs)
        """
        compression_tasks = compression_task_repository.get_all_by_input_model_id(db=db, input_model_id=model_id)
        if not compression_tasks:
            return None, []

        task_ids = [task.task_id for task in compression_tasks]
        compression_task = compression_task_repository.get_latest_compression_task(db=db, input_model_id=model_id)
        latest_status = compression_task.status

        return latest_status, task_ids

    def _attach_child_task_info(self, db: Session, model: ModelPayload) -> ModelPayload:
        """Attach child tasks (conversion, benchmark, evaluation) information to model

        Args:
            db: Database session
            model: Model to attach task information to

        Returns:
            ModelPayload: Model with attached task information
        """
        # Get evaluation tasks
        eval_status, eval_task_ids = self._get_evaluation_info(db, model.model_id)
        if eval_status:
            model.latest_experiments.evaluate = eval_status
            model.evaluation_task_ids.extend(eval_task_ids)

        # Get compression tasks
        comp_status, comp_task_ids = self._get_compression_info(db, model.model_id)
        if comp_status:
            model.latest_experiments.compress = comp_status
            model.compress_task_ids.extend(comp_task_ids)

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

    def get_models(
        self,
        db: Session,
        api_key: str,
        task_type: Optional[TaskType] = None,
        project_id: Optional[str] = None
    ) -> List[ModelPayload]:
        netspresso = NetsPresso(api_key=api_key)

        # Base query by user ID
        user_id = netspresso.user_info.user_id

        # If project_id is provided, filter by project
        if project_id:
            models = model_repository.get_all_by_project_id(db=db, project_id=project_id)
            # Filter models that belong to the user for security
            models = [model for model in models if model.user_id == user_id]
        else:
            models = model_repository.get_all_by_user_id(db=db, user_id=user_id)

        new_models = []
        for model in models:
            # Handle retraining task - only return completed trained and compressed models
            if task_type == TaskType.RETRAIN:
                if model.type not in [SubFolder.TRAINED_MODELS, SubFolder.COMPRESSED_MODELS]:
                    continue
                training_task = (
                    training_task_repository.get_by_model_id(db=db, model_id=model.model_id)
                    if model.type == SubFolder.TRAINED_MODELS
                    else training_task_repository.get_by_model_id(
                        db=db,
                        model_id=compression_task_repository.get_by_model_id(
                            db=db,
                            model_id=model.model_id
                        ).input_model_id
                    )
                )
                if not training_task or training_task.status != Status.COMPLETED:
                    continue
            # Handle compression task separately as it only needs trained models
            elif task_type == TaskType.COMPRESS:
                if model.type != SubFolder.TRAINED_MODELS:
                    continue
            # Handle other conversion/evaluation tasks that can use both trained and compressed models
            elif task_type in [TaskType.BENCHMARK, TaskType.EVALUATE, TaskType.CONVERT]:
                if model.type not in [SubFolder.TRAINED_MODELS, SubFolder.COMPRESSED_MODELS]:
                    continue
            # For other cases (like listing), exclude converted and benchmarked models
            else:
                if model.type in [SubFolder.CONVERTED_MODELS, SubFolder.BENCHMARKED_MODELS]:
                    continue

            model_payload = ModelPayload.model_validate(model)

            # Get training task based on model type
            if model.type == SubFolder.COMPRESSED_MODELS:
                compression_task = compression_task_repository.get_by_model_id(db=db, model_id=model.model_id)
                training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
            else:
                training_task = training_task_repository.get_by_model_id(db=db, model_id=model.model_id)

            model_payload.train_task_id = training_task.task_id
            model_payload.status = training_task.status
            model_payload = self._attach_child_task_info(db, model_payload)
            new_models.append(model_payload)

        return new_models

    def get_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        model_payload = ModelPayload.model_validate(model)

        if model.type == SubFolder.COMPRESSED_MODELS:
            compression_task = compression_task_repository.get_by_model_id(db=db, model_id=model_id)
            training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
            model_payload.status = compression_task.status
        else:
            training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
            model_payload.status = training_task.status

        model_payload.train_task_id = training_task.task_id
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
        # Get model before deletion
        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        if model.type not in [SubFolder.TRAINED_MODELS, SubFolder.COMPRESSED_MODELS]:
            raise ModelCannotBeDeletedException(model_id=model_id)

        # Delete model and training task
        model = model_repository.soft_delete(db=db, model=model)

        if model.type == SubFolder.COMPRESSED_MODELS:
            compression_task = compression_task_repository.get_by_model_id(db=db, model_id=model_id)
            training_task = train_task_service.get_training_task_by_model_id(db=db, model_id=compression_task.input_model_id)
            compression_task_repository.soft_delete(db=db, compression_task=compression_task)
        else:
            training_task = train_task_service.get_training_task_by_model_id(db=db, model_id=model_id)

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
        model = model_repository.get_by_model_id(db=db, model_id=model_id)

        if model.type == SubFolder.TRAINED_MODELS:
            object_path = f"{model.object_path}/model.onnx"
            extension = "onnx"
        else:
            object_path = model.object_path
            extension = Path(object_path).suffix.lstrip('.')

        try:
            # Generate presigned URL for download
            download_name = f"{model.name}.{extension}"  # 모델 이름으로 파일명 생성
            url = self.storage_handler.get_download_presigned_url(
                bucket_name=self.BUCKET_NAME,
                object_path=object_path,
                download_name=download_name,
                expires_in=3600  # URL expires in 1 hour
            )

            return PresignedUrl(
                model_id=model.model_id,
                file_name=download_name,  # 변경된 파일명 사용
                presigned_url=url
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate download URL: {str(e)}"
            )


model_service = ModelService()
