from typing import List, Optional

from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload
from app.services.training_task import train_task_service
from app.services.user import user_service
from netspresso.enums.project import SubFolder
from netspresso.exceptions.model import ModelCannotBeDeletedException
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository


class ModelService:
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

    def get_models(self, db: Session, api_key: str) -> List[ModelPayload]:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        models = model_repository.get_all_by_user_id(db=db, user_id=netspresso.user_info.user_id)

        new_models = []
        for model in models:
            if model.type in [SubFolder.CONVERTED_MODELS, SubFolder.BENCHMARKED_MODELS]:
                continue

            training_task = training_task_repository.get_by_model_id(db=db, model_id=model.model_id)
            task_status = training_task.status
            model.train_task_id = training_task.task_id

            model = ModelPayload.model_validate(model)

            # Get conversion tasks ordered by created_at desc
            conversion_tasks = conversion_task_repository.get_all_by_model_id(db=db, model_id=model.model_id)

            if conversion_tasks:
                # Set latest experiment status from the most recent conversion task
                model.latest_experiments.convert = conversion_tasks[0].status
                # Collect conversion task IDs
                converted_model_ids = []
                for conversion_task in conversion_tasks:
                    model.convert_task_ids.append(conversion_task.task_id)
                    converted_model_ids.append(conversion_task.model_id)

                # Get all benchmark tasks for converted models in single query
                benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
                    db=db, converted_model_ids=converted_model_ids
                )

                if benchmark_tasks:
                    # First task is most recent due to order_by in query
                    model.latest_experiments.benchmark = benchmark_tasks[0].status
                    # Collect all benchmark task IDs
                    for benchmark_task in benchmark_tasks:
                        model.benchmark_task_ids.append(benchmark_task.task_id)

            model.status = task_status
            new_models.append(model)

        return new_models

    def get_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        _ = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
        task_status = training_task.status
        model.train_task_id = training_task.task_id

        model = ModelPayload.model_validate(model)

        # Get conversion tasks ordered by created_at desc
        conversion_tasks = conversion_task_repository.get_all_by_model_id(db=db, model_id=model.model_id)

        if conversion_tasks:
            # Set latest experiment status from the most recent conversion task
            model.latest_experiments.convert = conversion_tasks[0].status
            # Collect conversion task IDs
            converted_model_ids = []
            for conversion_task in conversion_tasks:
                model.convert_task_ids.append(conversion_task.task_id)
                converted_model_ids.append(conversion_task.model_id)

            # Get all benchmark tasks for converted models in single query
            benchmark_tasks = benchmark_task_repository.get_all_by_converted_models(
                db=db, converted_model_ids=converted_model_ids
            )

            if benchmark_tasks:
                # First task is most recent due to order_by in query
                model.latest_experiments.benchmark = benchmark_tasks[0].status
                # Collect all benchmark task IDs
                for benchmark_task in benchmark_tasks:
                    model.benchmark_task_ids.append(benchmark_task.task_id)

        model.status = task_status

        return model

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

        # Get model before deletion to return its info
        model = model_repository.get_by_model_id(db=db, model_id=model_id)
        model = model_repository.delete_by_model_id(db=db, model_id=model_id)

        # Deletion is allowed for Trained or Compressed models(Compressed model is not implemented yet)
        # Deletion is not allowed for Converted or Benchmark models
        if model.type in [SubFolder.TRAINED_MODELS]:
            task = train_task_service.delete_training_task_by_model_id(db=db, model_id=model_id)

        elif model.type in [SubFolder.BENCHMARKED_MODELS, SubFolder.CONVERTED_MODELS]:
            raise ModelCannotBeDeletedException(model_id=model_id)

        # Process and return the model info
        model = ModelPayload.model_validate(model)
        model.status = task.status
        model.train_task_id = task.task_id

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


model_service = ModelService()
