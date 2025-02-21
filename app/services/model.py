from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload
from app.services.user import user_service
from netspresso.enums.project import SubFolder
from netspresso.utils.db.repositories.benchmark import benchmark_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository


class ModelService:
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
        _ = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = model_repository.delete_by_model_id(db=db, model_id=model_id)

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


model_service = ModelService()
