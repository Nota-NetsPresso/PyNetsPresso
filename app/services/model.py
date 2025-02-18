from typing import List

from sqlalchemy.orm import Session

from app.api.v1.schemas.model import ModelPayload
from app.services.user import user_service
from netspresso.enums.project import SubFolder
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
                # Collect all task IDs
                for conversion_task in conversion_tasks:
                    model.convert_task_ids.append(conversion_task.task_id)

            model.status = task_status
            new_models.append(model)

        return new_models

    def get_model(self, db: Session, model_id: str, api_key: str) -> ModelPayload:
        netspresso = user_service.build_netspresso_with_api_key(db=db, api_key=api_key)

        model = model_repository.get_by_model_id(db=db, model_id=model_id, user_id=netspresso.user_info.user_id)
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
        task_status = training_task.status
        model.train_task_id = training_task.task_id

        model = ModelPayload.model_validate(model)

        # Get conversion tasks ordered by created_at desc
        conversion_tasks = conversion_task_repository.get_all_by_model_id(db=db, model_id=model.model_id)

        if conversion_tasks:
            # Set latest experiment status from the most recent conversion task
            model.latest_experiments.convert = conversion_tasks[0].status
            # Collect all task IDs
            for conversion_task in conversion_tasks:
                model.convert_task_ids.append(conversion_task.task_id)

        model.status = task_status

        return model


model_service = ModelService()
