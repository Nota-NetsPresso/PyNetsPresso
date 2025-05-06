import copy
from pathlib import Path
from typing import Dict, List

from sqlalchemy.orm import Session

from app.api.v1.schemas.task.train.hyperparameter import OptimizerPayload, SchedulerPayload, TrainerModel
from app.api.v1.schemas.task.train.train_task import (
    FrameworkPayload,
    PretrainedModelPayload,
    TaskPayload,
    TrainingCreate,
    TrainingCreatePayload,
    TrainingPayload,
)
from app.worker.training_task import train_model
from netspresso.enums.train import MODEL_DISPLAY_MAP, MODEL_GROUP_MAP
from netspresso.trainer.augmentations.augmentation import Normalize, Resize, ToTensor
from netspresso.trainer.models import get_all_available_models
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.optimizers.optimizers import get_supported_optimizers
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.schedulers.schedulers import get_supported_schedulers
from netspresso.trainer.trainer import Trainer
from netspresso.utils.db.models.base import generate_uuid
from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository


class TrainTaskService:
    def get_supported_models(self) -> Dict[str, List[TrainerModel]]:
        """Get all supported models grouped by task."""
        available_models = get_all_available_models()
        supported_models = {
            task: [
                TrainerModel(
                    name=model,
                    display_name=MODEL_DISPLAY_MAP.get(model),
                    group_name=MODEL_GROUP_MAP.get(model),
                )
                for model in models
            ]
            for task, models in available_models.items()
        }

        return supported_models

    def get_supported_optimizers(self) -> List[OptimizerPayload]:
        """Get all supported optimizers."""
        supported_optimizers = get_supported_optimizers()
        optimizers = [OptimizerPayload(name=optimizer.get("name")) for optimizer in supported_optimizers]
        return optimizers

    def get_supported_schedulers(self) -> List[SchedulerPayload]:
        """Get all supported schedulers."""
        supported_schedulers = get_supported_schedulers()
        schedulers = [SchedulerPayload(name=scheduler.get("name")) for scheduler in supported_schedulers]
        return schedulers

    def _setup_trainer(self, trainer, training_in: TrainingCreate) -> Trainer:
        """Configure trainer with the given training parameters."""
        trainer.set_dataset_config(
            name="test",
            root_path="/root/projects/traffic-sign",
            train_image="train/images",
            train_label="train/labels",
            valid_image="valid/images",
            valid_label="valid/labels",
            id_mapping=["prohibitory", "danger", "mandatory", "other"],
        )

        img_size = training_in.input_shapes[0].dimension[0]
        trainer.set_model_config(model_name=training_in.pretrained_model, img_size=img_size)

        trainer.set_augmentation_config(
            train_transforms=[Resize(), ToTensor(), Normalize()],
            inference_transforms=[Resize(), ToTensor(), Normalize()],
        )

        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(name=training_in.hyperparameter.scheduler)

        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        return trainer

    def _convert_to_payload_format(self, training_task: TrainingTask) -> TrainingPayload:
        """Convert training task to payload format."""
        # 원본 데이터를 변경하지 않기 위해 깊은 복사 수행
        task_data = copy.deepcopy(training_task.__dict__)

        # 필요한 필드들을 Payload 객체로 변환
        task_data['task'] = TaskPayload(name=training_task.task)
        task_data['framework'] = FrameworkPayload(name=training_task.framework)
        task_data['pretrained_model'] = PretrainedModelPayload(name=training_task.pretrained_model)

        # 하이퍼파라미터 정보 변환
        hyperparameter_data = copy.deepcopy(training_task.hyperparameter.__dict__)
        hyperparameter_data['learning_rate'] = training_task.hyperparameter.optimizer["lr"]
        hyperparameter_data['optimizer'] = OptimizerPayload(name=training_task.hyperparameter.optimizer["name"])
        hyperparameter_data['scheduler'] = SchedulerPayload(name=training_task.hyperparameter.scheduler["name"])
        task_data['hyperparameter'] = hyperparameter_data

        # 모델 ID 설정
        task_data['model_id'] = training_task.model.model_id if training_task.model else None

        # SQLAlchemy 내부 상태 속성 제거
        if '_sa_instance_state' in task_data:
            del task_data['_sa_instance_state']

        # Pydantic 모델로 변환하여 반환
        return TrainingPayload.model_validate(task_data)

    def _generate_unique_model_name(self, db: Session, project_id: str, name: str, api_key: str) -> str:
        """Generate a unique model name by adding numbering if necessary.

        Args:
            db (Session): Database session
            project_id (str): Project ID to check existing models
            name (str): Original model name
            api_key (str): API key for authentication

        Returns:
            str: Unique model name with numbering if needed
        """
        # Get existing model names from the database for the same project
        models = model_repository.get_all_by_project_id(
            db=db,
            project_id=project_id,
        )

        # Extract existing names from models and count occurrences of base name
        base_name_count = sum(1 for model in models if model.type == "trained_models" and model.name.startswith(name))

        # If no models with this name exist, return original name
        if base_name_count == 0:
            return name

        # If models exist, return name with count
        return f"{name} ({base_name_count})"

    def create_training_task(self, db: Session, training_in: TrainingCreate, api_key: str) -> TrainingCreatePayload:
        """Create and execute a new training task."""
        unique_model_name = self._generate_unique_model_name(
            db=db,
            project_id=training_in.project_id,
            name=training_in.name,
            api_key=api_key,
        )

        training_task_id = generate_uuid(entity="task")
        _ = train_model.apply_async(
            kwargs={
                "task_id": training_task_id,
                "api_key": api_key,
                "training_in": training_in.model_dump(),
                "unique_model_name": unique_model_name,
            },
            task_id=training_task_id,
        )

        return TrainingCreatePayload(task_id=training_task_id)

    def get_training_task(self, db: Session, task_id: str, api_key: str) -> TrainingPayload:
        """Get training task by task ID."""
        training_task = training_task_repository.get_by_task_id(db=db, task_id=task_id)

        return self._convert_to_payload_format(training_task)

    def delete_training_task_by_model_id(self, db: Session, model_id: str) -> TrainingPayload:
        """Delete training task by model ID."""
        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
        training_task_repository.soft_delete(db=db, model=training_task)

        return self._convert_to_payload_format(training_task)

train_task_service = TrainTaskService()
