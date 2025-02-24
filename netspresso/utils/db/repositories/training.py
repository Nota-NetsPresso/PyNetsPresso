from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.repositories.base import BaseRepository, Order


class TrainingTaskRepository(BaseRepository[TrainingTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[TrainingTask]:
        task = (
            db.query(self.model)
            .filter(
                self.model.task_id == task_id,
            )
            .first()
        )

        return task

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[TrainingTask]:
        task = (
            db.query(self.model)
            .filter(
                self.model.model_id == model_id,
            )
            .first()
        )

        return task

    def delete_by_model_id(self, db: Session, model_id: str) -> TrainingTask:
        task = self.get_by_model_id(db, model_id)
        task.is_deleted = True
        task = self.update(db, task)

        return task

training_task_repository = TrainingTaskRepository(TrainingTask)
