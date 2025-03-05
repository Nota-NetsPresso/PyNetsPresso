from typing import Optional

from sqlalchemy.orm import Session

from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.repositories.base import BaseRepository


class TrainingTaskRepository(BaseRepository[TrainingTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return task

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.model_id == model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return task


training_task_repository = TrainingTaskRepository(TrainingTask)
