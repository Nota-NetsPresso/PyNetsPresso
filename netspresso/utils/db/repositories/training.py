from typing import Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.training import TrainingTaskIsDeletedException, TrainingTaskNotFoundException
from netspresso.utils.db.models.training import TrainingTask
from netspresso.utils.db.repositories.base import BaseRepository


class TrainingTaskRepository(BaseRepository[TrainingTask]):
    def __is_available(self, task: Optional[TrainingTask]) -> TrainingTask:
        if task is None:
            raise TrainingTaskNotFoundException()

        if task.is_deleted:
            raise TrainingTaskIsDeletedException(task_id=task.task_id)

        return task

    def get_by_task_id(self, db: Session, task_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[TrainingTask]:
        conditions = [self.model.model_id == model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_by_output_model_id(self, db: Session, output_model_id: str) -> Optional[TrainingTask]:
        """
        Get training task by output model ID

        Args:
            db: Database session
            output_model_id: Output model ID

        Returns:
            TrainingTask or None if not found
        """
        conditions = [self.model.model_id == output_model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)


training_task_repository = TrainingTaskRepository(TrainingTask)
