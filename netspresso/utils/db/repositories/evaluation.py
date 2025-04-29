from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.enums.metadata import Status
from netspresso.exceptions.evaluation import EvaluationTaskIsDeletedException, EvaluationTaskNotFoundException
from netspresso.utils.db.models.evaluation import EvaluationTask
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class EvaluationTaskRepository(BaseRepository[EvaluationTask]):
    def __is_available(self, task: Optional[EvaluationTask]) -> EvaluationTask:
        if task is None:
            raise EvaluationTaskNotFoundException()

        if task.is_deleted:
            raise EvaluationTaskIsDeletedException(task_id=task.task_id)

        return task

    def get_by_task_id(self, db: Session, task_id: str) -> Optional[EvaluationTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)


evaluation_task_repository = EvaluationTaskRepository(EvaluationTask)
