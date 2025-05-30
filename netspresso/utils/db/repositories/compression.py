from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.compression import CompressionTaskIsDeletedException, CompressionTaskNotFoundException
from netspresso.utils.db.models.compression import CompressionModelResult, CompressionTask
from netspresso.utils.db.repositories.base import BaseRepository


class CompressionTaskRepository(BaseRepository[CompressionTask]):
    def __is_available(self, task: Optional[CompressionTask]) -> CompressionTask:
        if task is None:
            raise CompressionTaskNotFoundException()

        if task.is_deleted:
            raise CompressionTaskIsDeletedException(task_id=task.task_id)

        return task

    def get_by_model_id(self, db: Session, model_id: str) -> CompressionTask:
        conditions = [self.model.model_id == model_id]
        task = self.find_first(db=db, conditions=conditions)

        return self.__is_available(task=task)

    def get_by_task_id(self, db: Session, task_id: str) -> CompressionTask:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_all_by_input_model_id(self, db: Session, input_model_id: str) -> List[CompressionTask]:
        conditions = [self.model.input_model_id == input_model_id]
        tasks = self.find_all(db=db, conditions=conditions)

        return tasks


class CompressionModelResultRepository(BaseRepository[CompressionModelResult]):
    pass


compression_task_repository = CompressionTaskRepository(CompressionTask)
compression_model_result_repository = CompressionModelResultRepository(CompressionModelResult)
