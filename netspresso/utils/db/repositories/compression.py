from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.enums.metadata import Status
from netspresso.exceptions.compression import CompressionTaskIsDeletedException, CompressionTaskNotFoundException
from netspresso.utils.db.models.compression import CompressionModelResult, CompressionTask
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


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

    def get_all_by_input_model_id(
        self, db: Session, input_model_id: str, order: Order = Order.DESC, time_sort: TimeSort = TimeSort.CREATED_AT
    ) -> List[CompressionTask]:
        conditions = [self.model.input_model_id == input_model_id]
        tasks = self.find_all(db=db, conditions=conditions, order=order, time_sort=time_sort)

        return tasks

    def get_latest_compression_task(
        self, db: Session, input_model_id: str, order: Order = Order.DESC, time_sort: TimeSort = TimeSort.UPDATED_AT
    ) -> Optional[CompressionTask]:
        """Get the latest compression task for a model.

        Args:
            db: Database session
            input_model_id: Input model ID
            order: Order of results (default: DESC)
            time_sort: Time field to sort by (default: UPDATED_AT)

        Returns:
            Optional[CompressionTask]: Latest compression task if exists
        """
        conditions = [self.model.input_model_id == input_model_id]
        task = self.find_first(db=db, conditions=conditions, order=order, time_sort=time_sort)

        return self.__is_available(task=task)

    def get_completed_tasks(self, db: Session, user_id: str) -> List[CompressionTask]:
        conditions = [self.model.status == Status.COMPLETED, self.model.user_id == user_id]
        tasks = self.find_all(
            db=db,
            conditions=conditions,
        )
        return tasks


class CompressionModelResultRepository(BaseRepository[CompressionModelResult]):
    pass


compression_task_repository = CompressionTaskRepository(CompressionTask)
compression_model_result_repository = CompressionModelResultRepository(CompressionModelResult)
