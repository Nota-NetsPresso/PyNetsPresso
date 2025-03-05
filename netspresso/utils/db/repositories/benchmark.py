from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.benchmark import BenchmarkTaskIsDeletedException, BenchmarkTaskNotFoundException
from netspresso.utils.db.models.benchmark import BenchmarkTask
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class BenchmarkTaskRepository(BaseRepository[BenchmarkTask]):
    def __is_available(self, task: Optional[BenchmarkTask]) -> BenchmarkTask:
        if task is None:
            raise BenchmarkTaskNotFoundException()

        if task.is_deleted:
            raise BenchmarkTaskIsDeletedException(task_id=task.task_id)

        return task

    def get_by_task_id(self, db: Session, task_id: str) -> Optional[BenchmarkTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_all_by_model_id(
        self,
        db: Session,
        model_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[BenchmarkTask]:
        conditions = [self.model.input_model_id == model_id]
        tasks = self.find_all(
            db=db,
            conditions=conditions,
            start=start,
            size=size,
            order=order,
            time_sort=time_sort,
        )

        return tasks

    def get_all_by_converted_models(
        self,
        db: Session,
        converted_model_ids: List[str],
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[BenchmarkTask]:
        """Get all benchmark tasks for given converted model IDs ordered by updated_at desc.

        Args:
            db: Database session
            converted_model_ids: List of converted model IDs

        Returns:
            List[BenchmarkTask]: List of benchmark tasks ordered by updated_at desc
        """

        conditions = [self.model.input_model_id.in_(converted_model_ids)]
        tasks = self.find_all(
            db=db,
            conditions=conditions,
            order=order,
            time_sort=time_sort,
        )

        return tasks

    def get_latest_benchmark_task(
        self,
        db: Session,
        converted_model_ids: List[str],
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> Optional[BenchmarkTask]:
        conditions = [self.model.input_model_id.in_(converted_model_ids)]
        task = self.find_first(
            db=db,
            conditions=conditions,
            order=order,
            time_sort=time_sort,
        )

        return task

benchmark_task_repository = BenchmarkTaskRepository(BenchmarkTask)
