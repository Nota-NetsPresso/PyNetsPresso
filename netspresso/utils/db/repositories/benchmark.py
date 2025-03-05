from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.utils.db.models.benchmark import BenchmarkTask
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class BenchmarkTaskRepository(BaseRepository[BenchmarkTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[BenchmarkTask]:
        conditions = [self.model.task_id == task_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return task

    def get_all_by_model_id(
        self,
        db: Session,
        model_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> Optional[List[BenchmarkTask]]:
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

    def get_all_by_converted_models(self, db: Session, converted_model_ids: List[str]) -> List[BenchmarkTask]:
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
            order=Order.DESC,
            time_sort=TimeSort.UPDATED_AT,
        )

        return tasks


benchmark_task_repository = BenchmarkTaskRepository(BenchmarkTask)
