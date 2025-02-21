from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from netspresso.utils.db.models.benchmark import BenchmarkTask
from netspresso.utils.db.repositories.base import BaseRepository, Order


class BenchmarkTaskRepository(BaseRepository[BenchmarkTask]):
    def get_by_task_id(self, db: Session, task_id: str) -> Optional[BenchmarkTask]:
        task = (
            db.query(self.model)
            .filter(
                self.model.task_id == task_id,
            )
            .first()
        )

        return task

    def _get_tasks(
        self,
        db: Session,
        condition,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[BenchmarkTask]]:
        ordering_func = self.choose_order_func(order)
        query = db.query(self.model).filter(condition)

        if order:
            query = query.order_by(ordering_func(self.model.updated_at))

        if start is not None and size is not None:
            query = query.offset(start).limit(size)

        models = query.all()

        return models

    def get_all_by_model_id(
        self,
        db: Session,
        model_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = Order.DESC,
    ) -> Optional[List[BenchmarkTask]]:
        return self._get_tasks(
            db=db,
            condition=self.model.input_model_id == model_id,
            start=start,
            size=size,
            order=order,
        )

    def get_all_by_converted_models(self, db: Session, converted_model_ids: List[str]) -> List[BenchmarkTask]:
        """Get all benchmark tasks for given converted model IDs ordered by updated_at desc.

        Args:
            db: Database session
            converted_model_ids: List of converted model IDs

        Returns:
            List[BenchmarkTask]: List of benchmark tasks ordered by updated_at desc
        """
        return (
            db.query(self.model)
            .filter(self.model.input_model_id.in_(converted_model_ids))
            .order_by(desc(self.model.updated_at))
            .all()
        )


benchmark_task_repository = BenchmarkTaskRepository(BenchmarkTask)
