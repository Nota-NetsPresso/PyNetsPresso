from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.enums.metadata import Status
from netspresso.exceptions.conversion import ConversionTaskIsDeletedException, ConversionTaskNotFoundException
from netspresso.utils.db.models.conversion import ConversionTask
from netspresso.utils.db.repositories.base import BaseRepository, Order, TimeSort


class ConversionTaskRepository(BaseRepository[ConversionTask]):
    def __is_available(self, task: Optional[ConversionTask]) -> ConversionTask:
        if task is None:
            raise ConversionTaskNotFoundException()

        if task.is_deleted:
            raise ConversionTaskIsDeletedException(task_id=task.task_id)

        return task

    def get_by_task_id(self, db: Session, task_id: str) -> ConversionTask:
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
    ) -> List[ConversionTask]:
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

    def get_by_model_id(self, db: Session, model_id: str) -> Optional[ConversionTask]:
        conditions = [self.model.model_id == model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
        )

        return self.__is_available(task=task)

    def get_unique_completed_tasks(self, db: Session, model_id: str) -> List[ConversionTask]:
        """Get unique completed conversion tasks for a model using SQLAlchemy ORM.

        Args:
            db: Database session
            model_id: Model ID to get tasks for

        Returns:
            List[ConversionTask]: List of unique completed conversion tasks
        """
        conditions = [self.model.input_model_id == model_id, self.model.status == Status.COMPLETED]
        group_fields = [self.model.framework, self.model.device_name, self.model.software_version, self.model.precision]
        tasks = self.find_all(
            db=db,
            conditions=conditions,
            group_fields=group_fields,
        )

        return tasks

    def get_latest_conversion_task(
        self,
        db: Session,
        model_id: str,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> Optional[ConversionTask]:
        conditions = [self.model.input_model_id == model_id]
        task = self.find_first(
            db=db,
            conditions=conditions,
            order=order,
            time_sort=time_sort,
        )

        return task


conversion_task_repository = ConversionTaskRepository(ConversionTask)
