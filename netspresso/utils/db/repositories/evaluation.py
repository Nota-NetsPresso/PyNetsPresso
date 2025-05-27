from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from netspresso.enums.metadata import Status
from netspresso.exceptions.evaluation import EvaluationTaskIsDeletedException, EvaluationTaskNotFoundException
from netspresso.utils.db.models.evaluation import EvaluationDataset, EvaluationTask
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

    def get_by_model_and_dataset(self, db: Session, model_id: str, dataset_id: str) -> Optional[EvaluationTask]:
        """
        Retrieve an evaluation task by model ID and dataset ID.

        Args:
            db: Database session
            model_id: Input model ID
            dataset_id: Dataset ID

        Returns:
            EvaluationTask or None
        """
        conditions = [
            self.model.input_model_id == model_id,
            self.model.dataset_id == dataset_id,
        ]
        return self.find_first(
            db=db,
            conditions=conditions,
        )

    def get_by_model_dataset_and_confidence(
        self,
        db: Session,
        model_id: str,
        dataset_id: str,
        confidence_score: float
    ) -> Optional[EvaluationTask]:
        """
        Retrieve an evaluation task by model ID, dataset ID, and confidence score.

        Args:
            db: Database session
            model_id: Input model ID
            dataset_id: Dataset ID
            confidence_score: Confidence score

        Returns:
            EvaluationTask or None
        """
        conditions = [
            self.model.input_model_id == model_id,
            self.model.dataset_id == dataset_id,
            self.model.confidence_score == confidence_score
        ]
        return self.find_first(
            db=db,
            conditions=conditions,
        )

    def get_all_by_user_id(self, db: Session, user_id: str) -> List[EvaluationTask]:
        conditions = [self.model.user_id == user_id]
        return self.find_all(
            db=db,
            conditions=conditions,
        )

    def count_by_user_id_and_model_id(self, db: Session, user_id: str, model_id: str) -> int:
        count_field = self.model.task_id
        conditions = [
            self.model.user_id == user_id,
            self.model.input_model_id == model_id
        ]

        count = self.count_by_field(db=db, count_field=count_field, conditions=conditions)

        return count

    def get_all_by_user_id_and_model_id(self, db: Session, user_id: str, model_id: str) -> List[EvaluationTask]:
        conditions = [
            self.model.user_id == user_id,
            self.model.input_model_id == model_id,
        ]
        return self.find_all(
            db=db,
            conditions=conditions,
        )

    def get_unique_datasets_by_model_id(self, db: Session, user_id: str, model_id: str) -> List[str]:
        """
        Retrieve a list of unique dataset IDs used for evaluating a specific model.

        Args:
            db: Database session
            user_id: User ID
            model_id: Model ID

        Returns:
            List[str]: List of unique dataset IDs
        """
        # We need to make a custom query to get unique dataset_ids
        query = db.query(
            self.model.dataset_id.distinct()
        ).filter(
            self.model.user_id == user_id,
            self.model.input_model_id == model_id,
            self.model.is_deleted.is_(False)
        )

        # Execute the query and extract dataset IDs
        result = query.all()

        # Convert the result (list of tuples) to a list of strings
        return [item[0] for item in result]

    def get_all_by_model_and_dataset(self, db: Session, user_id: str, model_id: str, dataset_id: str) -> List[EvaluationTask]:
        """
        Retrieve all evaluation tasks for a specific model and dataset.

        Args:
            db: Database session
            user_id: User ID
            model_id: Model ID
            dataset_id: Dataset ID

        Returns:
            List[EvaluationTask]: List of evaluation tasks
        """
        conditions = [
            self.model.user_id == user_id,
            self.model.input_model_id == model_id,
            self.model.dataset_id == dataset_id,
            self.model.status == Status.COMPLETED
        ]

        return self.find_all(
            db=db,
            conditions=conditions,
            time_sort=TimeSort.CREATED_AT,
            order=Order.ASC,
        )

    def get_all_by_model_id(
        self,
        db: Session,
        model_id: str,
        start: Optional[int] = None,
        size: Optional[int] = None,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> List[EvaluationTask]:
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

    def get_latest_evaluation_task(
        self,
        db: Session,
        model_id: str,
        order: Optional[Order] = None,
        time_sort: Optional[TimeSort] = None,
    ) -> Optional[EvaluationTask]:
        conditions = [self.model.input_model_id == model_id]

        return self.find_first(
            db=db,
            conditions=conditions,
            order=order,
            time_sort=time_sort,
        )


class EvaluationDatasetRepository(BaseRepository[EvaluationDataset]):
    def get_by_dataforge_dataset_id(self, db: Session, dataset_id: str) -> Optional[EvaluationDataset]:
        try:
            all_datasets = db.query(self.model).all()
            logger.info(f"All datasets: {all_datasets}")
            for dataset in all_datasets:
                if dataset.storage_info and dataset.storage_info.get('dataset_id') == dataset_id:
                    logger.info(f"Found dataset: {dataset}")
                    return dataset
        except Exception as e:
            logger.warning(f"Failed to query JSON field with SQL function: {str(e)}")
            all_datasets = db.query(self.model).all()
            logger.info(f"All datasets: {all_datasets}")
            for dataset in all_datasets:
                if dataset.storage_info and dataset.storage_info.get('dataset_id') == dataset_id:
                    return dataset
            return None

    def get_by_dataset_path(self, db: Session, dataset_path: str) -> Optional[EvaluationDataset]:
        return db.query(self.model).filter(self.model.path == dataset_path).first()

    def get_by_dataset_ids(self, db: Session, dataset_ids: List[str]) -> List[EvaluationDataset]:
        return db.query(self.model).filter(self.model.dataset_id.in_(dataset_ids)).all()


evaluation_task_repository = EvaluationTaskRepository(EvaluationTask)
evaluation_dataset_repository = EvaluationDatasetRepository(EvaluationDataset)
