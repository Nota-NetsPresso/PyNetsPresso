from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.exceptions.evaluation import EvaluationTaskIsDeletedException, EvaluationTaskNotFoundException
from netspresso.utils.db.models.evaluation import EvaluationTask
from netspresso.utils.db.repositories.base import BaseRepository


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
            self.model.input_model_id == model_id
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
        ]

        return self.find_all(
            db=db,
            conditions=conditions,
        )


evaluation_task_repository = EvaluationTaskRepository(EvaluationTask)
