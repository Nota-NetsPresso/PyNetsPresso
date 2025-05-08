from typing import Optional

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
        모델 ID와 데이터셋 ID로 평가 태스크를 조회합니다.

        Args:
            db: 데이터베이스 세션
            model_id: 입력 모델 ID
            dataset_id: 데이터셋 ID

        Returns:
            EvaluationTask 또는 None
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
        모델 ID, 데이터셋 ID, 그리고 신뢰도 점수로 평가 태스크를 조회합니다.

        Args:
            db: 데이터베이스 세션
            model_id: 입력 모델 ID
            dataset_id: 데이터셋 ID
            confidence_score: 신뢰도 점수

        Returns:
            EvaluationTask 또는 None
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


evaluation_task_repository = EvaluationTaskRepository(EvaluationTask)
