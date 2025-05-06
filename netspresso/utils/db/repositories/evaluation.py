from typing import List, Optional

from sqlalchemy.orm import Session

from netspresso.enums.metadata import Status
from netspresso.exceptions.evaluation import EvaluationTaskIsDeletedException, EvaluationTaskNotFoundException
from netspresso.utils.db.models.evaluation import EvaluationResult, EvaluationTask
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
            self.model.dataset_id == dataset_id
        ]
        return self.find_first(
            db=db,
            conditions=conditions,
        )


class EvaluationResultRepository(BaseRepository[EvaluationResult]):
    def get_by_result_id(self, db: Session, result_id: str) -> Optional[EvaluationResult]:
        """
        결과 ID로 평가 결과를 조회합니다.

        Args:
            db: 데이터베이스 세션
            result_id: 결과 ID

        Returns:
            EvaluationResult 또는 None
        """
        conditions = [self.model.result_id == result_id]
        return self.find_first(
            db=db,
            conditions=conditions,
        )

    def get_by_evaluation_task_id(self, db: Session, evaluation_task_id: str) -> List[EvaluationResult]:
        """
        평가 태스크 ID로 모든 결과를 조회합니다.

        Args:
            db: 데이터베이스 세션
            evaluation_task_id: 평가 태스크 ID

        Returns:
            평가 결과 목록
        """
        conditions = [self.model.evaluation_task_id == evaluation_task_id]
        return self.find_all(
            db=db,
            conditions=conditions,
        )

    def get_by_task_id_and_confidence_score(
        self,
        db: Session,
        evaluation_task_id: str,
        confidence_score: float
    ) -> Optional[EvaluationResult]:
        """
        평가 태스크 ID와 신뢰도 점수로 특정 결과를 조회합니다.

        Args:
            db: 데이터베이스 세션
            evaluation_task_id: 평가 태스크 ID
            confidence_score: 신뢰도 점수

        Returns:
            평가 결과 또는 None
        """
        conditions = [
            self.model.evaluation_task_id == evaluation_task_id,
            self.model.confidence_score == confidence_score
        ]
        return self.find_first(
            db=db,
            conditions=conditions,
        )


evaluation_task_repository = EvaluationTaskRepository(EvaluationTask)
evaluation_result_repository = EvaluationResultRepository(EvaluationResult)
