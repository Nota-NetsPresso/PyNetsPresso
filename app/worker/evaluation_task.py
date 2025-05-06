import os
from celery import chain

from app.api.v1.schemas.task.train.dataset import DatasetCreate
from app.api.v1.schemas.task.train.environment import EnvironmentCreate
from app.api.v1.schemas.task.train.hyperparameter import HyperparameterCreate
from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.worker.celery_app import celery_app
from app.services.training_task import train_task_service
from netspresso import NetsPresso
from netspresso.enums import Status
from netspresso.utils.db.models.evaluation import EvaluationResult, EvaluationTask
from netspresso.utils.db.repositories.evaluation import evaluation_result_repository, evaluation_task_repository
from netspresso.utils.db.session import SessionLocal
import logging

POLLING_INTERVAL = 30  # seconds
logger = logging.getLogger(__name__)


@celery_app.task
def evaluate_model_task(
    api_key: str,
    model_id: str,
    dataset_id: str,
    training_task_id: str,
    conversion_task_id: str,
    confidence_score: float,
    evaluation_task_id: str = None,
    result_id: str = None,
    gpus: int = 0,
):
    """특정 confidence score에 대한 평가를 수행하는 Celery 태스크

    Args:
        api_key: 인증용 API 키
        model_id: 평가할 모델 ID
        dataset_id: 평가에 사용할 데이터셋 ID
        training_task_id: 관련 학습 태스크 ID
        conversion_task_id: 관련 변환 태스크 ID
        confidence_score: 평가에 사용할 신뢰도 점수 (0.3, 0.5, 0.6 중 하나)
        evaluation_task_id: 평가 태스크 ID (선택적)
        result_id: 업데이트할 평가 결과 ID (선택적)
        gpus: 사용할 GPU 수

    Returns:
        result_id: 생성된 평가 결과 ID
    """
    session = SessionLocal()
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
        
        netspresso = NetsPresso(api_key=api_key)
        training_task = train_task_service.get_training_task(db=session, task_id=training_task_id, api_key=api_key)

        # 학습 과제에서 트레이너 인스턴스 가져오기
        trainer = netspresso.trainer(task=training_task.task.name)
        
        # pretrained_model 값 안전하게 추출
        pretrained_model_name = None
        if hasattr(training_task.pretrained_model, 'name'):
            # 객체인 경우 name 속성 추출
            pretrained_model_name = training_task.pretrained_model.name
        else:
            # 문자열인 경우 그대로 사용
            pretrained_model_name = training_task.pretrained_model
            
        logger.info(f"Using pretrained model: {pretrained_model_name}")
        
        training_in = TrainingCreate(
            pretrained_model=pretrained_model_name,
            task=training_task.task.name,
            input_shapes=training_task.input_shapes,
            dataset=DatasetCreate(
                train_path=training_task.dataset.train_path,
                valid_path=training_task.dataset.valid_path,
                test_path=training_task.dataset.valid_path,
            ),
            hyperparameter=HyperparameterCreate(
                epochs=training_task.hyperparameter.epochs,
                batch_size=training_task.hyperparameter.batch_size,
                learning_rate=training_task.hyperparameter.learning_rate,
                optimizer=training_task.hyperparameter.optimizer.name,
                scheduler=training_task.hyperparameter.scheduler.name,
            ),
            environment=EnvironmentCreate(
                gpus=training_task.environment.gpus,
            ),
            project_id="",
            name="",
        )
        trainer = train_task_service._setup_trainer(trainer, training_in)

        # 평가기 생성
        evaluator = netspresso.evaluator(trainer=trainer)

        # 먼저 기존 평가 태스크를 조회 또는 생성
        if not evaluation_task_id:
            # 태스크 ID가 없는 경우 모델 ID와 데이터셋 ID로 검색
            evaluation_task = evaluation_task_repository.get_by_model_and_dataset(
                db=session,
                model_id=model_id,
                dataset_id=dataset_id
            )
            
            if not evaluation_task:
                # 태스크가 없다면 새로 생성
                evaluation_task = EvaluationTask(
                    dataset_id=dataset_id,
                    input_model_id=model_id,
                    training_task_id=training_task_id,
                    conversion_task_id=conversion_task_id,
                    status=Status.NOT_STARTED
                )
                session.add(evaluation_task)
                session.commit()
                evaluation_task_id = evaluation_task.task_id
            else:
                evaluation_task_id = evaluation_task.task_id

        # 특정 결과 ID가 제공된 경우 해당 결과 레코드를 가져옴
        if result_id:
            evaluation_result = evaluation_result_repository.get_by_result_id(db=session, result_id=result_id)
        else:
            # 결과 ID가 없는 경우 태스크 ID와 신뢰도 점수로 검색
            evaluation_result = evaluation_result_repository.get_by_task_id_and_confidence_score(
                db=session,
                evaluation_task_id=evaluation_task_id,
                confidence_score=confidence_score
            )

        if not evaluation_result:
            # 결과 레코드가 없다면 새로 생성
            evaluation_result = EvaluationResult(
                evaluation_task_id=evaluation_task_id,
                confidence_score=confidence_score,
                status=Status.IN_PROGRESS
            )
            session.add(evaluation_result)
            session.commit()
        else:
            # 결과 레코드가 있다면 상태 업데이트
            evaluation_result.status = Status.IN_PROGRESS
            session.commit()

        # 실제 평가 수행
        try:
            evaluation_output = evaluator.evaluate_from_id(
                model_id=model_id,
                dataset_id=dataset_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
                confidence_score=confidence_score,
                gpus=gpus
            )

            # 평가 결과 업데이트
            evaluation_result.status = Status.COMPLETED
            session.commit()
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            evaluation_result.status = Status.ERROR
            evaluation_result.error_detail = str(e)
            session.commit()
            raise

        # 부모 태스크의 전체 상태 업데이트
        update_parent_task_status(db=session, evaluation_task_id=evaluation_task_id)

        return evaluation_result.result_id
    except Exception as e:
        logger.error(f"Evaluation task error: {str(e)}")
        raise
    finally:
        session.close()


def update_parent_task_status(db, evaluation_task_id):
    """
    모든 결과의 상태를 확인하여 부모 태스크의 상태를 업데이트
    
    Args:
        db: DB 세션
        evaluation_task_id: 평가 태스크 ID
    """
    # 모든 결과 조회
    results = evaluation_result_repository.get_by_evaluation_task_id(
        db=db,
        evaluation_task_id=evaluation_task_id
    )

    if not results:
        return

    # 부모 태스크 조회
    parent_task = evaluation_task_repository.get_by_task_id(
        db=db,
        task_id=evaluation_task_id
    )

    if not parent_task:
        return

    # 모든 결과 완료 시 COMPLETED, 일부 완료 시 PARTIALLY_COMPLETED, 모두 실패 시 FAILED
    if all(result.status == Status.COMPLETED for result in results):
        parent_task.status = Status.COMPLETED
    elif any(result.status == Status.COMPLETED for result in results):
        parent_task.status = Status.IN_PROGRESS
    elif all(result.status == Status.ERROR for result in results):
        parent_task.status = Status.ERROR

    db.commit()


@celery_app.task
def run_multiple_evaluations(
    api_key: str,
    model_id: str,
    dataset_id: str,
    training_task_id: str, 
    conversion_task_id: str,
    gpus: int = 0
):
    """여러 신뢰도 점수에 대한 평가를 순차적으로 실행하는 태스크

    Args:
        api_key: 인증용 API 키
        model_id: 평가할 모델 ID
        dataset_id: 평가에 사용할 데이터셋 ID
        training_task_id: 관련 학습 태스크 ID
        conversion_task_id: 관련 변환 태스크 ID
        gpus: 사용할 GPU 수

    Returns:
        evaluation_task_id: 생성된 평가 태스크 ID
    """
    # 평가 태스크 생성
    session = SessionLocal()
    try:
        # 먼저 모델 ID와 데이터셋 ID로 기존 태스크 검색
        evaluation_task = evaluation_task_repository.get_by_model_and_dataset(
            db=session,
            model_id=model_id,
            dataset_id=dataset_id
        )
        
        if not evaluation_task:
            # 태스크가 없다면 새로 생성
            evaluation_task = EvaluationTask(
                dataset_id=dataset_id,
                input_model_id=model_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
                status=Status.NOT_STARTED
            )
            session.add(evaluation_task)
            session.commit()
        
        evaluation_task_id = evaluation_task.task_id
        
        # 신뢰도 점수 목록
        confidence_scores = [0.3, 0.5, 0.6]
        
        # 각 신뢰도 점수에 대한 평가 태스크 체인 생성
        tasks = []
        for score in confidence_scores:
            task = evaluate_model_task.s(
                api_key=api_key,
                model_id=model_id,
                dataset_id=dataset_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
                confidence_score=score,
                evaluation_task_id=evaluation_task_id,
                gpus=gpus
            )
            tasks.append(task)
        
        # 태스크 체인 실행
        chain(*tasks).apply_async()
        
        return evaluation_task_id
    finally:
        session.close()


@celery_app.task
def poll_evaluation_status(api_key: str, task_id: str):
    """평가 태스크의 상태를 확인하고 데이터베이스에서 업데이트합니다.

    Args:
        api_key: 인증용 API 키
        task_id: 확인할 평가 태스크 ID
    """
    session = SessionLocal()
    try:
        # 평가 태스크 상태 확인
        db_task = evaluation_task_repository.get_by_task_id(db=session, task_id=task_id)

        if not db_task:
            # 태스크를 찾을 수 없는 경우 폴링 중단
            return True

        # 평가 결과 상태 확인
        results = evaluation_result_repository.get_by_evaluation_task_id(
            db=session,
            evaluation_task_id=task_id
        )

        # 평가 태스크가 진행 중이거나 시작되지 않은 경우
        if db_task.status in [Status.IN_PROGRESS, Status.NOT_STARTED]:
            # 결과 중 하나라도 진행 중이면 계속 폴링
            if any(result.status in [Status.IN_PROGRESS, Status.NOT_STARTED] for result in results):
                poll_evaluation_status.apply_async(args=[api_key, task_id], countdown=POLLING_INTERVAL)
                return False

            # 모든 결과가 완료되었거나 실패한 경우 부모 태스크 상태 업데이트
            update_parent_task_status(db=session, evaluation_task_id=task_id)

        # 태스크가 완료, 오류, 또는 중지된 경우 폴링 중단
        return True

    finally:
        session.close()