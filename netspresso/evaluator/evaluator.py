import tempfile
from pathlib import Path

from loguru import logger
from netspresso_trainer.evaluator_main import evaluation_with_yaml_impl

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.enums import Status
from netspresso.trainer.trainer import Trainer
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.utils.db.models.evaluation import EvaluationResult, EvaluationTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.evaluation import evaluation_result_repository, evaluation_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.session import get_db_session

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"

class Evaluator:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def get_input_model(self, input_model_id: str) -> Model:
        """Get model by ID.

        Args:
            input_model_id: ID of the model to retrieve

        Returns:
            Model object
        """
        with get_db_session() as db:
            input_model = model_repository.get_by_model_id(db=db, model_id=input_model_id)
            return input_model

    def evaluate(self, model_path: str, confidence_score: float, gpus: int = 0):
        try:
            self.trainer.model.checkpoint.path = model_path
            self.trainer.environment.batch_size = 1
            self.trainer.model.postprocessor["params"]["score_thresh"] = confidence_score

            self.configs = TrainerConfigs(
                self.trainer.data,
                self.trainer.augmentation,
                self.trainer.model,
                self.trainer.training,
                self.trainer.logging,
                self.trainer.environment,
            )

            evaluation_logging_dir = evaluation_with_yaml_impl(
                gpus=gpus,
                data=self.configs.data.as_posix(),
                augmentation=self.configs.augmentation.as_posix(),
                model=self.configs.model.as_posix(),
                logging=self.configs.logging.as_posix(),
                environment=self.configs.environment.as_posix(),
            )

            return evaluation_logging_dir

        except Exception as e:
            raise e

    def create_evaluate_task(
        self,
        dataset_id: str,
        input_model_id: str,
        training_task_id: str,
        conversion_task_id: str,
    ) -> EvaluationTask:
        with get_db_session() as db:
            evaluation_task = EvaluationTask(
                dataset_id=dataset_id,
                input_model_id=input_model_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
                status=Status.NOT_STARTED,
            )
            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)
            return evaluation_task

    def create_evaluation_result(
        self,
        evaluation_task_id: str,
        confidence_score: float,
        status: Status
    ) -> EvaluationResult:
        with get_db_session() as db:
            evaluation_result = EvaluationResult(
                evaluation_task_id=evaluation_task_id,
                confidence_score=confidence_score,
                status=status
            )
            evaluation_result = evaluation_result_repository.save(db=db, model=evaluation_result)
            return evaluation_result

    def evaluate_from_id(
        self,
        model_id: str,
        dataset_id: str,
        training_task_id: str,
        conversion_task_id: str,
        confidence_score: float,
        gpus: int = 0
    ) -> EvaluationTask:
        output_dir = tempfile.mkdtemp(prefix="netspresso_convert_")

        input_model: Model = self.get_input_model(input_model_id=model_id)

        # Download model to temporary directory
        download_dir = Path(output_dir) / "input_model"
        download_dir.mkdir(parents=True, exist_ok=True)

        remote_model_path = Path(input_model.object_path)
        local_path = download_dir / remote_model_path.name

        logger.info(f"Downloading input model from Zenko: {remote_model_path}")
        storage_handler.download_file_from_s3(
            bucket_name=BUCKET_NAME,
            local_path=str(local_path),
            object_path=str(remote_model_path)
        )
        logger.info(f"Downloaded input model from Zenko: {local_path}")

        # 모델 ID와 데이터셋 ID로 기존 태스크 조회
        evaluation_task = None
        with get_db_session() as db:
            evaluation_task = evaluation_task_repository.get_by_model_and_dataset(
                db=db,
                model_id=model_id,
                dataset_id=dataset_id
            )

        # 기존 태스크가 없으면 새로 생성
        if not evaluation_task:
            evaluation_task = self.create_evaluate_task(
                dataset_id=dataset_id,
                input_model_id=model_id,
                training_task_id=training_task_id,
                conversion_task_id=conversion_task_id,
            )

        # 현재 신뢰도 점수에 대한 결과 조회 또는 생성
        evaluation_result = None
        with get_db_session() as db:
            evaluation_result = evaluation_result_repository.get_by_task_id_and_confidence_score(
                db=db,
                evaluation_task_id=evaluation_task.task_id,
                confidence_score=confidence_score
            )

            if not evaluation_result:
                # 결과가 없으면 생성
                evaluation_result = self.create_evaluation_result(
                    evaluation_task_id=evaluation_task.task_id,
                    confidence_score=confidence_score,
                    status=Status.NOT_STARTED
                )

        # 평가 수행
        evaluation_logging_dir = self.evaluate(model_path=str(local_path), confidence_score=confidence_score, gpus=gpus)

        print(evaluation_logging_dir)

        # # 결과 업데이트
        # with get_db_session() as db:
        #     result = evaluation_result_repository.get_by_result_id(db=db, result_id=evaluation_result.result_id)
        #     if result:
        #         result.metric_unit = evaluation_output.get("metric_unit")
        #         result.metric_value = evaluation_output.get("metric_value")
        #         result.results_path = evaluation_output.get("results_path")
        #         result.status = Status.COMPLETED
        #         db.commit()

        return evaluation_task
