import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger
from netspresso_trainer.evaluator_main import evaluation_with_yaml_impl
from sqlalchemy.orm import Session

from app.zenko.storage_handler import ObjectStorageHandler
from netspresso.enums import Status
from netspresso.enums.conversion import EvaluationTargetFramework
from netspresso.enums.project import SubFolder
from netspresso.exceptions.conversion import ConversionTaskNotFoundException
from netspresso.exceptions.evaluation import (
    EvaluationDownloadURLGenerationException,
    EvaluationResultFileNotFoundException,
    EvaluationTaskAlreadyExistsException,
    UnsupportedEvaluationFrameworkException,
)
from netspresso.exceptions.trainer import NotCompletedTrainingException
from netspresso.exceptions.training import TrainingTaskNotFoundException
from netspresso.trainer.trainer import Trainer
from netspresso.trainer.trainer_configs import TrainerConfigs
from netspresso.utils.db.models.evaluation import EvaluationDataset, EvaluationTask
from netspresso.utils.db.models.model import Model
from netspresso.utils.db.repositories.compression import compression_task_repository
from netspresso.utils.db.repositories.conversion import conversion_task_repository
from netspresso.utils.db.repositories.evaluation import evaluation_dataset_repository, evaluation_task_repository
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import get_db_session
from netspresso.utils.file import FileHandler

storage_handler = ObjectStorageHandler()
BUCKET_NAME = "model"
EVALUATION_BUCKET_NAME = "evaluation"


class Evaluator:
    def __init__(self, trainer: Optional[Trainer] = None):
        self.trainer = trainer

    def evaluate(self, model_path: str, confidence_score: float, gpus: int = 0):
        if self.trainer is None:
            raise ValueError("Trainer is required for evaluate method")

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

    def evaluate_from_id(
        self,
        model_id: str,
        confidence_score: float,
        gpus: int = 0,
        evaluation_task_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> str:
        if self.trainer is None:
            raise ValueError("Trainer is required for evaluation")

        logger.info(f"Starting evaluation for model {model_id} with confidence score {confidence_score}")

        # Session management
        external_session = db is not None  # Check if a session was provided externally

        # Use with block for internal session management if no external session is provided
        if external_session:
            return self._evaluate_with_session(
                db=db,
                model_id=model_id,
                confidence_score=confidence_score,
                gpus=gpus,
                evaluation_task_id=evaluation_task_id,
            )
        else:
            with get_db_session() as db:
                return self._evaluate_with_session(
                    db=db,
                    model_id=model_id,
                    confidence_score=confidence_score,
                    gpus=gpus,
                    evaluation_task_id=evaluation_task_id,
                )

    def _check_evaluation_task_status(self, db: Session, model_id: str, dataset_id: str, confidence_score: float):
        evaluation_task = evaluation_task_repository.get_by_model_dataset_and_confidence(
            db=db,
            model_id=model_id,
            dataset_id=dataset_id,
            confidence_score=confidence_score
        )

        if evaluation_task:
            if evaluation_task.status == Status.COMPLETED:
                logger.warning(f"Evaluation task already completed: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, status=Status.COMPLETED)
            elif evaluation_task.status == Status.IN_PROGRESS:
                logger.warning(f"Evaluation task already in progress: {evaluation_task.task_id}")
                raise EvaluationTaskAlreadyExistsException(task_id=evaluation_task.task_id, status=Status.IN_PROGRESS)
            elif evaluation_task.status == Status.ERROR:
                logger.info(f"Retrying failed evaluation task: {evaluation_task.task_id}")
            else:
                # Other status (NOT_STARTED, STOPPED, etc.)
                logger.info(f"Using existing evaluation task with ID: {evaluation_task.task_id}")

    def _evaluate_with_session(
        self,
        db: Session,
        model_id: str,
        confidence_score: float,
        gpus: int = 0,
        evaluation_task_id: Optional[str] = None,
    ) -> str:
        evaluation_task = None  # Initialize so it can be safely referenced in except block

        # try:
        #     self._check_evaluation_task_status(db=db, model_id=model_id, dataset_id=dataset_id, confidence_score=confidence_score)
        # except EvaluationTaskAlreadyExistsException as e:
        #     raise e

        try:
            output_dir = tempfile.mkdtemp(prefix="netspresso_evaluate_")

            # 1. Get model information
            input_model = model_repository.get_by_model_id(db=db, model_id=model_id)
            if not input_model:
                raise Exception(f"Model with ID {model_id} not found")

            # 2. Check if this is a direct ONNX model or converted model
            conversion_task = None
            try:
                conversion_task = conversion_task_repository.get_by_model_id(db=db, model_id=model_id)
            except ConversionTaskNotFoundException:
                logger.info(f"No conversion task found for model {model_id}. Treating as direct ONNX model.")
                # No conversion task for ONNX models

            if conversion_task is None:
                # For ONNX models, model_id is the same as training_task's output_model_id
                try:
                    # Get training task based on model type
                    if input_model.type == SubFolder.COMPRESSED_MODELS:
                        compression_task = compression_task_repository.get_by_model_id(db=db, model_id=input_model.model_id)
                        training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
                    else:
                        training_task = training_task_repository.get_by_model_id(db=db, model_id=input_model.model_id)

                    if training_task is None:
                        raise Exception(f"No training task found for model {model_id}")
                except TrainingTaskNotFoundException:
                    try:
                        # Last attempt: query directly by model_id
                        training_task = training_task_repository.get_by_model_id(db=db, model_id=model_id)
                        if training_task is None:
                            raise Exception(f"No training task found for model {model_id}")
                    except TrainingTaskNotFoundException:
                        raise Exception(f"No training task found for model {model_id}")

                # 3. Check training task is completed
                if training_task.status != Status.COMPLETED:
                    raise NotCompletedTrainingException(training_task_id=training_task.task_id)

            else:
                conversion_task_input_model = model_repository.get_by_model_id(db=db, model_id=conversion_task.input_model_id)
                if conversion_task_input_model.type == SubFolder.COMPRESSED_MODELS:
                    compression_task = compression_task_repository.get_by_model_id(db=db, model_id=conversion_task_input_model.model_id)
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=compression_task.input_model_id)
                else:
                    training_task = training_task_repository.get_by_model_id(db=db, model_id=conversion_task_input_model.model_id)

                # 3. Check training task is completed
                if training_task.status != Status.COMPLETED:
                    raise NotCompletedTrainingException(training_task_id=training_task.task_id)

                # 4. Check conversion framework is supported
                if conversion_task.framework not in [EvaluationTargetFramework.TENSORFLOW_LITE, EvaluationTargetFramework.ONNX]:
                    raise UnsupportedEvaluationFrameworkException(framework=conversion_task.framework)

            # Create task with DB session
            if evaluation_task_id:
                if self.trainer.test_dataset_id:
                    evaluation_task = EvaluationTask(
                        task_id=evaluation_task_id,
                        dataset_id=self.trainer.test_dataset_id,
                        input_model_id=model_id,
                        training_task_id=training_task.task_id,
                        conversion_task_id=conversion_task.task_id if conversion_task else None,
                        confidence_score=confidence_score,
                        status=Status.NOT_STARTED,
                        user_id=input_model.user_id,
                    )
                if self.trainer.test_dataset:
                    evaluation_task = EvaluationTask(
                        task_id=evaluation_task_id,
                        dataset_id=self.trainer.test_dataset.dataset_id,
                        input_model_id=model_id,
                        training_task_id=training_task.task_id,
                        conversion_task_id=conversion_task.task_id if conversion_task else None,
                        confidence_score=confidence_score,
                        status=Status.NOT_STARTED,
                        user_id=input_model.user_id,
                    )
            else:
                evaluation_task = EvaluationTask(
                    dataset_id=self.trainer.test_dataset,
                    input_model_id=model_id,
                    training_task_id=training_task.task_id,
                    conversion_task_id=conversion_task.task_id if conversion_task else None,
                    confidence_score=confidence_score,
                    status=Status.NOT_STARTED,
                    user_id=input_model.user_id,
                )
            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)
            logger.info(f"Created new evaluation task with ID: {evaluation_task.task_id}")

            local_path = self._download_model(input_model, output_dir)

            # Update status - Pass DB session
            evaluation_task.status = Status.IN_PROGRESS
            evaluation_task = evaluation_task_repository.save(db=db, model=evaluation_task)

            logger.info(f"Running evaluation with confidence score: {confidence_score}")
            evaluation_logging_dir = self.evaluate(
                model_path=str(local_path),
                confidence_score=confidence_score,
                gpus=gpus
            )
            logger.info(f"Evaluation completed. Logging directory: {evaluation_logging_dir}")

            # predictions.json file path
            predictions_file = evaluation_logging_dir / "predictions.json"
            logger.info(f"Predictions file: {predictions_file}")

            # Raise error if predictions.json file doesn't exist
            if not predictions_file.exists():
                error_msg = f"Predictions file not found: {predictions_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Upload predictions.json file to evaluation bucket
            object_path = f"{input_model.user_id}/{evaluation_task.task_id}/predictions.json"
            logger.info(f"Uploading predictions.json to storage: {object_path}")
            storage_handler.upload_file_to_s3(
                bucket_name=EVALUATION_BUCKET_NAME,
                local_path=str(predictions_file),
                object_path=object_path
            )

            # Upload result images to evaluation bucket
            result_images_dir = Path(evaluation_logging_dir) / "result_image" / "evaluation"
            VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
            for image_file in result_images_dir.glob("*"):
                if image_file.suffix not in VALID_IMAGE_EXTENSIONS:
                    logger.warning(f"Skipping non-image file: {image_file}")
                    continue
                object_path = f"{input_model.user_id}/{evaluation_task.task_id}/result_images/{image_file.name}"
                logger.info(f"Uploading result image to storage: {object_path}")
                storage_handler.upload_file_to_s3(
                    bucket_name=EVALUATION_BUCKET_NAME,
                    local_path=str(image_file),
                    object_path=object_path
                )

            evaluation_summary_path = Path(evaluation_logging_dir) / "evaluation_summary.json"
            evaluation_summary = FileHandler.load_json(evaluation_summary_path)

            # Update status after evaluation complete - Use the same DB session
            evaluation_task.metrics = evaluation_summary["metrics"]
            evaluation_task.metrics_names = evaluation_summary["metrics_list"]
            evaluation_task.primary_metric = evaluation_summary["primary_metric"]
            evaluation_task.status = Status.COMPLETED
            evaluation_task.results_path = object_path  # Save storage path
            evaluation_task_repository.save(db=db, model=evaluation_task)

            return evaluation_task.task_id

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")

            # Use the same session for error handling
            if evaluation_task is not None:
                try:
                    evaluation_task.status = Status.ERROR
                    evaluation_task.error_detail = {"error": str(e)}
                    evaluation_task_repository.save(db=db, model=evaluation_task)
                except Exception as inner_e:
                    logger.error(f"Failed to update task status: {str(inner_e)}")
            raise e

    def _download_model(self, input_model: Model, output_dir: str) -> str:
        download_dir = Path(output_dir) / "input_model"
        download_dir.mkdir(parents=True, exist_ok=True)

        if input_model.type == SubFolder.TRAINED_MODELS:
            remote_model_path = Path(input_model.object_path) / "model.onnx"
        else:
            remote_model_path = Path(input_model.object_path)
        local_path = download_dir / remote_model_path.name

        logger.info(f"Downloading input model from Zenko: {remote_model_path}")
        storage_handler.download_file_from_s3(
            bucket_name=BUCKET_NAME,
            local_path=str(local_path),
            object_path=str(remote_model_path)
        )
        logger.info(f"Downloaded input model from Zenko: {local_path}")

        return local_path.as_posix()

    def get_predictions_download_url(
        self,
        evaluation_task_id: str,
        expires_in: int = 3600  # Valid for 1 hour
    ) -> str:
        """Creates a presigned URL for downloading the predictions.json file from the evaluation results.

        Args:
            evaluation_task_id: Evaluation task ID
            expires_in: URL expiration time in seconds

        Returns:
            str: Download URL

        Raises:
            EvaluationTaskNotFoundException: When evaluation task is not found
            EvaluationResultFileNotFoundException: When evaluation result file is not found
        """
        with get_db_session() as db:
            # Query evaluation task
            evaluation_task = evaluation_task_repository.get_by_task_id(db=db, task_id=evaluation_task_id)

            # Check if results path exists
            if not evaluation_task.results_path:
                raise EvaluationResultFileNotFoundException(task_id=evaluation_task_id)

            # Set download filename
            download_filename = f"predictions_{evaluation_task_id}.json"

            # Generate presigned URL
            logger.info(f"Generating download URL for evaluation result file: {evaluation_task.results_path}")
            try:
                download_url = storage_handler.get_download_presigned_url(
                    bucket_name=EVALUATION_BUCKET_NAME,
                    object_path=evaluation_task.results_path,
                    download_name=download_filename,
                    expires_in=expires_in
                )
                logger.info(f"Download URL generated successfully: {download_url[:100]}...")
                return download_url
            except Exception as e:
                error_msg = f"Failed to generate download URL: {str(e)}"
                logger.error(error_msg)
                raise EvaluationDownloadURLGenerationException(task_id=evaluation_task_id, error_details=str(e)) from e

    def get_evaluation_tasks(self, db: Session, user_id: str, model_id: str) -> List[EvaluationTask]:
        return evaluation_task_repository.get_all_by_user_id_and_model_id(
            db=db,
            user_id=user_id,
            model_id=model_id
        )

    def count_evaluation_task_by_user_id(self, db: Session, user_id: str, model_id: str) -> int:
        return evaluation_task_repository.count_by_user_id_and_model_id(
            db=db,
            user_id=user_id,
            model_id=model_id
        )

    def get_evaluation_task(self, db: Session, evaluation_id: str) -> EvaluationTask:
        return evaluation_task_repository.get_by_task_id(db=db, task_id=evaluation_id)

    def get_unique_datasets_by_model_id(self, db: Session, user_id: str, model_id: str) -> List[EvaluationDataset]:
        """Get unique dataset IDs used for evaluating a specific model.

        Args:
            db: Database session
            user_id: User ID
            model_id: Model ID

        Returns:
            List[str]: List of unique dataset IDs
        """
        dataset_ids = evaluation_task_repository.get_unique_datasets_by_model_id(
            db=db,
            user_id=user_id,
            model_id=model_id
        )
        evaluation_datasets = evaluation_dataset_repository.get_by_dataset_ids(
            db=db,
            dataset_ids=dataset_ids
        )

        return evaluation_datasets

    def get_completed_evaluation_results_by_model_and_dataset(
        self,
        db: Session,
        user_id: str,
        model_id: str,
        dataset_id: str
    ) -> List[EvaluationTask]:
        """Get evaluation results for a specific model and dataset.

        Args:
            db: Database session
            user_id: User ID
            model_id: Model ID
            dataset_id: Dataset ID

        Returns:
            List[EvaluationTask]: List of evaluation tasks with results
        """
        return evaluation_task_repository.get_all_by_model_and_dataset(
            db=db,
            user_id=user_id,
            model_id=model_id,
            dataset_id=dataset_id
        )
