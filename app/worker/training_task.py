import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.worker.celery_app import celery_app
from netspresso import NetsPresso
from netspresso.enums.conversion import EvaluationTargetFramework
from netspresso.enums.metadata import Status
from netspresso.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.storage.dataforge import Split
from netspresso.trainer.trainer import Trainer
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.project import project_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import get_db_session

# Constants
NP_TRAINING_STUDIO_PATH = Path(os.environ.get("NP_TRAINING_STUDIO_PATH", "/np_training_studio"))
DEFAULT_CONFIDENCE_SCORES = [0.3, 0.5, 0.6]
DEFAULT_AUGMENTATIONS = [Resize(), Pad(fill=114), ToTensor(), Normalize()]


def prepare_training_data(trainer: Trainer, training_in: TrainingCreate) -> Path:
    """Download and prepare training dataset."""
    dataset_dir = NP_TRAINING_STUDIO_PATH / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading training dataset: {training_in.dataset.train_path}")
    train_dataset_path = trainer.download_dataset_for_training(
        dataset_uuid=training_in.dataset.train_path,
        output_dir=dataset_dir.as_posix()
    )

    train_dataset_version = trainer.get_dataset_version_from_storage(
        dataset_uuid=training_in.dataset.train_path,
        split=Split.TRAIN
    )

    train_dataset_info = trainer.get_dataset_info_from_storage(
        project_id=train_dataset_version.project_id,
        dataset_uuid=training_in.dataset.train_path,
        split=Split.TRAIN
    )

    trainer.set_dataset(
        dataset_root_path=train_dataset_path,
        dataset_name=train_dataset_info.dataset.dataset_title,
    )

    return dataset_dir


def configure_model_and_training(trainer: Trainer, training_in: TrainingCreate):
    """Configure model, augmentations, and training parameters."""
    img_size = training_in.input_shapes[0].dimension[0]
    logger.info(f"Setting model config with size: {img_size} and model: {training_in.pretrained_model}")

    trainer.set_model_config(
        model_name=training_in.pretrained_model,
        img_size=img_size
    )

    trainer.set_augmentation_config(
        train_transforms=DEFAULT_AUGMENTATIONS,
        inference_transforms=DEFAULT_AUGMENTATIONS,
    )

    optimizer = OptimizerManager.get_optimizer(
        name=training_in.hyperparameter.optimizer,
        lr=training_in.hyperparameter.learning_rate,
    )

    scheduler = SchedulerManager.get_scheduler(
        name=training_in.hyperparameter.scheduler
    )

    trainer.set_training_config(
        epochs=training_in.hyperparameter.epochs,
        batch_size=training_in.hyperparameter.batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def check_training_result(training_task_id: str) -> bool:
    """Check if training completed successfully.

    Args:
        training_task_id: ID of the training task

    Returns:
        bool: True if training completed successfully, False otherwise
    """
    with get_db_session() as session:
        training_task = training_task_repository.get_by_task_id(db=session, task_id=training_task_id)

        if training_task.status != Status.COMPLETED:
            logger.warning(f"Training task {training_task_id} did not complete successfully. Status: {training_task.status}")
            return False

        return True


def prepare_evaluation_data(trainer: Trainer, training_in: TrainingCreate, dataset_dir: Path):
    """Download and prepare evaluation dataset."""
    logger.info(f"Downloading test dataset: {training_in.dataset.test_path}")

    test_dataset_path = trainer.download_dataset_for_evaluation(
        dataset_uuid=training_in.dataset.test_path,
        output_dir=dataset_dir.as_posix()
    )

    test_dataset_version = trainer.get_dataset_version_from_storage(
        dataset_uuid=training_in.dataset.test_path,
        split=Split.TEST
    )

    test_dataset_info = trainer.get_dataset_info_from_storage(
        project_id=test_dataset_version.project_id,
        dataset_uuid=training_in.dataset.test_path,
        split=Split.TEST
    )

    trainer.set_test_dataset(
        test_dataset_path,
        test_dataset_info.dataset.dataset_title
    )


def get_model_paths(training_task_id: str) -> Optional[Tuple[Path, Path]]:
    """Get paths for model files and directories.

    Returns:
        Tuple containing (input_model_path, output_dir) or None if unsuccessful
    """
    with get_db_session() as session:
        training_task = training_task_repository.get_by_task_id(db=session, task_id=training_task_id)
        model_info = model_repository.get_by_model_id(db=session, model_id=training_task.model_id)
        project = project_repository.get_by_project_id(db=session, project_id=model_info.project_id)

        project_abs_path = Path(project.project_abs_path)
        input_model_dir = project_abs_path / model_info.object_path
        input_model_path = input_model_dir / "model.onnx"
        output_dir = input_model_dir / "converted"

        logger.info(f"Input model path: {input_model_path}")
        logger.info(f"Output directory: {output_dir}")

        if not input_model_path.exists():
            logger.error(f"Model file not found at {input_model_path}")
            return None

        return input_model_path, output_dir


def trigger_conversion_evaluation(
    api_key: str,
    input_model_path: Path,
    output_dir: Path,
    model_id: str,
    training_in: TrainingCreate,
    training_task_id: str
):
    """Trigger the conversion and evaluation chain."""
    from app.worker.evaluation_task import chain_conversion_and_evaluation, run_multiple_evaluations

    conversion_option = training_in.conversion

    if conversion_option.framework == EvaluationTargetFramework.ONNX:
        logger.info("ONNX model detected - skipping conversion and running evaluation directly")

        _ = run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": model_id,
                "dataset_id": training_in.dataset.test_path,
                "training_task_id": training_task_id,
                "confidence_scores": DEFAULT_CONFIDENCE_SCORES,
            }
        )
    else:
        _ = chain_conversion_and_evaluation.apply_async(
            kwargs={
                "api_key": api_key,
                "input_model_path": input_model_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "target_framework": conversion_option.framework,
                "target_device_name": conversion_option.device_name,
                "target_data_type": conversion_option.precision,
                "target_software_version": conversion_option.software_version,
                "input_layer": None,
                "dataset_path": None,
                "input_model_id": model_id,
                "dataset_id": training_in.dataset.test_path,
                "training_task_id": training_task_id,
                "confidence_scores": DEFAULT_CONFIDENCE_SCORES,
            }
        )

    logger.info("Successfully initiated conversion and evaluation process")


@celery_app.task(bind=True, name='train_model')
def train_model(
    self,
    task_id: str,
    api_key: str,
    training_in: Dict[str, Any],
    unique_model_name: str,
) -> Dict[str, Any]:
    """Main training task that orchestrates the training, conversion, and evaluation workflow."""
    try:
        # Parse and validate input
        training_in: TrainingCreate = TrainingCreate.model_validate(training_in)
        logger.info(f"Starting training task: {task_id} for model: {unique_model_name}")

        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = training_in.environment.gpus

        # Initialize NetsPresso
        netspresso = NetsPresso(api_key=api_key)
        trainer = netspresso.trainer(task=training_in.task)

        # Step 1: Prepare training data
        dataset_dir = prepare_training_data(trainer, training_in)

        # Step 2: Configure model and training parameters
        configure_model_and_training(trainer, training_in)

        # Step 3: Execute training
        logger.info(f"Starting training with task_id: {task_id}")
        training_task_id = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=unique_model_name,
            project_id=training_in.project_id,
            task_id=task_id,
        )
        logger.info(f"Training completed with task_id: {training_task_id}")

        # Step 4: Verify training result
        if not check_training_result(training_task_id):
            with get_db_session() as session:
                training_task = training_task_repository.get_by_task_id(db=session, task_id=training_task_id)
                return {"task_id": training_task_id, "status": training_task.status}

        result = {"task_id": training_task_id, "status": "completed"}

        # Step 5: Process conversion and evaluation if training was successful
        if training_in.dataset.test_path and training_in.conversion:
            try:
                logger.info("Starting post-training chain for conversion and evaluation")

                # Step 5.1: Prepare evaluation data
                prepare_evaluation_data(trainer, training_in, dataset_dir)

                # Step 5.2: Get model paths
                paths = get_model_paths(training_task_id)
                if not paths:
                    return result

                input_model_path, output_dir = paths

                # Step 5.3: Get model ID
                with get_db_session() as session:
                    training_task = training_task_repository.get_by_task_id(db=session, task_id=training_task_id)
                    model_id = training_task.model_id

                # Step 5.4: Trigger conversion and evaluation
                trigger_conversion_evaluation(
                    api_key=api_key,
                    input_model_path=input_model_path,
                    output_dir=output_dir,
                    model_id=model_id,
                    training_in=training_in,
                    training_task_id=training_task_id
                )

            except Exception as chain_error:
                logger.error(f"Error in conversion-evaluation chain: {str(chain_error)}")

        return result

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise Exception(f"Training failed: {str(e)}")
