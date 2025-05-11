import logging
import os
from typing import List

from app.api.v1.schemas.task.train.dataset import DatasetCreate
from app.api.v1.schemas.task.train.environment import EnvironmentCreate
from app.api.v1.schemas.task.train.hyperparameter import HyperparameterCreate
from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.services.training_task import train_task_service
from app.worker.celery_app import celery_app
from netspresso import NetsPresso
from netspresso.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.utils.db.models.base import generate_uuid
from netspresso.utils.db.session import SessionLocal

POLLING_INTERVAL = 30  # seconds
logger = logging.getLogger(__name__)
NP_TRAINING_STUDIO_PATH = os.environ.get("NP_TRAINING_STUDIO_PATH", "/np_training_studio")


@celery_app.task(bind=True, name='evaluate_model_task')
def evaluate_model_task(
    self,
    api_key: str,
    model_id: str,
    dataset_id: str,
    training_task_id: str,
    evaluation_task_id: str,
    confidence_score: float,
    gpus: int = 0,
):
    """Celery task to perform evaluation with a specific confidence score

    Args:
        api_key: API key for authentication
        model_id: ID of the model to evaluate
        dataset_id: ID of the dataset to use for evaluation
        training_task_id: ID of the related training task
        conversion_task_id: ID of the related conversion task
        confidence_score: Confidence score for evaluation (one of 0.3, 0.5, 0.6)
        evaluation_task_id: Evaluation task ID
        gpus: Number of GPUs to use

    Returns:
        result_id: Generated evaluation result ID
    """
    session = SessionLocal()
    try:
        netspresso = NetsPresso(api_key=api_key)
        training_task = train_task_service.get_training_task(db=session, task_id=training_task_id, api_key=api_key)

        # Get trainer instance from the training task
        trainer = netspresso.trainer(task=training_task.task.name)

        logger.info(f"Using pretrained model: {training_task.pretrained_model.name}")

        training_in = TrainingCreate(
            pretrained_model=training_task.pretrained_model.name,
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

        # Get NP_TRAINING_STUDIO_PATH
        dataset_dir = os.path.join(NP_TRAINING_STUDIO_PATH, "datasets")

        # Create datasets directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)

        logger.info(f"Downloading dataset from DataForge: {dataset_id}")
        test_dataset_path = trainer.download_dataset_for_evaluation(dataset_uuid=dataset_id, output_dir=dataset_dir)
        trainer.set_test_dataset(test_dataset_path)
        logger.info(f"Downloaded dataset to: {test_dataset_path}")

        img_size = training_in.input_shapes[0].dimension[0]
        trainer.set_model_config(model_name=training_in.pretrained_model, img_size=img_size)
        trainer.set_augmentation_config(
            train_transforms=[Resize(), Pad(), ToTensor(), Normalize()],
            inference_transforms=[Resize(), Pad(), ToTensor(), Normalize()],
        )
        optimizer = OptimizerManager.get_optimizer(
            name=training_in.hyperparameter.optimizer,
            lr=training_in.hyperparameter.learning_rate,
        )
        scheduler = SchedulerManager.get_scheduler(name=training_in.hyperparameter.scheduler)
        trainer.set_training_config(
            epochs=training_in.hyperparameter.epochs,
            batch_size=training_in.hyperparameter.batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer._apply_img_size()

        # Create evaluator
        evaluator = netspresso.evaluator(trainer=trainer)

        # Perform actual evaluation
        try:
            task_id = evaluator.evaluate_from_id(
                model_id=model_id,
                dataset_id=dataset_id,
                confidence_score=confidence_score,
                gpus=gpus,
                evaluation_task_id=evaluation_task_id,
            )
            result = {"task_id": task_id, "status": "completed"}
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise e

    except Exception as e:
        logger.error(f"Evaluation task error: {str(e)}")
        raise e
    finally:
        session.close()


@celery_app.task(bind=True, name='run_multiple_evaluations')
def run_multiple_evaluations(
    self,
    api_key: str,
    model_id: str,
    dataset_id: str,
    training_task_id: str,
    confidence_scores: List[float],
    gpus: int = 0
):
    """Task to sequentially run evaluations for multiple confidence scores

    Args:
        api_key: API key for authentication
        model_id: ID of the model to evaluate
        dataset_id: ID of the dataset to use for evaluation
        training_task_id: ID of the related training task
        gpus: Number of GPUs to use

    Returns:
        evaluation_task_id: Generated evaluation task ID
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)
    logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpus}")

    # List of confidence scores

    # Run individual tasks for each confidence score (instead of chaining)
    results = []
    for score in confidence_scores:
        # Run each task independently
        evaluation_task_id = generate_uuid(entity="task")
        result = evaluate_model_task.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "training_task_id": training_task_id,
                "evaluation_task_id": evaluation_task_id,
                "confidence_score": score,
                "gpus": gpus
            },
            evaluation_task_id=evaluation_task_id,
        )
        results.append(evaluation_task_id)

    logger.info(f"Evaluation tasks: {results}")

    last_task_id = results[-1] if results else None

    return last_task_id
