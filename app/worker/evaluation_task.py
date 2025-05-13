import logging
import os
from typing import List

from celery import chain, signature

from app.api.v1.schemas.task.train.dataset import DatasetCreate
from app.api.v1.schemas.task.train.environment import EnvironmentCreate
from app.api.v1.schemas.task.train.hyperparameter import HyperparameterCreate
from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.services.training_task import train_task_service
from app.worker.celery_app import celery_app
from netspresso import NetsPresso
from netspresso.enums.metadata import Status
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
            train_transforms=[Resize(), Pad(fill=114), ToTensor(), Normalize()],
            inference_transforms=[Resize(), Pad(fill=114), ToTensor(), Normalize()],
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
        _ = evaluate_model_task.apply_async(
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


@celery_app.task(name='poll_and_start_evaluation')
def poll_and_start_evaluation(
    conversion_task_id: str,
    api_key: str,
    dataset_id: str,
    training_task_id: str,
    confidence_scores: List[float],
    gpus: int = 0,
    evaluation_task_id: str = None
):
    """Poll conversion task status and start evaluation when complete.

    This task is called after a conversion task and checks its status.
    If completed, it starts the evaluation task with the converted model.

    Args:
        conversion_task_id: ID of the conversion task to poll
        api_key: API key for authentication
        dataset_id: ID of the dataset to use for evaluation
        training_task_id: ID of the related training task
        confidence_scores: List of confidence scores to evaluate with
        gpus: Number of GPUs to use
        evaluation_task_id: Optional ID to use for the evaluation task

    Returns:
        evaluation_task_id: ID of the started evaluation task
    """
    netspresso = NetsPresso(api_key=api_key)
    converter = netspresso.converter_v2()
    conversion_task = converter.get_conversion_task(conversion_task_id)

    if conversion_task.status == Status.COMPLETED:
        model_id = conversion_task.output_model_id
        logger.info(f"Conversion completed successfully. Model ID: {model_id}")

        # 생성된 평가 작업 ID가 없으면 생성
        if not evaluation_task_id:
            evaluation_task_id = generate_uuid(entity="task")

        # 변환이 완료되었으므로 평가 실행
        return run_multiple_evaluations.apply_async(
            kwargs={
                "api_key": api_key,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "training_task_id": training_task_id,
                "confidence_scores": confidence_scores,
                "gpus": gpus
            },
            task_id=evaluation_task_id
        ).get()
    elif conversion_task.status in [Status.STOPPED, Status.ERROR]:
        error_message = conversion_task.error_log
        logger.error(f"Conversion failed: {error_message}")
        raise Exception(f"Conversion failed: {error_message}")
    else:
        # 아직 변환이 진행 중이므로 자신을 다시 예약
        logger.info(f"Conversion in progress. Status: {conversion_task.status}. Scheduling poll again.")
        return poll_and_start_evaluation.apply_async(
            args=[
                conversion_task_id,
                api_key,
                dataset_id,
                training_task_id,
                confidence_scores,
                gpus,
                evaluation_task_id
            ],
            countdown=POLLING_INTERVAL
        )

@celery_app.task(name='chain_conversion_and_evaluation')
def chain_conversion_and_evaluation(
    api_key: str,
    input_model_path: str,
    output_dir: str,
    target_framework: str,
    target_device_name: str,
    target_data_type: str,
    target_software_version: str,
    input_layer,
    dataset_path: str,
    input_model_id: str,
    dataset_id: str,
    training_task_id: str,
    confidence_scores: List[float],
    gpus: int = 0
):
    """Chain conversion and evaluation tasks using Celery's chain.

    Args:
        api_key: API key for authentication
        input_model_path: Path to the input model
        output_dir: Directory to output the converted model
        target_framework: Target framework for conversion
        target_device_name: Target device for conversion
        target_data_type: Target data type for conversion
        target_software_version: Target software version for conversion
        input_layer: Input layer specification
        dataset_path: Path to the dataset
        input_model_id: ID of the input model
        dataset_id: ID of the dataset to use for evaluation
        training_task_id: ID of the related training task
        confidence_scores: List of confidence scores to evaluate with
        gpus: Number of GPUs to use

    Returns:
        task_id: Chain task ID
    """
    # 모든 태스크에서 공유할 평가 작업 ID 생성
    evaluation_task_id = generate_uuid(entity="task")
    logger.info(f"Starting conversion and evaluation chain with evaluation ID: {evaluation_task_id}")

    # 변환 태스크 설정
    conversion_task = signature(
        'convert_model',
        kwargs={
            "api_key": api_key,
            "input_model_path": input_model_path,
            "output_dir": output_dir,
            "target_framework": target_framework,
            "target_device_name": target_device_name,
            "target_data_type": target_data_type,
            "target_software_version": target_software_version,
            "input_layer": input_layer,
            "dataset_path": dataset_path,
            "input_model_id": input_model_id,
        }
    )

    # 폴링 태스크 설정 - 변환 완료 체크 후 평가 시작
    poll_task = signature(
        'poll_and_start_evaluation',
        kwargs={
            "api_key": api_key,
            "dataset_id": dataset_id,
            "training_task_id": training_task_id,
            "confidence_scores": confidence_scores,
            "gpus": gpus,
            "evaluation_task_id": evaluation_task_id
        }
    )

    # 체인 생성 및 실행
    task_chain = chain(
        conversion_task,
        poll_task
    )

    # 체인 실행 - 결과는 async로 처리됨
    task_chain.apply_async()

    # evaluation_task_id를 즉시 반환
    return evaluation_task_id
