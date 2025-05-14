import os
from pathlib import Path
from typing import Dict

from loguru import logger

from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.worker.celery_app import celery_app
from app.worker.evaluation_task import chain_conversion_and_evaluation
from netspresso import NetsPresso
from netspresso.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.storage.dataforge import Split
from netspresso.utils.db.repositories.model import model_repository
from netspresso.utils.db.repositories.project import project_repository
from netspresso.utils.db.repositories.training import training_task_repository
from netspresso.utils.db.session import SessionLocal

NP_TRAINING_STUDIO_PATH = os.environ.get("NP_TRAINING_STUDIO_PATH", "/np_training_studio")


@celery_app.task(bind=True, name='train_model')
def train_model(
    self,
    task_id: str,
    api_key: str,
    training_in: Dict,
    unique_model_name: str,
):
    try:
        training_in: TrainingCreate = TrainingCreate.model_validate(training_in)
        logger.info(f"Starting training task: {task_id} for model: {unique_model_name}")

        os.environ['CUDA_VISIBLE_DEVICES'] = training_in.environment.gpus

        netspresso = NetsPresso(api_key=api_key)
        trainer = netspresso.trainer(task=training_in.task)

        # Get NP_TRAINING_STUDIO_PATH
        dataset_dir = os.path.join(NP_TRAINING_STUDIO_PATH, "datasets")

        # Create datasets directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)

        # Download training dataset from dataforage
        logger.info(f"Downloading training dataset: {training_in.dataset.train_path}")
        train_dataset_path = trainer.download_dataset_for_training(dataset_uuid=training_in.dataset.train_path, output_dir=dataset_dir)
        train_dataset_version = trainer.get_dataset_version_from_storage(dataset_uuid=training_in.dataset.train_path, split=Split.TRAIN)
        train_dataset_info = trainer.get_dataset_info_from_storage(project_id=train_dataset_version.project_id, dataset_uuid=training_in.dataset.train_path, split=Split.TRAIN)
        trainer.set_dataset(train_dataset_path, train_dataset_info.dataset.dataset_title)

        img_size = training_in.input_shapes[0].dimension[0]
        logger.info(f"Setting model config with size: {img_size} and model: {training_in.pretrained_model}")
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

        logger.info(f"Starting training with task_id: {task_id}")
        training_task_id = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=unique_model_name,
            project_id=training_in.project_id,
            task_id=task_id,
        )
        logger.info(f"Training completed with task_id: {training_task_id}")

        result = {
            "task_id": training_task_id,
            "status": "completed",
        }

        # If test dataset path is available and conversion is configured, chain conversion and evaluation tasks
        if training_in.dataset.test_path and training_in.conversion:
            try:
                logger.info("Starting post-training chain for conversion and evaluation")

                # Download evaluation dataset from dataforage
                logger.info(f"Downloading test dataset: {training_in.dataset.test_path}")
                test_dataset_path = trainer.download_dataset_for_evaluation(dataset_uuid=training_in.dataset.test_path, output_dir=dataset_dir)
                test_dataset_version = trainer.get_dataset_version_from_storage(dataset_uuid=training_in.dataset.test_path, split=Split.TEST)
                test_dataset_info = trainer.get_dataset_info_from_storage(project_id=test_dataset_version.project_id, dataset_uuid=training_in.dataset.test_path, split=Split.TEST)
                trainer.set_test_dataset(test_dataset_path, test_dataset_info.dataset.dataset_title)

                session = SessionLocal()
                training_task = training_task_repository.get_by_task_id(db=session, task_id=training_task_id)
                model_info = model_repository.get_by_model_id(db=session, model_id=training_task.model_id)

                # Get project information
                project = project_repository.get_by_project_id(db=session, project_id=model_info.project_id)

                # Create input model and output directory paths
                project_abs_path = Path(project.project_abs_path)
                input_model_dir = project_abs_path / model_info.object_path

                input_model_path = input_model_dir / "model.onnx"
                output_dir = input_model_dir / "converted"

                logger.info(f"Input model path: {input_model_path}")
                logger.info(f"Output directory: {output_dir}")

                conversion_option = training_in.conversion
                confidence_scores = [0.3, 0.5, 0.6]
                task_result = chain_conversion_and_evaluation.apply_async(
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
                        "input_model_id": model_info.model_id,
                        "dataset_id": training_in.dataset.test_path,
                        "training_task_id": training_task_id,
                        "confidence_scores": confidence_scores,
                    }
                )
                logger.info("Successfully initiated conversion and evaluation chain")
            except Exception as chain_error:
                logger.error(f"Error in conversion-evaluation chain: {str(chain_error)}")

        return result

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise Exception(f"Training failed: {str(e)}")
