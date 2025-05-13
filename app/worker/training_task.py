import os
from typing import Dict

from app.api.v1.schemas.task.train.train_task import TrainingCreate
from app.worker.celery_app import celery_app
from netspresso import NetsPresso
from netspresso.trainer.augmentations.augmentation import Normalize, Pad, Resize, ToTensor
from netspresso.trainer.optimizers.optimizer_manager import OptimizerManager
from netspresso.trainer.schedulers.scheduler_manager import SchedulerManager
from netspresso.trainer.storage.dataforge import Split

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

        os.environ['CUDA_VISIBLE_DEVICES'] = training_in.environment.gpus

        netspresso = NetsPresso(api_key=api_key)
        trainer = netspresso.trainer(task=training_in.task)

        # Get NP_TRAINING_STUDIO_PATH
        dataset_dir = os.path.join(NP_TRAINING_STUDIO_PATH, "datasets")

        # Create datasets directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)

        # Download training dataset from dataforage
        train_dataset_path = trainer.download_dataset_for_training(dataset_uuid=training_in.dataset.train_path, output_dir=dataset_dir)
        train_dataset_info = trainer.get_dataset_info_from_storage(dataset_uuid=training_in.dataset.train_path, split=Split.TRAIN)
        trainer.set_dataset(train_dataset_path, train_dataset_info.dataset.dataset_title)

        # Download evaluation dataset from dataforage
        if training_in.dataset.test_path:
            test_dataset_path = trainer.download_dataset_for_evaluation(dataset_uuid=training_in.dataset.test_path, output_dir=dataset_dir)
            test_dataset_info = trainer.get_dataset_info_from_storage(dataset_uuid=training_in.dataset.test_path, split=Split.TEST)
            trainer.set_test_dataset(test_dataset_path, test_dataset_info.dataset.dataset_title)

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

        training_task_id = trainer.train(
            gpus=training_in.environment.gpus,
            model_name=unique_model_name,
            project_id=training_in.project_id,
            task_id=task_id,
        )

        result = {
            "task_id": training_task_id,
            "status": "completed",
        }
        return result

    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")
