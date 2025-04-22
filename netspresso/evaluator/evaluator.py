from netspresso_trainer.evaluator_main import evaluation_with_yaml_impl

from netspresso.trainer.trainer import Trainer
from netspresso.trainer.trainer_configs import TrainerConfigs


class Evaluator:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

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
