from netspresso_trainer import trainer, train, set_arguments
import subprocess


class ModelTrainerV1:
    def __init__(self, data, augmentation, model, training, logging, environment) -> None:
        self.data = data
        self.augmentation = augmentation
        self.model = model
        self.training = training
        self.logging = logging
        self.environment = environment

    def train(self):
        subprocess.run(
            [
                "netspresso-train",
                "--data", self.data,
                "--augmentation", self.augmentation,
                "--model", self.model,
                "--training", self.training,
                "--logging", self.logging,
                "--environment", self.environment,
            ],
            stdout=subprocess.PIPE
        )


class ModelTrainerV2:
    def __init__(self, is_graphmodule_training) -> None:
        self.is_graphmodule_training = is_graphmodule_training

    def train():
        args_parsed, args = set_arguments(is_graphmodule_training=False)
        train(args_parsed, args, is_graphmodule_training=False)
