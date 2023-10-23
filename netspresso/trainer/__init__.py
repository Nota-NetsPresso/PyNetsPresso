from netspresso_trainer import trainer, train, set_arguments


class ModelTrainerV1:
    def __init__(self, data, augmentation, model, training, logging, environment) -> None:
        self.data = data
        self.augmentation = augmentation
        self.model = model
        self.training = training
        self.logging = logging
        self.environment = environment

    def train():
        trainer()


class ModelTrainerV2:
    def __init__(self, is_graphmodule_training) -> None:
        self.is_graphmodule_training = is_graphmodule_training

    def train():
        args_parsed, args = set_arguments(is_graphmodule_training=False)
        train(args_parsed, args, is_graphmodule_training=False)
