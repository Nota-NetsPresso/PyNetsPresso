from netspresso.trainer import ModelTrainer


# 1. Declare Trainer
trainer = ModelTrainer()

# 2. Train
trainer.train(
    gpus="0, 1",
    data="config/data/beans.yaml",
    model="config/model/resnet.yaml",
    augmentation="config/augmentation/resnet.yaml",
    training="config/training/resnet.yaml",
    logging="config/logging.yaml",
    environment="config/environment.yaml",
)
