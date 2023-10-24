from netspresso.trainer import ModelTrainerV1, ModelTrainerV2


# 1-1. yaml
'''
    yaml path
'''
data_yaml = "config/data/beans.yaml"
augmentation_yaml = "config/augmentation/resnet.yaml"
model_yaml = "config/model/resnet.yaml"
training_yaml = "config/training/resnet.yaml"
logging_yaml = "config/logging.yaml"
environment_yaml = "config/environment.yaml"
trainer_v1 = ModelTrainerV1(
    data=data_yaml,
    augmentation=augmentation_yaml,
    model=model_yaml,
    training=training_yaml,
    logging=logging_yaml,
    environment=environment_yaml,
)
trainer_v1.train()

# 1-2. function argument
'''
    Data
    Augmentation
    Model
    Training config
    Logging
    Environment
'''
# trainer_v2 = ModelTrainerV2(

# )
# trainer_v2.train()


# 2. fx
# 3. multi GPU
