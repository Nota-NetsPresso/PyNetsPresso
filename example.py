from netspresso import ModelTrainerV1, ModelTrainerV2


# 1. 학습에 필요한 정보 받아서 처리하기
# 1-1. yaml
'''
    yaml path를 입력
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
trainer_v1.train(is_graphmodule_training=False)

# 1-2. function argument
'''
    Data
    Augmentation
    Model
    Training config
    Logging
    Environment
'''
trainer_v2 = ModelTrainerV2(

)
trainer_v2.train()


# 2. fx에 대한 구분
# 3. 멀티 GPU (이건 내 영역이 아니니 물어보면서 하기)