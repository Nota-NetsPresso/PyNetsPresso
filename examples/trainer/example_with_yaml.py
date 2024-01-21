from netspresso.trainer import Trainer, Task


# 1. Declare Trainer
trainer = Trainer(task=Task.IMAGE_CLASSIFICATION)

# 2. Set Config
trainer.set_dataset_config_with_yaml(yaml_path="config/data/beans.yaml")
trainer.set_model_config_with_yaml(yaml_path="config/model/resnet50-classification.yaml")

# 3. Train
trainer.train(gpus="0, 1")
