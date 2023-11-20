from netspresso.trainer import ModelTrainer, Task


# 1. Declare Trainer
trainer = ModelTrainer(task=Task.Image_Classification)

data = "config/data/beans.yaml"
model = "config/model/resnet50-classification.yaml"

# 2. Train
trainer.train(
    gpus="0, 1",
    data=data,
    model=model,
)
