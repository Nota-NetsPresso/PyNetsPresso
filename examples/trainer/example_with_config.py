from netspresso.trainer import (
    ModelTrainer,
    Task,
    Backbone,
    Head,
    Format,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomCutmix,
)


# 1. Declare Trainer
trainer = ModelTrainer(task=Task.Object_Detection)

# 2. Set Config
# 2-1. Data
data = trainer.set_dataset_config(
    name="traffic_sign_config_example",
    root_path="/root/traffic-sign",
    train_image="images/train",
    train_label="labels/train",
    valid_image="image/valid",
    valid_label="labels/valid",
    id_mapping=["prohibitory", "danger", "mandatory", "other"],
)

# 2-2. Model
model = trainer.set_model_config(backbone=Backbone.EfficientFormer, head=Head.FC)

# 2-3. Augmentation
augmentation = trainer.set_augmentation_config(
    img_size=256, transforms=[RandomResizedCrop(size=256), RandomHorizontalFlip()], mix_transforms=[RandomCutmix()]
)

# 2-4. Training
training = trainer.set_training_config(epochs=10, batch_size=32)

# 3. Train
gpus = "0, 1"  # multi-gpu
trainer.train(gpus=gpus, data=data, model=model, augmentation=augmentation, training=training)
