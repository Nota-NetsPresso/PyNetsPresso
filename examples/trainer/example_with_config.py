from netspresso.trainer import ModelTrainer, Task, Backbone, Head, Resize


# 1. Declare Trainer
trainer = ModelTrainer(task=Task.OBJECT_DETECTION)

# 2. Set Config
# 2-1. Data
trainer.set_dataset_config(
    name="traffic_sign_config_example",
    root_path="/root/traffic-sign",
    train_image="images/train",
    train_label="labels/train",
    valid_image="images/valid",
    valid_label="labels/valid",
    id_mapping=["prohibitory", "danger", "mandatory", "other"],
)

# 2-2. Model
trainer.set_model_config(backbone=Backbone.CSPDarkNet, head=Head.YOLOX_Head)

# 2-3. Augmentation
trainer.set_augmentation_config(img_size=512, transforms=[Resize(size=512)])

# 2-4. Training
trainer.set_training_config(epochs=40, batch_size=16, lr=6e-3, opt="adamw", warmup_epochs=10)

# 3. Train
trainer.train(gpus="0, 1")
