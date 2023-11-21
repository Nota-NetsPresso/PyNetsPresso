from netspresso.trainer import ModelTrainer, Task, Backbone, Head, Resize


# 1. Declare Trainer
trainer = ModelTrainer(task=Task.Object_Detection)

# 2. Set Config
# 2-1. Data
example_dataset = trainer.set_dataset_config(
    name="traffic_sign_config_example",
    root_path="/root/traffic-sign",
    train_image="images/train",
    train_label="labels/train",
    valid_image="images/valid",
    valid_label="labels/valid",
    id_mapping=["prohibitory", "danger", "mandatory", "other"],
)

# 2-2. Model
example_model = trainer.set_model_config(backbone=Backbone.YOLOX, head=Head.YOLOX_Head)

# 2-3. Augmentation
augmentation = trainer.set_augmentation_config(img_size=512, transforms=[Resize(size=512)])

# 2-4. Training
training = trainer.set_training_config(epochs=10, batch_size=8)

# 3. Train
trainer.train(
    gpus="0, 1", 
    data=example_dataset,
    model=example_model,
    # augmentation=augmentation,
    training=training,
)
