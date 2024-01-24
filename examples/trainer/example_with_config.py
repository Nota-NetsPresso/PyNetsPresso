from netspresso.trainer import Trainer, Task
from netspresso.trainer.optimizers import AdamW
from netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp
from netspresso.trainer.augmentations import Resize

# 1. Declare Trainer
trainer = Trainer(task=Task.OBJECT_DETECTION)

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
print(trainer.available_models)  # ['EfficientFormer', 'YOLOX-S']
trainer.set_model_config(model_name="YOLOX-S", img_size=512)

# 2-3. Augmentation
trainer.set_augmentation_config(
    train_transforms=[Resize()],
    inference_transforms=[Resize()],
)

# 2-4. Training
optimizer = AdamW(lr=6e-3)
scheduler = CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10)
trainer.set_training_config(
    epochs=40,
    batch_size=16,
    optimizer=optimizer,
    scheduler=scheduler,
)

# 3. Train
trainer.train(gpus="0, 1", project_name="project_name_1")
