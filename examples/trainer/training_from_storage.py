from netspresso import NetsPresso
from netspresso.enums import Task
from netspresso.trainer.augmentations import Normalize, Pad, Resize, ToTensor
from netspresso.trainer.optimizers import AdamW
from netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare trainer
trainer = netspresso.trainer(task=Task.OBJECT_DETECTION)

# 2. Set config for training
# 2-1. Data
dataset_uuid = "project_PxS20YEZJN"
trainer.download_dataset_for_training(dataset_uuid=dataset_uuid)

# 2-2. Model
print(trainer.available_models)  # ['EfficientFormer', 'YOLOX-S']
trainer.set_model_config(model_name="yolox_s", img_size=512)

# 2-3. Augmentation
trainer.set_augmentation_config(
    train_transforms=[Resize(), Pad(), ToTensor(), Normalize()],
    inference_transforms=[Resize(), Pad(), ToTensor(), Normalize()],
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
PROJECT_NAME = "project_sample"
trainer.train(gpus="0, 1", model_name="dataforge_test", project_id=PROJECT_NAME)
