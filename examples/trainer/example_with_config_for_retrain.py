from netspresso import NetsPresso
from netspresso.trainer.optimizers import AdamW
from netspresso.trainer.schedulers import CosineAnnealingWarmRestartsWithCustomWarmUp

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare trainer
HPARAMS_YAML_PATH = "./temp/hparams.yaml"
trainer = netspresso.trainer(yaml_path=HPARAMS_YAML_PATH)

# 2. Set config for retraining
# 2-1. FX Model
FX_MODEL_PATH = "./temp/FX_MODEL_PATH.pt"
trainer.set_fx_model(fx_model_path=FX_MODEL_PATH)

# 2-2. Training
optimizer = AdamW(lr=6e-3)
scheduler = CosineAnnealingWarmRestartsWithCustomWarmUp(warmup_epochs=10)
trainer.set_training_config(
    epochs=30,
    batch_size=16,
    optimizer=optimizer,
    scheduler=scheduler,
)

# 3. Train
PROJECT_NAME = "project_retrain_sample"
trainer.train(gpus="0, 1", project_name=PROJECT_NAME)
