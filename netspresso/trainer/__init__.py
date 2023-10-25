import subprocess


class ModelTrainer:
    def __init__(self, data, augmentation, model, training, logging, environment) -> None:
        self.data = data
        self.augmentation = augmentation
        self.model = model
        self.training = training
        self.logging = logging
        self.environment = environment

    def train(self, nproc_per_node):
        subprocess.run([
            "netspresso-train", "-m", "torch.distributed.launch",
            "--nproc_per_node", str(nproc_per_node),
            "--data", self.data,
            "--augmentation", self.augmentation,
            "--model", self.model,
            "--training", self.training,
            "--logging", self.logging,
            "--environment", self.environment
        ])
