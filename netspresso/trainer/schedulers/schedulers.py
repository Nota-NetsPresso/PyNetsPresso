from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class BaseScheduler:
    def asdict(self) -> Dict:
        return asdict(self)


@dataclass
class StepLR(BaseScheduler):
    name: str = "step"
    iters_per_phase: int = 1
    gamma: float = 0.1


@dataclass
class PolynomialLRWithWarmUp(BaseScheduler):
    name: str = "poly"
    warmup_epochs: int = 5
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6
    power: float = 1.0


@dataclass
class CosineAnnealingLRWithCustomWarmUp(BaseScheduler):
    name: str = "cosine_no_sgdr"
    warmup_epochs: int = 5
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6


@dataclass
class CosineAnnealingWarmRestartsWithCustomWarmUp(BaseScheduler):
    name: str = "cosine"
    warmup_epochs: int = 5
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6
    iters_per_phase: int = 10
