from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class BaseScheduler:
    def asdict(self) -> Dict:
        return asdict(self)

    def to_parameters(self) -> Dict:
        """
        Extract all fields except 'name' as parameters.
        """
        return {k: v for k, v in asdict(self).items() if k != "name"}


@dataclass
class StepLR(BaseScheduler):
    name: str = "step"
    iters_per_phase: int = 1
    gamma: float = 0.1
    end_epoch: int = 80


@dataclass
class PolynomialLRWithWarmUp(BaseScheduler):
    name: str = "poly"
    warmup_epochs: int = 5
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6
    power: float = 1.0
    end_epoch: int = 80


@dataclass
class CosineAnnealingLRWithCustomWarmUp(BaseScheduler):
    name: str = "cosine_no_sgdr"
    warmup_epochs: int = 5
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6
    end_epoch: int = 80


@dataclass
class CosineAnnealingWarmRestartsWithCustomWarmUp(BaseScheduler):
    name: str = "cosine"
    warmup_epochs: int = 10
    warmup_bias_lr: float = 1e-5
    min_lr: float = 1e-6
    iters_per_phase: int = 10


@dataclass
class MultiStepLR(BaseScheduler):
    name: str = "multi_step"
    milestones: List[int] = field(default_factory=lambda: [30, 80])
    gamma: float = 0.1


def get_supported_schedulers() -> List[Dict[str, Any]]:
    """Return a list of supported schedulers with their parameters and default values."""
    schedulers = [
        CosineAnnealingWarmRestartsWithCustomWarmUp(),
        StepLR(),
        PolynomialLRWithWarmUp(),
        CosineAnnealingLRWithCustomWarmUp(),
        MultiStepLR(),
    ]
    return [{"name": scheduler.name, "parameters": scheduler.to_parameters()} for scheduler in schedulers]
