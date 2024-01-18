from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class BaseOptimizer:
    def asdict(self) -> Dict:
        return asdict(self)


@dataclass
class Adadelta(BaseOptimizer):
    name: str = "adadelta"
    lr: float = 1.0
    rho: float = 0.9
    weight_decay: float = 0.0


@dataclass
class Adagrad(BaseOptimizer):
    name: str = "adagrad"
    lr: float = 1e-2
    lr_decay: float = 0.0
    weight_decay: float = 0.0


@dataclass
class Adam(BaseOptimizer):
    name: str = "adam"
    lr: float = 1e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0


@dataclass
class Adamax(BaseOptimizer):
    name: str = "adamax"
    lr: float = 2e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0


@dataclass
class AdamW(BaseOptimizer):
    name: str = "adamw"
    lr: float = 1e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.01


@dataclass
class RMSprop(BaseOptimizer):
    name: str = "rmsprop"
    lr: float = 1e-2
    alpha: float = 0.99
    momentum: float = 0.0
    weight_decay: float = 0.0
    eps: float = 1e-8


@dataclass
class SGD(BaseOptimizer):
    name: str = "sgd"
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
