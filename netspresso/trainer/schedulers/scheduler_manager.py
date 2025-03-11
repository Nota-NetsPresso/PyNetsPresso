from netspresso.enums.train import Scheduler
from netspresso.trainer.schedulers.schedulers import (
    CosineAnnealingLRWithCustomWarmUp,
    CosineAnnealingWarmRestartsWithCustomWarmUp,
    MultiStepLR,
    PolynomialLRWithWarmUp,
    StepLR,
)


class SchedulerManager:
    @staticmethod
    def get_scheduler(name: str):
        scheduler_map = {
            Scheduler.STEP_LR: StepLR,
            Scheduler.POLYNOMIAL_LR: PolynomialLRWithWarmUp,
            Scheduler.COSINE_ANNEALING_LR: CosineAnnealingLRWithCustomWarmUp,
            Scheduler.COSINE_ANNEALING_WARM_RESTARTS: CosineAnnealingWarmRestartsWithCustomWarmUp,
            Scheduler.MULTI_STEP_LR: MultiStepLR,
        }

        scheduler_class = scheduler_map.get(name.lower())
        if not scheduler_class:
            raise ValueError(f"Scheduler '{name}' not found.")

        return scheduler_class()
