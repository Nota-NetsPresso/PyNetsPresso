from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from netspresso.enums.compression import GroupPolicy, LayerNorm, Policy, StepOp
from netspresso.enums.device import DeviceName
from netspresso.exceptions.compressor import NotValidChannelAxisRangeException


@dataclass
class OptionsBase:
    reshape_channel_axis: int = -1

    def __post_init__(self):
        valid_values = [0, 1, -1, -2]
        if self.reshape_channel_axis not in valid_values:
            raise NotValidChannelAxisRangeException(self.reshape_channel_axis)


@dataclass
class Options(OptionsBase):
    policy: Policy = Policy.AVERAGE
    layer_norm: LayerNorm = LayerNorm.STANDARD_SCORE
    group_policy: GroupPolicy = GroupPolicy.AVERAGE
    step_size: int = 2
    step_op: StepOp = StepOp.ROUND
    reverse: bool = False
    target_device_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        # Apply hardware-specific step size configuration
        if self.target_device_name is not None:
            self._apply_hardware_config()

    def _apply_hardware_config(self):
        """Apply hardware-specific step size and operation configuration for target device."""
        if self.target_device_name is None:
            return

        # Normalize device name (case-insensitive matching)
        device_name_lower = self.target_device_name.lower()

        # Check if target device is NXP iMX93
        if device_name_lower in [DeviceName.NXP_iMX93.lower(), DeviceName.NXP_iMX93.value.lower()]:
            self.step_size = 8
            self.step_op = StepOp.ROUND_DOWN
            logger.info(f"🎯 Hardware-aware compression enabled for {DeviceName.NXP_iMX93.value}")
            logger.info(f"   → step_size: {self.step_size}, step_op: {self.step_op.value}")
            logger.info("   → Reason: NPU requires 8-channel alignment for optimal performance")


@dataclass
class RecommendationOptions(Options):
    min_num_of_value: int = 8


@dataclass
class Layer:
    use: bool = False
    name: str = "input"
    channels: List[int] = field(default_factory=list)
    values: List[int] = field(default_factory=list)
