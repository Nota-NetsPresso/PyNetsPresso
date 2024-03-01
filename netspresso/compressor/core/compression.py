from dataclasses import dataclass, field
from typing import Any, List

from netspresso.enums import GroupPolicy, LayerNorm, Policy, StepOp


@dataclass
class AvailableLayer:
    """Represents an available layer for compression.

    Attributes:
        name (str): The name of the layer.
        values (List[Any]): The compression parameters for the layer.
        use (bool): The compression selection status for the layer.
        channels (List[int]): The channel information for the layer.
    """

    name: str
    values: List[Any] = field(default_factory=list)
    use: bool = field(default=False, repr=False)
    channels: List[int] = field(default_factory=list)


@dataclass
class Options:
    """Represents an options for compression.

    Attributes:
        reshape_channel_axis (int):
        policy (str):
        layer_norm (str):
        group_policy (str):
    """

    reshape_channel_axis: int = -1
    policy: Policy = Policy.AVERAGE
    layer_norm: LayerNorm = LayerNorm.STANDARD_SCORE
    group_policy: GroupPolicy = GroupPolicy.AVERAGE
    step_size: int = 2
    step_op: StepOp = StepOp.ROUND
    reverse: bool = False


@dataclass
class CompressionInfo:
    """Represents compression information for a model.

    Attributes:
        compressed_model_id (str): The ID of the compressed model.
        compression_id (str): The ID of the compression.
        compression_method (str): The compression method used.
        available_layers (List[AvailableLayer]): The compressible layers information.

            AvailableLayer Attributes:
                - name (str): The name of the layer.
                - values (List[Any]): The compression parameters for the layer.
                - channels (List[int]): The channel information for the layer.

        options(Options, optional): The options for pruning method.

        original_model_id (str): The ID of the original model.
    """

    compressed_model_id: str = ""
    compression_id: str = ""
    compression_method: str = ""
    available_layers: List[AvailableLayer] = field(default_factory=list)
    original_model_id: str = ""
    options: Options = field(default_factory=Options)

    def set_available_layers(self, available_layers):
        self.available_layers = [AvailableLayer(**available_layer.dict()) for available_layer in available_layers]
