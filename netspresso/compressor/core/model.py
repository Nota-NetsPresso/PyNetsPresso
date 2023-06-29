from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


@dataclass
class InputShape:
    """Represents the shape of an input tensor.

    Attributes:
        batch (Any): The batch size of the input tensor.
        channel (int): The number of channels in the input tensor.
        dimension (List[int]): The dimensions of the input tensor.
    """

    batch: Any
    channel: int
    dimension: List[int]


@dataclass
class Model:
    """Represents a uploaded model.

    Attributes:
        model_id (str): The ID of the model.
        model_name (str): The name of the model.
        task (Task): The task of the model.
        framework (Framework): The framework of the model.
        input_shapes (List[InputShape]): The input shapes of the model.

            InputShape Attributes:
                - batch (Any): The batch size of the input tensor.
                - channel (int): The number of channels in the input tensor.
                - dimension (List[int]): The dimensions of the input tensor.

        model_size (float): The size of the model.
        flops (float): The FLOPs (floating point operations) of the model.
        trainable_parameters (float): The number of trainable parameters in the model.
        non_trainable_parameters (float): The number of non-trainable parameters in the model.
        number_of_layers (float): The number of layers in the model.
    """

    model_id: str
    model_name: str
    task: str
    framework: str
    input_shapes: List[InputShape]
    model_size: float
    flops: float
    trainable_parameters: float
    non_trainable_parameters: float
    number_of_layers: int


@dataclass
class CompressedModel(Model):
    """Represents a compressed model.

    Attributes:
        compression_id (str): The ID of the compression.
        original_model_id (str): The ID of the uploaded model.
    """

    compression_id: str
    original_model_id: str


@dataclass
class ModelCollection(Model):
    """A collection of models that includes the uploaded model and its compressed models.

    Attributes:
        compressed_models (List[CompressedModel]): A list of compressed models compressed from this model.
    """

    compressed_models: List[CompressedModel] = field(default_factory=list)
