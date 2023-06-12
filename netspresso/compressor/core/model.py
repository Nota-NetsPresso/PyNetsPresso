from typing import Any, List
from pydantic import BaseModel, Field


class InputShape(BaseModel):
    """Represents the shape of an input tensor.

    Attributes:
        batch (int): The batch size of the input tensor.
        channel (int): The number of channels in the input tensor.
        dimension (List[int]): The dimensions of the input tensor.
    """

    batch: Any = Field(None, description="Input Batch")
    channel: int = Field(0, description="Input Channel")
    dimension: List[int] = Field([0], description="Input Diemension ex) [height, width]")


class Model(BaseModel):
    """Represents a uploaded model.

    Attributes:
        model_id (str): The ID of the model.
        model_name (str): The name of the model.
        task (Task): The task of the model.
        framework (Framework): The framework of the model.
        input_shapes (List[InputShape]): The input shapes of the model.

            InputShape Attributes:
                - batch (int): The batch size of the input tensor.
                - channel (int): The number of channels in the input tensor.
                - dimension (List[int]): The dimensions of the input tensor.

        model_size (float): The size of the model.
        flops (float): The FLOPs (floating point operations) of the model.
        trainable_parameters (float): The number of trainable parameters in the model.
        non_trainable_parameters (float): The number of non-trainable parameters in the model.
        number_of_layers (float): The number of layers in the model.
    """

    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model Name")
    task: str = Field(..., description="Task")
    framework: str = Field(..., description="Framework")
    input_shapes: List[InputShape] = Field([], description="Input Shapes")
    model_size: float = Field(0, description="Model Size")
    flops: float = Field(0, description="FLOPs")
    trainable_parameters: float = Field(0, description="Trainable Parameters")
    non_trainable_parameters: float = Field(0, description="Non Trainable Parameters")
    number_of_layers: int = Field(0, description="Number of Layers")


class CompressedModel(Model):
    """Represents a compressed model.

    Attributes:
        compression_id (str): The ID of the compression.
        original_model_id (str): The ID of the uploaded model.
    """

    compression_id: str = Field("", description="Compression ID")
    original_model_id: str = Field(..., description="Original Model ID")


class ModelCollection(Model):
    """A collection of models that includes the uploaded model and its compressed models.

    Attributes:
        compressed_models (List[CompressedModel]): A list of compressed models compressed from this model.
    """

    compressed_models: List[CompressedModel] = Field([], description="Compressed Models")
