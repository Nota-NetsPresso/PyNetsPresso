from dataclasses import dataclass, field
from typing import Any, List


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


class ModelFactory:
    @staticmethod
    def create_input_shapes(input_layers):
        return [InputShape(**layer.dict()) for layer in input_layers]

    @staticmethod
    def create_model_object(model_info):
        input_shapes = ModelFactory.create_input_shapes(model_info.spec.input_layers)

        model = Model(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            input_shapes=input_shapes,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
        )
        return model

    @staticmethod
    def create_compressed_model_object(model_info):
        input_shapes = [InputShape(**layer.dict()) for layer in model_info.spec.input_layers]

        compressed_model = CompressedModel(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            input_shapes=input_shapes,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
            compression_id=model_info.original_compression_id,
            original_model_id=model_info.original_model_id,
        )
        return compressed_model

    @staticmethod
    def create_model_collection_object(model_info):
        input_shapes = [InputShape(**layer.dict()) for layer in model_info.spec.input_layers]

        model_collection = ModelCollection(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            input_shapes=input_shapes,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
        )
        return model_collection
