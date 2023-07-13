from dataclasses import dataclass, field
from typing import Any, List

from netspresso.compressor.client.schemas.model import InputLayer, ModelResponse


@dataclass
class InputShape:
    """Represents the shape of an input tensor.

    Attributes:
        batch (int): The batch size of the input tensor.
        channel (int): The number of channels in the input tensor.
        dimension (List[int]): The dimensions of the input tensor.
    """

    batch: int
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
                - batch (int): The batch size of the input tensor.
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
    model_size: float
    flops: float
    trainable_parameters: float
    non_trainable_parameters: float
    number_of_layers: int
    _input_layers: List[InputLayer] = field(repr=False, default_factory=list)
    input_shapes: List[InputShape] = field(default_factory=list)

    def __post_init__(self):
        self.input_shapes = [InputShape(**layer.dict()) for layer in self._input_layers]


@dataclass
class CompressedModel(Model):
    """Represents a compressed model.

    Attributes:
        compression_id (str): The ID of the compression.
        original_model_id (str): The ID of the uploaded model.
    """

    compression_id: str = ""
    original_model_id: str = ""


@dataclass
class ModelCollection(Model):
    """A collection of models that includes the uploaded model and its compressed models.

    Attributes:
        compressed_models (List[CompressedModel]): A list of compressed models compressed from this model.
    """

    _compressed_models: List[ModelResponse] = field(repr=False, default_factory=list)
    compressed_models: List[CompressedModel] = field(default_factory=list)

    def __post_init__(self):
        self.input_shapes = [InputShape(**layer.dict()) for layer in self._input_layers]
        self.compressed_models = [
            CompressedModel(
                model_id=children_model_info.model_id,
                model_name=children_model_info.model_name,
                task=children_model_info.task,
                framework=children_model_info.framework,
                model_size=children_model_info.spec.model_size,
                flops=children_model_info.spec.flops,
                trainable_parameters=children_model_info.spec.trainable_parameters,
                non_trainable_parameters=children_model_info.spec.non_trainable_parameters,
                number_of_layers=children_model_info.spec.number_of_layers,
                compression_id=children_model_info.original_compression_id,
                original_model_id=children_model_info.original_model_id,
                _input_layers=children_model_info.spec.input_layers,
            )
            for children_model_info in self._compressed_models
        ]


class ModelFactory:
    def create_model(self, model_info: ModelResponse) -> Model:
        return Model(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
            _input_layers=model_info.spec.input_layers,
        )

    def create_compressed_model(self, model_info: ModelResponse) -> CompressedModel:
        return CompressedModel(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
            compression_id=model_info.original_compression_id,
            original_model_id=model_info.original_model_id,
            _input_layers=model_info.spec.input_layers,
        )

    def create_model_collection(
        self, model_info: ModelResponse, children_models: List[ModelResponse]
    ) -> ModelCollection:
        return ModelCollection(
            model_id=model_info.model_id,
            model_name=model_info.model_name,
            task=model_info.task,
            framework=model_info.framework,
            model_size=model_info.spec.model_size,
            flops=model_info.spec.flops,
            trainable_parameters=model_info.spec.trainable_parameters,
            non_trainable_parameters=model_info.spec.non_trainable_parameters,
            number_of_layers=model_info.spec.number_of_layers,
            _input_layers=model_info.spec.input_layers,
            _compressed_models=children_models,
        )
