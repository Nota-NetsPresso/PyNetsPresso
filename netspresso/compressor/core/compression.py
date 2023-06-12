from typing import Any, List
from pydantic import BaseModel, Field


class AvailableLayer(BaseModel):
    """Represents an available layer for compression.

    Attributes:
        name (str): The name of the layer.
        values (List[Any]): The compression parameters for the layer.
        use (bool): The compression selection status for the layer.
        channels (List[int]): The channel information for the layer.
    """

    name: str = Field(..., description="Layer Name")
    values: List[Any] = Field([], description="Compression Parameters")
    use: bool = Field(False, description="Compression Selection Status")
    channels: List[int] = Field([], description="Channel Info")


class CompressionInfo(BaseModel):
    """Represents compression information for a model.

    Attributes:
        compressed_model_id (str): The ID of the compressed model.
        compression_id (str): The ID of the compression.
        compression_method (str): The compression method used.
        available_layers (List[AvailableLayer]): The compressible layers information.

            AvailableLayer Attributes:
                - name (str): The name of the layer.
                - values (List[Any]): The compression parameters for the layer.
                - use (bool): The compression selection status for the layer.
                - channels (List[int]): The channel information for the layer.

        original_model_id (str): The ID of the original model.
    """

    compressed_model_id: str = Field("", description="Compressed Model ID")
    compression_id: str = Field("", description="Compression ID")
    compression_method: str = Field(..., description="Compression Method")
    available_layers: List[AvailableLayer] = Field([], description="Compressible Layers")
    original_model_id: str = Field("", description="Original Model ID")
