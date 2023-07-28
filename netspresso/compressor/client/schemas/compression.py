from typing import Dict, List, Any

from pydantic import BaseModel, Field, root_validator

from netspresso.compressor.client.utils.validator import CompressionParamsValidator
from netspresso.compressor.client.utils.enum import (
    compression_literal,
    recommendation_literal,
    policy_literal,
    grouppolicy_literal,
    layernorm_literal,
    Policy,
    GroupPolicy,
    LayerNorm,
)


class OptionsBase(BaseModel):
    reshape_channel_axis: int = Field(-1, description="Reshape Channel Axis")


class Options(OptionsBase):
    policy: policy_literal = Field(Policy.AVERAGE, description="Policy")
    layer_norm: layernorm_literal = Field(LayerNorm.TSS_NORM, description="layer Norm")
    group_policy: grouppolicy_literal = Field(GroupPolicy.COUNT, description="Group Policy")


class CreateCompressionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model Name")
    description: str = Field("", description="Description")
    compression_method: compression_literal = Field(..., description="Compression Method")
    options: Options = Field(default_factory=Options, description="Compression Options")


class AvailableLayerBase(BaseModel):
    name: str = Field(..., description="Layer Name")
    values: List[Any] = Field([], description="Compression Parameters")


class AvailableLayer(AvailableLayerBase):
    use: bool = Field(False, description="Compression Selection Status")
    channels: List[int] = Field([], description="Channel Info")


class CompressionResponse(BaseModel):
    new_model_id: str = Field(..., description="Model ID")
    compression_id: str = Field(..., description="Compression ID")
    compression_method: compression_literal = Field(..., description="Compression Method")
    available_layers: List[AvailableLayer] = Field([], description="Compressible Layers")


class RecommendationRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    compression_id: str = Field(..., description="Compression ID")
    recommendation_method: recommendation_literal = Field(..., description="Recommendation Method")
    recommendation_ratio: float = Field(..., description="Recommendation Ratio")
    options: Options = Field(default_factory=Options, description="Compression Options")

    @root_validator
    def validate_ratio(cls, values):
        method = values.get("recommendation_method")
        ratio = values.get("recommendation_ratio")

        if method in ["slamp"]:
            assert 0 < ratio <= 1, "The ratio range for SLAMP is 0 < ratio < = 1."
        elif method in ["vbmf"]:
            assert -1 <= ratio <= 1, "The ratio range for VBMF is -1 <= ratio <= 1."
        return values


class RecommendationInfo(BaseModel):
    name: str = Field(..., description="Layer Name")
    values: List[Any] = Field(..., description="Recommended Values")


class RecommendationResponse(BaseModel):
    recommended_layers: List[RecommendationInfo] = Field([], description="Recommended Layers")


class CompressionRequest(BaseModel):
    compression_id: str = Field(..., description="Compression ID")
    compression_method: compression_literal = Field(..., description="Compression Method")
    layers: List[AvailableLayer] = Field([], description="Compressible Layers")
    options: Options = Field(default_factory=Options, description="Compression Options")
    compressed_model_id: str = Field(..., description="Compressed Model ID")

    @root_validator
    def validate_compression_params(cls, values):
        compression_method = values.get("compression_method")
        layers = values.get("layers")

        compression_params_validator = CompressionParamsValidator(compression_method, layers)
        compression_params_validator.validate()

        return values


class AutoCompressionRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    model_name: str = Field(..., description="Model Name")
    description: str = Field("", description="Description")
    recommendation_ratio: float = Field(..., gt=0, le=1, description="Recommendation Ratio")
    save_path: str = Field(..., description="Compressed Model Save Path")


class UploadDatasetRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    file_path: str = Field(..., description="Model Path")


class GetAvailableLayersRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    compression_method: compression_literal = Field(..., description="Compression Method")
    options: Options = Field(default_factory=Options, description="Compression Options")


class GetAvailableLayersReponse(BaseModel):
    compression_method: compression_literal = Field(..., description="Compression Method")
    available_layers: List[AvailableLayer] = Field([], description="Compressible Layers")
