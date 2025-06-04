from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.v1.schemas.base import ResponseItem, ResponsePaginationItems
from netspresso.enums.compression import CompressionMethod, GroupPolicy, LayerNorm, Policy, RecommendationMethod, StepOp
from netspresso.exceptions.compressor import (
    NotValidChannelAxisRangeException,
    NotValidSlampRatioException,
    NotValidVbmfRatioException,
)


class OptionsBase(BaseModel):
    reshape_channel_axis: int = -1

    @field_validator('reshape_channel_axis')
    def validate_reshape_channel_axis(cls, v):
        valid_values = [0, 1, -1, -2]
        if v not in valid_values:
            raise NotValidChannelAxisRangeException(v)
        return v


class Options(OptionsBase):
    policy: Policy = Policy.AVERAGE
    layer_norm: LayerNorm = LayerNorm.STANDARD_SCORE
    group_policy: GroupPolicy = GroupPolicy.AVERAGE
    step_size: int = 2
    step_op: StepOp = StepOp.ROUND
    reverse: bool = False


class RecommendationOptions(Options):
    min_num_of_value: int = 8


class CompressionCreate(BaseModel):
    input_model_id: str = Field(description="Input model ID")
    method: CompressionMethod = Field(description="Compression method")
    recommendation_method: RecommendationMethod = Field(description="Recommendation method")
    ratio: float = Field(description="Compression ratio")
    options: Optional[RecommendationOptions] = Field(default_factory=RecommendationOptions, description="Compression options")

    @field_validator('ratio')
    def validate_ratio(cls, v, values):
        recommendation_method = values.data.get('recommendation_method')
        if recommendation_method == RecommendationMethod.SLAMP and not 0 < v < 1:
            raise NotValidSlampRatioException(ratio=v)
        elif recommendation_method == RecommendationMethod.VBMF and not -1 <= v <= 1:
            raise NotValidVbmfRatioException(ratio=v)
        return v


class ModelResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    size: int
    flops: int
    number_of_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    number_of_layers: Optional[int]
    result_type: str  # 'original' or 'compressed'


class CompressionPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    model_id: Optional[str] = None
    input_model_id: str
    method: CompressionMethod
    ratio: float
    options: RecommendationOptions
    model_results: List[ModelResult]
    related_task_ids: List[str] = Field(default_factory=list)
    user_id: str
    status: str
    is_deleted: bool
    error_detail: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CompressionCreatePayload(BaseModel):
    task_id: str


class CompressionCreateResponse(ResponseItem):
    data: CompressionCreatePayload


class CompressionResponse(ResponseItem):
    data: CompressionPayload


class CompressionsResponse(ResponsePaginationItems):
    data: List[CompressionPayload]
    result_count: int
    total_count: int
