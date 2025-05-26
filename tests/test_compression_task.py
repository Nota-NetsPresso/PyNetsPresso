import pytest
from pydantic import ValidationError

from app.api.v1.schemas.task.compression.compression_task import (
    CompressionCreate,
    CompressionModelResult,
    CompressionPayload,
    Options,
)
from netspresso.enums.compression import CompressionMethod, GroupPolicy, LayerNorm, Policy, RecommendationMethod, StepOp
from netspresso.exceptions.compressor import (
    NotValidChannelAxisRangeException,
    NotValidSlampRatioException,
    NotValidVbmfRatioException,
)


def test_compression_create_with_default_options():
    # 기본 옵션을 사용하는 경우
    compression = CompressionCreate(
        input_model_id="model_123",
        method=CompressionMethod.PR_L2,
        recommendation_method=RecommendationMethod.SLAMP,
        ratio=0.5
    )

    assert compression.input_model_id == "model_123"
    assert compression.method == CompressionMethod.PR_L2
    assert compression.recommendation_method == RecommendationMethod.SLAMP
    assert compression.ratio == 0.5
    assert isinstance(compression.options, Options)

def test_compression_create_with_custom_options():
    # 커스텀 옵션을 사용하는 경우
    custom_options = Options(
        reshape_channel_axis=1,
        policy=Policy.SUM,
        layer_norm=LayerNorm.STANDARD_SCORE,
        group_policy=GroupPolicy.SUM,
        step_size=4,
        step_op=StepOp.ROUND,
        reverse=True
    )

    compression = CompressionCreate(
        input_model_id="model_456",
        method=CompressionMethod.PR_GM,
        recommendation_method=RecommendationMethod.VBMF,
        ratio=0.7,
        options=custom_options
    )

    assert compression.options.reshape_channel_axis == 1
    assert compression.options.policy == Policy.SUM
    assert compression.options.layer_norm == LayerNorm.STANDARD_SCORE
    assert compression.options.group_policy == GroupPolicy.SUM
    assert compression.options.step_size == 4
    assert compression.options.reverse is True

def test_compression_create_invalid_channel_axis():
    # 잘못된 channel_axis 값으로 테스트
    with pytest.raises(NotValidChannelAxisRangeException):
        custom_options = Options(reshape_channel_axis=3)  # 유효하지 않은 값
        CompressionCreate(
            input_model_id="model_789",
            method=CompressionMethod.PR_L2,
            recommendation_method=RecommendationMethod.SLAMP,
            ratio=0.5,
            options=custom_options
        )

@pytest.mark.parametrize("method,ratio,expected_exception", [
    (RecommendationMethod.SLAMP, 0.0, NotValidSlampRatioException),  # SLAMP: ratio must be 0 < x < 1
    (RecommendationMethod.SLAMP, 1.0, NotValidSlampRatioException),
    (RecommendationMethod.SLAMP, -0.1, NotValidSlampRatioException),
    (RecommendationMethod.VBMF, -1.5, NotValidVbmfRatioException),  # VBMF: ratio must be -1 <= x <= 1
    (RecommendationMethod.VBMF, 1.5, NotValidVbmfRatioException),
])
def test_compression_create_invalid_ratio(method, ratio, expected_exception):
    # 각 recommendation method에 따른 잘못된 ratio 값 테스트
    with pytest.raises(expected_exception):
        CompressionCreate(
            input_model_id="model_789",
            method=CompressionMethod.PR_L2,
            recommendation_method=method,
            ratio=ratio
        )

def test_compression_create_valid_ratio_ranges():
    # SLAMP: 0 < ratio < 1
    compression_slamp = CompressionCreate(
        input_model_id="model_123",
        method=CompressionMethod.PR_L2,
        recommendation_method=RecommendationMethod.SLAMP,
        ratio=0.5
    )
    assert compression_slamp.ratio == 0.5

    # VBMF: -1 <= ratio <= 1
    compression_vbmf = CompressionCreate(
        input_model_id="model_456",
        method=CompressionMethod.PR_GM,
        recommendation_method=RecommendationMethod.VBMF,
        ratio=-0.5
    )
    assert compression_vbmf.ratio == -0.5

    compression_vbmf_boundary = CompressionCreate(
        input_model_id="model_789",
        method=CompressionMethod.PR_GM,
        recommendation_method=RecommendationMethod.VBMF,
        ratio=-1.0  # 경계값 테스트
    )
    assert compression_vbmf_boundary.ratio == -1.0

def test_compression_model_result():
    # CompressionModelResult 모델 테스트
    result = CompressionModelResult(
        size=1000000,  # 1MB
        flops=2000000,  # 2M FLOPs
        number_of_parameters=500000,
        trainable_parameters=400000,
        non_trainable_parameters=100000,
        number_of_layers=50
    )

    assert result.size == 1000000
    assert result.flops == 2000000
    assert result.number_of_parameters == 500000
    assert result.trainable_parameters == 400000
    assert result.non_trainable_parameters == 100000
    assert result.number_of_layers == 50

def test_compression_payload():
    # CompressionPayload 모델 테스트
    model_result = CompressionModelResult(
        size=1000000,
        flops=2000000,
        number_of_parameters=500000,
        trainable_parameters=400000,
        non_trainable_parameters=100000,
        number_of_layers=50
    )

    options = Options(
        reshape_channel_axis=1,
        policy=Policy.SUM,
        layer_norm=LayerNorm.STANDARD_SCORE,
        group_policy=GroupPolicy.SUM,
        step_size=4,
        step_op=StepOp.ROUND,
        reverse=True
    )

    payload = CompressionPayload(
        task_id="task_123",
        model_id="model_456",
        input_model_id="input_model_789",
        method=CompressionMethod.PR_L2,
        recommendation_method=RecommendationMethod.SLAMP,
        ratio=0.5,
        options=options,
        model_results=model_result,
        status="completed",
        is_deleted=False
    )

    # 기본 필드 검증
    assert payload.task_id == "task_123"
    assert payload.model_id == "model_456"
    assert payload.input_model_id == "input_model_789"
    assert payload.method == CompressionMethod.PR_L2
    assert payload.recommendation_method == RecommendationMethod.SLAMP
    assert payload.ratio == 0.5
    assert payload.status == "completed"
    assert payload.is_deleted is False

    # Options 검증
    assert payload.options.reshape_channel_axis == 1
    assert payload.options.policy == Policy.SUM
    assert payload.options.layer_norm == LayerNorm.STANDARD_SCORE
    assert payload.options.group_policy == GroupPolicy.SUM
    assert payload.options.step_size == 4
    assert payload.options.step_op == StepOp.ROUND
    assert payload.options.reverse is True

    # ModelResult 검증
    assert payload.model_results.size == 1000000
    assert payload.model_results.flops == 2000000
    assert payload.model_results.number_of_parameters == 500000
    assert payload.model_results.trainable_parameters == 400000
    assert payload.model_results.non_trainable_parameters == 100000
    assert payload.model_results.number_of_layers == 50

def test_compression_payload_with_minimal_data():
    # 최소한의 필수 데이터로 CompressionPayload 생성 테스트
    model_result = CompressionModelResult(
        size=0,
        flops=0,
        number_of_parameters=0,
        trainable_parameters=0,
        non_trainable_parameters=0,
        number_of_layers=0
    )

    payload = CompressionPayload(
        task_id="task_123",
        input_model_id="input_model_789",
        method=CompressionMethod.PR_L2,
        recommendation_method=RecommendationMethod.SLAMP,
        ratio=0.5,
        options=Options(),  # 기본 옵션 사용
        model_results=model_result,
        status="pending",
        is_deleted=False
    )

    assert payload.task_id == "task_123"
    assert payload.model_id is None  # Optional 필드
    assert payload.input_model_id == "input_model_789"
    assert payload.method == CompressionMethod.PR_L2
    assert payload.status == "pending"
    assert isinstance(payload.options, Options)

def test_compression_payload_invalid_data():
    # 잘못된 데이터로 CompressionPayload 생성 시도
    model_result = CompressionModelResult(
        size=0,
        flops=0,
        number_of_parameters=0,
        trainable_parameters=0,
        non_trainable_parameters=0,
        number_of_layers=0
    )

    # task_id가 없는 경우
    with pytest.raises(ValidationError):
        CompressionPayload(
            input_model_id="input_model_789",
            method=CompressionMethod.PR_L2,
            recommendation_method=RecommendationMethod.SLAMP,
            ratio=0.5,
            options=Options(),
            model_results=model_result,
            status="pending",
            is_deleted=False
        )

    # input_model_id가 없는 경우
    with pytest.raises(ValidationError):
        CompressionPayload(
            task_id="task_123",
            method=CompressionMethod.PR_L2,
            recommendation_method=RecommendationMethod.SLAMP,
            ratio=0.5,
            options=Options(),
            model_results=model_result,
            status="pending",
            is_deleted=False
        )

    # 잘못된 method 값
    with pytest.raises(ValidationError):
        CompressionPayload(
            task_id="task_123",
            input_model_id="input_model_789",
            method="invalid_method",  # 잘못된 method
            recommendation_method=RecommendationMethod.SLAMP,
            ratio=0.5,
            options=Options(),
            model_results=model_result,
            status="pending",
            is_deleted=False
        )
