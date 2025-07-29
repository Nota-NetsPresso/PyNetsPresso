from dataclasses import dataclass, field
from typing import List, Optional, Union

from netspresso.enums.base import StrEnum
from netspresso.np_qai.options.common import CommonOptions


class TfliteDelegates(StrEnum):
    QNN = "qnn"
    QNN_GPU = "qnn-gpu"
    NNAPI = "nnapi"
    NNAPI_GPU = "nnapi-gpu"
    GPU = "gpu"
    XNNPACK = "xnnpack"


class ExecutionMode(StrEnum):
    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"


class GraphOptimizationLevel(StrEnum):
    DISABLE_ALL = "DISABLE_ALL"
    ENABLE_BASIC = "ENABLE_BASIC"
    ENABLE_EXTENDED = "ENABLE_EXTENDED"
    ENABLE_ALL = "ENABLE_ALL"


class OnnxQnnHtpPerformanceMode(StrEnum):
    DEFAULT = "default"
    LOW_POWER_SAVER = "low_power_saver"
    POWER_SAVER = "power_saver"
    HIGH_POWER_SAVER = "high_power_saver"
    LOW_BALANCED = "low_balanced"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"
    BURST = "burst"


class OnnxExecutionProviders(StrEnum):
    QNN = "qnn"
    QNN_GPU = "qnn-gpu"
    DIRECTML = "directml"


class QnnLogLevel(StrEnum):
    K_LOG_OFF = "kLogOff"
    K_LOG_LEVEL_ERROR = "kLogLevelError"
    K_LOG_LEVEL_WARN = "kLogLevelWarn"
    K_LOG_LEVEL_INFO = "kLogLevelInfo"
    K_LOG_LEVEL_VERBOSE = "kLogLevelVerbose"
    K_LOG_LEVEL_DEBUG = "kLogLevelDebug"


class QnnGraphPriority(StrEnum):
    K_QNN_PRIORITY_DEFAULT = "kQnnPriorityDefault"
    K_QNN_PRIORITY_LOW = "kQnnPriorityLow"
    K_QNN_PRIORITY_NORMAL = "kQnnPriorityNormal"
    K_QNN_PRIORITY_NORMAL_HIGH = "kQnnPriorityNormalHigh"
    K_QNN_PRIORITY_HIGH = "kQnnPriorityHigh"
    K_QNN_PRIORITY_UNDEFINED = "kQnnPriorityUndefined"


class QnnGpuPrecision(StrEnum):
    K_GPU_USER_PROVIDED = "kGpuUserProvided"
    K_GPU_FP32 = "kGpuFp32"
    K_GPU_FP16 = "kGpuFp16"
    K_GPU_HYBRID = "kGpuHybrid"


class QnnGpuPerformanceMode(StrEnum):
    K_GPU_DEFAULT = "kGpuDefault"
    K_GPU_HIGH = "kGpuHigh"
    K_GPU_NORMAL = "kGpuNormal"
    K_GPU_LOW = "kGpuLow"


class QnnDspPerformanceMode(StrEnum):
    K_DSP_LOW_POWER_SAVER = "kDspLowPowerSaver"
    K_DSP_POWER_SAVER = "kDspPowerSaver"
    K_DSP_HIGH_POWER_SAVER = "kDspHighPowerSaver"
    K_DSP_LOW_BALANCED = "kDspLowBalanced"
    K_DSP_BALANCED = "kDspBalanced"
    K_DSP_HIGH_PERFORMANCE = "kDspHighPerformance"
    K_DPS_SUSTAINED_HIGH_PERFORMANCE = "kDspSustainedHighPerformance"
    K_DSP_BURST = "kDspBurst"


class QnnDspEncoding(StrEnum):
    K_DSP_STATIC = "kDspStatic"
    K_DSP_DYNAMIC = "kDspDynamic"


class TfliteQnnHtpPerformanceMode(StrEnum):
    K_HTP_LOW_POWER_SAVER = "kHtpLowPowerSaver"
    K_HTP_POWER_SAVER = "kHtpPowerSaver"
    K_HTP_HIGH_POWER_SAVER = "kHtpHighPowerSaver"
    K_HTP_LOW_BALANCED = "kHtpLowBalanced"
    K_HTP_BALANCED = "kHtpBalanced"
    K_HTP_HIGH_PERFORMANCE = "kHtpHighPerformance"
    K_HTP_SUSTAINED_HIGH_PERFORMANCE = "kHtpSustainedHighPerformance"
    K_HTP_BURST = "kHtpBurst"


class QnnHtpPrecision(StrEnum):
    K_HTP_QUANTIZED = "kHtpQuantized"
    K_HTP_FP16 = "kHtpFp16"


class QnnHtpOptimizationStrategy(StrEnum):
    K_HTP_OPTIMIZE_FOR_INFERENCE = "kHtpOptimizeForInference"
    K_HTP_OPTIMIZE_FOR_PREPARE = "kHtpOptimizeForPrepare"


class GpuInferencePreference(StrEnum):
    TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = "TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER"
    TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED = "TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED"
    TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED = "TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED"


class GpuInferencePriority(StrEnum):
    TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED = "TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED"
    TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION = "TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION"
    TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = "TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY"
    TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE = "TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE"


class NnapiExecutionPreference(StrEnum):
    K_LOW_POWER = "kLowPower"
    K_FAST_SINGLE_ANSWER = "kFastSingleAnswer"
    K_SUSTAINED_SPEED = "kSustainedSpeed"


class ContextErrorReportingOptionsLevel(StrEnum):
    BRIEF = "BRIEF"
    DETAILED = "DETAILED"


class Priority(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    NORMAL_HIGH = "NORMAL_HIGH"
    HIGH = "HIGH"


class ContextGpuPerformanceHint(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class ContextHtpPerformanceMode(StrEnum):
    EXTREME_POWER_SAVER = "EXTREME_POWER_SAVER"
    LOW_POWER_SAVER = "LOW_POWER_SAVER"
    POWER_SAVER = "POWER_SAVER"
    HIGH_POWER_SAVER = "HIGH_POWER_SAVER"
    LOW_BALANCED = "LOW_BALANCED"
    BALANCED = "BALANCED"
    HIGH_PERFORMANCE = "HIGH_PERFORMANCE"
    SUSTAINED_HIGH_PERFORMANCE = "SUSTAINED_HIGH_PERFORMANCE"
    BURST = "BURST"


class DefaultGraphGpuPrecision(StrEnum):
    FLOAT32 = "FLOAT32"
    FLOAT16 = "FLOAT16"
    HYBRID = "HYBRID"
    USER_PROVIDED = "USER_PROVIDED"


class DefaultGraphHtpOptimizationType(StrEnum):
    FINALIZE_OPTIMIZATION_FLAG = "FINALIZE_OPTIMIZATION_FLAG"


class DefaultGraphHtpPrecision(StrEnum):
    FLOAT16 = "FLOAT16"


@dataclass
class OnnxOptions:
    execution_mode: Optional[ExecutionMode] = ExecutionMode.SEQUENTIAL
    intra_op_num_threads: Optional[int] = 0
    inter_op_num_threads: Optional[int] = 0
    enable_memory_pattern: Optional[bool] = False
    enable_cpu_memory_arena: Optional[bool] = False
    graph_optimization_level: Optional[GraphOptimizationLevel] = GraphOptimizationLevel.ENABLE_ALL

    def to_cli_string(self) -> str:
        args = []
        if self.execution_mode is not None:
            args.append(f"execution_mode={self.execution_mode}")
        if self.intra_op_num_threads is not None:
            args.append(f"intra_op_num_threads={self.intra_op_num_threads}")
        if self.inter_op_num_threads is not None:
            args.append(f"inter_op_num_threads={self.inter_op_num_threads}")
        if self.enable_memory_pattern is not None:
            args.append(f"enable_memory_pattern={'true' if self.enable_memory_pattern else 'false'}")
        if self.enable_cpu_memory_arena is not None:
            args.append(f"enable_cpu_memory_arena={'true' if self.enable_cpu_memory_arena else 'false'}")
        if self.graph_optimization_level is not None:
            args.append(f"graph_optimization_level={self.graph_optimization_level}")

        return f"--onnx_options {';'.join(args)}"


@dataclass
class OnnxQnnOptions(OnnxOptions):
    qnn_htp_performance_mode: Optional[OnnxQnnHtpPerformanceMode] = OnnxQnnHtpPerformanceMode.BURST
    qnn_htp_graph_optimization_mode: Optional[str] = 3
    qnn_enable_htp_fp16_precision: Optional[str] = 1

    def to_cli_string(self) -> str:
        base_string = super().to_cli_string().split(" ")[1]  # Get base TfliteOptions part
        args = [base_string]
        if self.qnn_htp_performance_mode is not None:
            args.append(f"qnn_htp_performance_mode={self.qnn_htp_performance_mode}")
        if self.qnn_htp_graph_optimization_mode is not None:
            args.append(f"qnn_htp_graph_optimization_mode={self.qnn_htp_graph_optimization_mode}")
        if self.qnn_enable_htp_fp16_precision is not None:
            args.append(f"qnn_enable_htp_fp16_precision={self.qnn_enable_htp_fp16_precision}")

        return f"--onnx_options {';'.join(args)}"


@dataclass
class TfliteOptions:
    enable_fallback: Optional[bool] = True
    invoke_interpreter_on_cold_load: Optional[bool] = False
    allow_fp32_as_fp16: Optional[bool] = True
    force_opengl: Optional[bool] = False
    number_of_threads: Optional[int] = -1
    release_dynamic_tensors: Optional[bool] = False

    def to_cli_string(self) -> str:
        args = []
        if self.enable_fallback is not None:
            args.append(f"enable_fallback={'true' if self.enable_fallback else 'false'}")
        if self.invoke_interpreter_on_cold_load is not None:
            args.append(
                f"invoke_interpreter_on_cold_load={'true' if self.invoke_interpreter_on_cold_load else 'false'}"
            )
        if self.allow_fp32_as_fp16 is not None:
            args.append(f"allow_fp32_as_fp16={'true' if self.allow_fp32_as_fp16 else 'false'}")
        if self.force_opengl is not None:
            args.append(f"force_opengl={'true' if self.force_opengl else 'false'}")
        if self.number_of_threads is not None:
            args.append(f"number_of_threads={self.number_of_threads}")
        if self.release_dynamic_tensors is not None:
            args.append(f"release_dynamic_tensors={'true' if self.release_dynamic_tensors else 'false'}")

        return f"--tflite_options {';'.join(args)}"


@dataclass
class TfliteQnnOptions(TfliteOptions):
    qnn_log_level: Optional[QnnLogLevel] = QnnLogLevel.K_LOG_LEVEL_WARN
    qnn_graph_priority: Optional[QnnGraphPriority] = QnnGraphPriority.K_QNN_PRIORITY_DEFAULT
    qnn_gpu_precision: Optional[QnnGpuPrecision] = QnnGpuPrecision.K_GPU_FP16
    qnn_gpu_performance_mode: Optional[QnnGpuPerformanceMode] = QnnGpuPerformanceMode.K_GPU_HIGH
    qnn_dsp_performance_mode: Optional[QnnDspPerformanceMode] = QnnDspPerformanceMode.K_DSP_BURST
    qnn_dsp_encoding: Optional[QnnDspEncoding] = QnnDspEncoding.K_DSP_STATIC
    qnn_htp_performance_mode: Optional[TfliteQnnHtpPerformanceMode] = TfliteQnnHtpPerformanceMode.K_HTP_BURST
    qnn_htp_precision: Optional[QnnHtpPrecision] = QnnHtpPrecision.K_HTP_FP16
    qnn_htp_optimization_strategy: Optional[
        QnnHtpOptimizationStrategy
    ] = QnnHtpOptimizationStrategy.K_HTP_OPTIMIZE_FOR_INFERENCE
    qnn_htp_use_conv_hmx: Optional[bool] = True
    qnn_htp_use_fold_relu: Optional[bool] = False
    qnn_htp_vtcm_size: Optional[int] = None
    qnn_htp_num_hvx_threads: Optional[int] = None

    def to_cli_string(self) -> str:
        base_string = super().to_cli_string().split(" ")[1]  # Get base TfliteOptions part
        args = [base_string]
        if self.qnn_log_level is not None:
            args.append(f"qnn_log_level={self.qnn_log_level.value}")
        if self.qnn_graph_priority is not None:
            args.append(f"qnn_graph_priority={self.qnn_graph_priority.value}")
        if self.qnn_gpu_precision is not None:
            args.append(f"qnn_gpu_precision={self.qnn_gpu_precision.value}")
        if self.qnn_gpu_performance_mode is not None:
            args.append(f"qnn_gpu_performance_mode={self.qnn_gpu_performance_mode.value}")
        if self.qnn_dsp_performance_mode is not None:
            args.append(f"qnn_dsp_performance_mode={self.qnn_dsp_performance_mode.value}")
        if self.qnn_dsp_encoding is not None:
            args.append(f"qnn_dsp_encoding={self.qnn_dsp_encoding.value}")
        if self.qnn_htp_performance_mode is not None:
            args.append(f"qnn_htp_performance_mode={self.qnn_htp_performance_mode.value}")
        if self.qnn_htp_precision is not None:
            args.append(f"qnn_htp_precision={self.qnn_htp_precision.value}")
        if self.qnn_htp_optimization_strategy is not None:
            args.append(f"qnn_htp_optimization_strategy={self.qnn_htp_optimization_strategy.value}")
        if self.qnn_htp_use_conv_hmx is not None:
            args.append(f"qnn_htp_use_conv_hmx={'true' if self.qnn_htp_use_conv_hmx else 'false'}")
        if self.qnn_htp_use_fold_relu is not None:
            args.append(f"qnn_htp_use_fold_relu={'true' if self.qnn_htp_use_fold_relu else 'false'}")
        if self.qnn_htp_vtcm_size is not None:
            args.append(f"qnn_htp_vtcm_size={self.qnn_htp_vtcm_size}")
        if self.qnn_htp_num_hvx_threads is not None:
            args.append(f"qnn_htp_num_hvx_threads={self.qnn_htp_num_hvx_threads}")

        return f"--tflite_options {';'.join(args)}"


@dataclass
class TfliteGpuv2Options(TfliteOptions):
    gpu_inference_preference: Optional[
        GpuInferencePreference
    ] = GpuInferencePreference.TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED
    gpu_inference_priority1: Optional[
        GpuInferencePriority
    ] = GpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
    gpu_inference_priority2: Optional[
        GpuInferencePriority
    ] = GpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE
    gpu_inference_priority3: Optional[
        GpuInferencePriority
    ] = GpuInferencePriority.TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION
    gpu_max_delegated_partitions: Optional[int] = 1

    def to_cli_string(self) -> str:
        base_string = super().to_cli_string().split(" ")[1]  # Get base TfliteOptions part
        args = [base_string]
        if self.gpu_inference_preference is not None:
            args.append(f"gpu_inference_preference={self.gpu_inference_preference.value}")
        if self.gpu_inference_priority1 is not None:
            args.append(f"gpu_inference_priority1={self.gpu_inference_priority1.value}")
        if self.gpu_inference_priority2 is not None:
            args.append(f"gpu_inference_priority2={self.gpu_inference_priority2.value}")
        if self.gpu_inference_priority3 is not None:
            args.append(f"gpu_inference_priority3={self.gpu_inference_priority3.value}")
        if self.gpu_max_delegated_partitions is not None:
            args.append(f"gpu_max_delegated_partitions={self.gpu_max_delegated_partitions}")

        return f"--tflite_options {';'.join(args)}"


@dataclass
class TfliteNnapiOptions(TfliteOptions):
    nnapi_execution_preference: Optional[NnapiExecutionPreference] = NnapiExecutionPreference.K_SUSTAINED_SPEED
    nnapi_max_number_delegated_partitions: Optional[int] = 3
    nnapi_allow_fp16: Optional[bool] = True

    def to_cli_string(self) -> str:
        base_string = super().to_cli_string().split(" ")[1]  # Get base TfliteOptions part
        args = [base_string]
        if self.nnapi_execution_preference is not None:
            args.append(f"nnapi_execution_preference={self.nnapi_execution_preference.value}")
        if self.nnapi_max_number_delegated_partitions is not None:
            args.append(f"nnapi_max_number_delegated_partitions={self.nnapi_max_number_delegated_partitions}")
        if self.nnapi_allow_fp16 is not None:
            args.append(f"nnapi_allow_fp16={'true' if self.nnapi_allow_fp16 else 'false'}")

        return f"--tflite_options {';'.join(args)}"


@dataclass
class QnnOptions:
    default_graph_htp_optimization_value: Optional[int] = True
    context_async_execution_queue_depth_numeric: Optional[int] = None
    context_enable_graphs: Optional[List[str]] = None
    context_error_reporting_options_level: Optional[ContextErrorReportingOptionsLevel] = None
    context_error_reporting_options_storage_limit: Optional[int] = None
    context_memory_limit_hint: Optional[int] = None
    context_priority: Optional[Priority] = None
    context_gpu_performance_hint: Optional[ContextGpuPerformanceHint] = ContextGpuPerformanceHint.HIGH
    context_gpu_use_gl_buffers: Optional[bool] = None
    context_htp_performance_mode: Optional[ContextHtpPerformanceMode] = ContextHtpPerformanceMode.BURST
    default_graph_priority: Optional[Priority] = True
    default_graph_gpu_precision: Optional[DefaultGraphGpuPrecision] = DefaultGraphGpuPrecision.USER_PROVIDED
    default_graph_gpu_disable_memory_optimizations: Optional[bool] = None
    default_graph_gpu_disable_node_optimizations: Optional[bool] = None
    default_graph_gpu_disable_queue_recording: Optional[bool] = None
    default_graph_htp_disable_fold_relu_activation_into_conv: Optional[bool] = False
    default_graph_htp_num_hvx_threads: Optional[int] = 4
    default_graph_htp_optimization_type: Optional[
        DefaultGraphHtpOptimizationType
    ] = DefaultGraphHtpOptimizationType.FINALIZE_OPTIMIZATION_FLAG
    default_graph_htp_optimization_value: Optional[int] = field(default=None, metadata={"valid_values": [1, 2, 3]})
    default_graph_htp_precision: Optional[DefaultGraphHtpPrecision] = DefaultGraphHtpPrecision.FLOAT16
    default_graph_htp_disable_short_depth_conv_on_hmx: Optional[bool] = False
    default_graph_htp_vtcm_size: Optional[int] = 4

    def __post_init__(self):
        valid_values = self.__dataclass_fields__["default_graph_htp_optimization_value"].metadata["valid_values"]
        if (
            self.default_graph_htp_optimization_value is not None
            and self.default_graph_htp_optimization_value not in valid_values
        ):
            raise ValueError(
                f"default_graph_htp_optimization_value must be one of {valid_values}, "
                f"got {self.default_graph_htp_optimization_value}"
            )

    def to_cli_string(self) -> str:
        args = []
        if self.default_graph_htp_optimization_value is not None:
            args.append(f"default_graph_htp_optimization_value={self.default_graph_htp_optimization_value}")
        if self.context_async_execution_queue_depth_numeric is not None:
            args.append(
                f"context_async_execution_queue_depth_numeric={self.context_async_execution_queue_depth_numeric}"
            )
        if self.context_enable_graphs is not None:
            args.append(f"context_enable_graphs={','.join(self.context_enable_graphs)}")
        if self.context_error_reporting_options_level is not None:
            args.append(f"context_error_reporting_options_level={self.context_error_reporting_options_level}")
        if self.context_error_reporting_options_storage_limit is not None:
            args.append(
                f"context_error_reporting_options_storage_limit={self.context_error_reporting_options_storage_limit}"
            )
        if self.context_memory_limit_hint is not None:
            args.append(f"context_memory_limit_hint={self.context_memory_limit_hint}")
        if self.context_priority is not None:
            args.append(f"context_priority={self.context_priority}")
        if self.context_gpu_performance_hint is not None:
            args.append(f"context_gpu_performance_hint={self.context_gpu_performance_hint}")
        if self.context_gpu_use_gl_buffers is not None:
            args.append(f"context_gpu_use_gl_buffers={'true' if self.context_gpu_use_gl_buffers else 'false'}")
        if self.context_htp_performance_mode is not None:
            args.append(f"context_htp_performance_mode={self.context_htp_performance_mode}")
        if self.default_graph_priority is not None:
            args.append(f"default_graph_priority={self.default_graph_priority}")
        if self.default_graph_gpu_precision is not None:
            args.append(f"default_graph_gpu_precision={self.default_graph_gpu_precision}")
        if self.default_graph_gpu_disable_memory_optimizations is not None:
            args.append(
                f"default_graph_gpu_disable_memory_optimizations={'true' if self.default_graph_gpu_disable_memory_optimizations else 'false'}"
            )
        if self.default_graph_gpu_disable_node_optimizations is not None:
            args.append(
                f"default_graph_gpu_disable_node_optimizations={'true' if self.default_graph_gpu_disable_node_optimizations else 'false'}"
            )
        if self.default_graph_gpu_disable_queue_recording is not None:
            args.append(
                f"default_graph_gpu_disable_queue_recording={'true' if self.default_graph_gpu_disable_queue_recording else 'false'}"
            )
        if self.default_graph_htp_disable_fold_relu_activation_into_conv is not None:
            args.append(
                f"default_graph_htp_disable_fold_relu_activation_into_conv={'true' if self.default_graph_htp_disable_fold_relu_activation_into_conv else 'false'}"
            )
        if self.default_graph_htp_num_hvx_threads is not None:
            args.append(f"default_graph_htp_num_hvx_threads={self.default_graph_htp_num_hvx_threads}")
        if self.default_graph_htp_optimization_type is not None:
            args.append(f"default_graph_htp_optimization_type={self.default_graph_htp_optimization_type}")
        if self.default_graph_htp_precision is not None:
            args.append(f"default_graph_htp_precision={self.default_graph_htp_precision}")
        if self.default_graph_htp_disable_short_depth_conv_on_hmx is not None:
            args.append(
                f"default_graph_htp_disable_short_depth_conv_on_hmx={'true' if self.default_graph_htp_disable_short_depth_conv_on_hmx else 'false'}"
            )
        if self.default_graph_htp_vtcm_size is not None:
            args.append(f"default_graph_htp_vtcm_size={self.default_graph_htp_vtcm_size}")

        return f"--qnn_options {';'.join(args)}"


@dataclass
class ProfileCommonOptions(CommonOptions):
    dequantize_outputs: Optional[bool] = True
    tflite_delegates: Optional[List[TfliteDelegates]] = None
    tflite_options: Optional[Union[TfliteOptions, TfliteQnnOptions, TfliteGpuv2Options, TfliteNnapiOptions]] = None
    qnn_options: Optional[QnnOptions] = None
    onnx_options: Optional[Union[OnnxOptions, OnnxQnnOptions]] = None
    onnx_execution_providers: Optional[List[OnnxExecutionProviders]] = None
    max_profiler_iterations: Optional[int] = 100
    max_profiler_time: Optional[int] = 600

    def handle_tflite_options(self) -> str:
        if isinstance(self.tflite_options, (TfliteOptions, TfliteQnnOptions, TfliteGpuv2Options, TfliteNnapiOptions)):
            return self.tflite_options.to_cli_string()
        else:
            return str(self.tflite_options)

    def handle_onnx_options(self) -> str:
        if isinstance(self.onnx_options, (OnnxOptions, OnnxQnnOptions)):
            return self.onnx_options.to_cli_string()
        else:
            return str(self.onnx_options)

    def handle_qnn_options(self) -> str:
        if isinstance(self.qnn_options, QnnOptions):
            return self.qnn_options.to_cli_string()
        else:
            return str(self.qnn_options)

    def handle_common_options(self) -> List[str]:
        args = []
        if self.compute_unit is not None:
            compute_units = ",".join(list(self.compute_unit))
            args.append(f"--compute_unit {compute_units}")
        if self.dequantize_outputs:
            args.append("--dequantize_outputs")
        if self.tflite_delegates is not None:
            tflite_delegates = ",".join(list(self.tflite_delegates))
            args.append(f"--tflite_delegates {tflite_delegates}")
        if self.tflite_options is not None:
            args.append(self.handle_tflite_options())
        if self.onnx_options is not None:
            args.append(self.handle_onnx_options())
        if self.qnn_options is not None:
            args.append(self.handle_qnn_options())
        if self.onnx_execution_providers is not None:
            onnx_execution_providers = ",".join((self.onnx_execution_providers))
            args.append(f"--onnx_execution_providers {onnx_execution_providers}")
        if self.max_profiler_iterations is not None:
            args.append(f"--max_profiler_iterations {self.max_profiler_iterations}")
        if self.max_profiler_time is not None:
            args.append(f"--max_profiler_time {self.max_profiler_time}")
        return args

    def to_cli_string(self) -> str:
        args = self.handle_common_options()
        return " ".join(args)


@dataclass
class ProfileOptions(ProfileCommonOptions):
    """
    Profile options for the model.

    Note:
        For details, see [ProfileOptions in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options).
    """

    pass


@dataclass
class InferenceOptions(ProfileCommonOptions):
    """
    Inference options for the model.

    Note:
        For details, see [InferenceOptions in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/api.html#profile-inference-options).
    """

    pass
