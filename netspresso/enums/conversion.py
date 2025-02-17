from enum import Enum


class SourceFramework(str, Enum):
    ONNX = "onnx"


class SourceFrameworkDisplay(str, Enum):
    ONNX = "ONNX"


SOURCE_FRAMEWORK_DISPLAY_MAP = {
    SourceFramework.ONNX: SourceFrameworkDisplay.ONNX,
}


class TargetFramework(str, Enum):
    TENSORRT = "tensorrt"
    TENSORFLOW_LITE = "tensorflow_lite"
    OPENVINO = "openvino"
    DRPAI = "drpai"


class TargetFrameworkDisplay(str, Enum):
    TENSORRT = "TensorRT"
    TENSORFLOW_LITE = "TensorFlow Lite"
    OPENVINO = "OpenVINO"
    DRPAI = "DRPAI"


TARGET_FRAMEWORK_DISPLAY_MAP = {
    TargetFramework.TENSORRT: TargetFrameworkDisplay.TENSORRT,
    TargetFramework.TENSORFLOW_LITE: TargetFrameworkDisplay.TENSORFLOW_LITE,
    TargetFramework.OPENVINO: TargetFrameworkDisplay.OPENVINO,
    TargetFramework.DRPAI: TargetFrameworkDisplay.DRPAI,
}


class PrecisionForConversion(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"


class PrecisionForConversionDisplay(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"


class PrecisionForBenchmark(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


class PrecisionForBenchmarkDisplay(str, Enum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


PRECISION_FOR_CONVERSION_DISPLAY_MAP = {
    PrecisionForConversion.FP16: PrecisionForConversionDisplay.FP16,
    PrecisionForConversion.INT8: PrecisionForConversionDisplay.INT8,
}

PRECISION_FOR_BENCHMARK_DISPLAY_MAP = {
    PrecisionForBenchmark.FP32: PrecisionForBenchmarkDisplay.FP32,
    PrecisionForBenchmark.FP16: PrecisionForBenchmarkDisplay.FP16,
    PrecisionForBenchmark.INT8: PrecisionForBenchmarkDisplay.INT8,
}
