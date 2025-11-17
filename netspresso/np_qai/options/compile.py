from dataclasses import dataclass
from typing import Optional

from netspresso.enums.base import StrEnum
from netspresso.np_qai.options.common import CommonOptions


class Framework(StrEnum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    ONNXRUNTIME = "onnxruntime"
    AIMET = "aimet"
    TENSORFLOW = "tensorflow"
    TFLITE = "tensorflow_lite"
    COREML = "coreml"
    TENSORRT = "tensorrt"
    QNN = "qnn"


class Extension(StrEnum):
    ONNX = ".onnx"
    PT = ".pt"
    AIMET = ".aimet"
    H5 = ".h5"


class Runtime(StrEnum):
    TFLITE = "tflite"
    QNN_LIB_AARCH64_ANDROID = "qnn_lib_aarch64_android"  # Deprecated in qai-hub 0.40.0
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    QNN_DLC = "qnn_dlc"  # Added in qai-hub 0.40.0 - Recommended for QNN deployment
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"


class QuantizeFullType(StrEnum):
    INT8 = "int8"
    INT16 = "int16"
    W8A16 = "w8a16"
    W4A8 = "w4a8"
    W4A16 = "w4a16"


class QuantizeWeightType(StrEnum):
    FP16 = "float16"


@dataclass
class CompileOptions(CommonOptions):
    """
    Compile options for the model.

    Note:
        For details, see [CompileOptions in QAI Hub API](https://app.aihub.qualcomm.com/docs/hub/api.html#compile-options).
    """

    target_runtime: Optional[Runtime] = Runtime.TFLITE
    output_names: Optional[str] = None
    truncate_64bit_tensors: Optional[bool] = False
    truncate_64bit_io: Optional[bool] = False
    force_channel_last_input: Optional[str] = None
    force_channel_last_output: Optional[str] = None
    quantize_full_type: Optional[QuantizeFullType] = None
    quantize_weight_type: Optional[QuantizeWeightType] = None
    quantize_io: Optional[bool] = False
    quantize_io_type: Optional[str] = None
    qnn_graph_name: Optional[str] = None
    qnn_context_binary_vtcm: Optional[str] = None
    qnn_context_binary_optimization_level: Optional[int] = None

    def to_cli_string(self) -> str:
        args = []
        if self.compute_unit is not None:
            compute_units = ",".join([unit.name.lower() for unit in self.compute_unit])
            args.append(f"--compute_unit {compute_units}")
        if self.target_runtime is not None:
            args.append(f"--target_runtime {self.target_runtime}")
        if self.output_names is not None:
            output_names_str = ",".join(self.output_names.split())  # Split and join to handle spaces
            args.append(f'--output_names "{output_names_str}"')
        if self.truncate_64bit_tensors:
            args.append("--truncate_64bit_tensors")
        if self.truncate_64bit_io:
            args.append("--truncate_64bit_io")
        if self.force_channel_last_input is not None:
            args.append(f'--force_channel_last_input "{self.force_channel_last_input}"')
        if self.force_channel_last_output is not None:
            args.append(f'--force_channel_last_output "{self.force_channel_last_output}"')
        if self.quantize_full_type is not None:
            args.append(f"--quantize_full_type {self.quantize_full_type}")
        if self.quantize_weight_type is not None:
            args.append(f"--quantize_weight_type {self.quantize_weight_type}")
        if self.quantize_io:
            args.append("--quantize_io")
        if self.quantize_io_type is not None:
            args.append(f"--quantize_io_type {self.quantize_io_type}")
        if self.qnn_graph_name is not None:
            args.append(f"--qnn_graph_name {self.qnn_graph_name}")
        if self.qnn_context_binary_vtcm is not None:
            args.append(f"--qnn_context_binary_vtcm {self.qnn_context_binary_vtcm}")
        if self.qnn_context_binary_optimization_level is not None:
            args.append(f"--qnn_context_binary_optimization_level {self.qnn_context_binary_optimization_level}")

        return " ".join(args)
