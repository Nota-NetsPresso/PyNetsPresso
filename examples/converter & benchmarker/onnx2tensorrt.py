from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework, SoftwareVersion

###
# Available target frameworks for conversion with onnx models
#
# Framework.TENSORRT <-- For NVIDIA Devices
# Framework.OPENVINO <-- For Intel CPUs
# Framework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
# Framework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
#

###
# Available devices for Framework.TENSORRT (target_framework)
#
# DeviceName.JETSON_NANO
# DeviceName.JETSON_TX2
# DeviceName.JETSON_XAVIER
# DeviceName.JETSON_NX
# DeviceName.JETSON_AGX_ORIN
# DeviceName.AWS_T4
#

###
# Available software versions for jetson devices
#
# DeviceName.JETSON_NANO : SoftwareVersion.JETPACK_4_6, SoftwareVersion.JETPACK_4_4_1
# DeviceName.JETSON_TX2 : SoftwareVersion.JETPACK_4_6
# DeviceName.JETSON_XAVIER : SoftwareVersion.JETPACK_4_6
# DeviceName.JETSON_NX : SoftwareVersion.JETPACK_5_0_2, SoftwareVersion.JETPACK_4_6,
# DeviceName.JETSON_AGX_ORIN : SoftwareVersion.JETPACK_5_0_1
#

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare converter
converter = netspresso.converter()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1"
TARGET_FRAMEWORK = Framework.TENSORRT
TARGET_DEVICE_NAME = DeviceName.JETSON_AGX_ORIN
TARGET_SOFTWARE_VERSION = SoftwareVersion.JETPACK_5_0_1

# 3. Run convert
conversion_result = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=TARGET_FRAMEWORK,
    target_device_name=TARGET_DEVICE_NAME,
    target_software_version=TARGET_SOFTWARE_VERSION,
)
print(conversion_result)

###
# !!WARNING!!
#
# For NVIDIA GPUs and Jetson devices,
# DeviceName and Software Version have to be matched with the target_device of the conversion.
# TensorRT Model has strong dependency with the device type and its jetpack version.

# 4. Declare benchmarker
benchmarker = netspresso.benchmarker()

# 5. Set variables for benchmark
CONVERTED_MODEL_PATH = conversion_result["converted_model_path"]

# 6. Run benchmark
benchmark_result = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=DeviceName.JETSON_AGX_ORIN,
    target_software_version=SoftwareVersion.JETPACK_5_0_1,
)
print(f"model inference latency: {benchmark_result['result']['latency']} ms")
print(f"model gpu memory footprint: {benchmark_result['result']['memory_footprint_gpu']} MB")
print(f"model cpu memory footprint: {benchmark_result['result']['memory_footprint_cpu']} MB")
