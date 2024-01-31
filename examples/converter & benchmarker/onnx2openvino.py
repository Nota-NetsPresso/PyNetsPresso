from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework

###
# Available target frameworks for conversion with onnx models
#
# Framework.TENSORRT <-- For NVIDIA Devices
# Framework.OPENVINO <-- For Intel CPUs
# Framework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
# Framework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
#

###
# Available devices for Framework.OPENVINO (target_framework)
#
# DeviceName.INTEL_XEON_W_2233
#

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare converter
converter = netspresso.converter()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/converted/OPENVINO_INTEL_XEON_W_2233"
TARGET_FRAMEWORK = Framework.OPENVINO
TARGET_DEVICE_NAME = DeviceName.INTEL_XEON_W_2233

# 3. Run convert
conversion_task = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=TARGET_FRAMEWORK,
    target_device_name=TARGET_DEVICE_NAME,
)
print(conversion_task)

# 4. Declare benchmarker
benchmarker = netspresso.benchmarker()

# 5. Set variables for benchmark
CONVERTED_MODEL_PATH = "./outputs/converted/OPENVINO_INTEL_XEON_W_2233/OPENVINO_INTEL_XEON_W_2233.zip"

# 6. Run benchmark
benchmark_task = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=TARGET_DEVICE_NAME,
)
print(f"model inference latency: {benchmark_task['result']['latency']} ms")
