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
# Available devices for Framework.TENSORFLOW_LITE (target_framework)
# DeviceName.RASPBERRY_PI_4B
# DeviceName.RASPBERRY_PI_3B_PLUS
# DeviceName.RASPBERRY_PI_ZERO_W
# DeviceName.RASPBERRY_PI_ZERO_2W
#

###
# Available devices with TFLite models(Framework.TENSORFLOW_LITE)
#
# DeviceName.RASPBERRY_PI_4B
# DeviceName.RASPBERRY_PI_3B_PLUS
# DeviceName.RASPBERRY_PI_ZERO_W
# DeviceName.RASPBERRY_PI_ZERO_2W
#

EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

# 1. Declare converter
converter = netspresso.converter()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/test.onnx"
OUTPUT_DIR = "./outputs/converted/TFLITE_RASPBERRY_PI_3B_PLUS"
TARGET_FRAMEWORK = Framework.TENSORFLOW_LITE
TARGET_DEVICE_NAME = DeviceName.RASPBERRY_PI_3B_PLUS

# 3. Run convert
conversion_result = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=TARGET_FRAMEWORK,
    target_device_name=TARGET_DEVICE_NAME,
)
print(conversion_result)

# 4. Declare benchmarker
benchmarker = netspresso.benchmarker()

# 5. Set variables for benchmark
CONVERTED_MODEL_PATH = conversion_result["converted_model_path"]

# 6. Run benchmark
benchmark_result = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=TARGET_DEVICE_NAME,
)
print(f"model inference latency: {benchmark_result['result']['latency']} ms")
