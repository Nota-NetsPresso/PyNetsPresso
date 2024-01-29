from netspresso import NetsPresso
from netspresso.enums import Framework, DeviceName


###
# Available target frameworks for conversion with keras models
#
# Framework.TENSORFLOW_LITE
#

###
# Available devices for Framework.TENSORFLOW_LITE (target_framework)
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
INPUT_MODEL_PATH = "./examples/sample_models/mobilenetv1.h5"
OUTPUT_DIR = "./outputs/converted/TFLITE_RASPBERRY_PI_4B"
TARGET_FRAMEWORK = Framework.TENSORFLOW_LITE
TARGET_DEVICE_NAME = DeviceName.RASPBERRY_PI_4B

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
CONVERTED_MODEL_PATH = "./outputs/converted/TFLITE_RASPBERRY_PI_4B/TFLITE_RASPBERRY_PI_4B.tflite"

# 6. Run benchmark
benchmark_task = benchmarker.benchmark_model(
    input_model_path=CONVERTED_MODEL_PATH,
    target_device_name=TARGET_DEVICE_NAME,
)
print(f"model inference latency: {benchmark_task['result']['latency']} ms")
