from netspresso import NetsPresso
from netspresso.enums import DeviceName, Framework, DataType

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
converter = netspresso.converter_v2()

# 2. Set variables for convert
INPUT_MODEL_PATH = "./examples/sample_models/yolo-fastest.onnx"
OUTPUT_DIR = "./outputs/converted/ARDUINO_NICLA_VISION"
TARGET_FRAMEWORK = Framework.TENSORFLOW_LITE
TARGET_DEVICE_NAME = DeviceName.ARDUINO_NICLA_VISION
TARGET_DATA_TYPE = DataType.INT8

# 3. Run convert
conversion_task = converter.convert_model(
    input_model_path=INPUT_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    target_framework=TARGET_FRAMEWORK,
    target_device_name=TARGET_DEVICE_NAME,
    target_data_type=TARGET_DATA_TYPE,
)
print(conversion_task)

# 4. Declare profiler
profiler = netspresso.profiler()

# 5. Run profile
profile_task = profiler.profile_model(
    input_model_path=conversion_task.converted_model_path,
    target_device_name=TARGET_DEVICE_NAME,

)
print(f"model inference latency: {profile_task.profile_result.latency} ms")
