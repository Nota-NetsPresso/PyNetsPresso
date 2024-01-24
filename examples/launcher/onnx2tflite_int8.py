from loguru import logger
from netspresso.clients.auth import SessionClient
from netspresso.launcher import (
    Converter,
    Benchmarker,
    ModelFramework,
    DeviceName,
    DataType,
    HardwareType,
)


if __name__ == "__main__":
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = Converter(user_session=session)

    ###
    # Available Target Frameworks for Conversion with ONNX Models
    #
    # ModelFramework.TENSORFLOW_LITE
    #

    ###
    # Available Devices for ModelFramework.TENSORFLOW_LITE and DataType.INT8 (target_framework)
    # DeviceName.RASPBERRY_PI_4B
    # DeviceName.RASPBERRY_PI_3B_PLUS
    # DeviceName.RASPBERRY_PI_ZERO_W
    # DeviceName.RASPBERRY_PI_ZERO_2W
    # DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    MODEL_PATH = "./examples/sample_models/pytorch_model_automatic_0.9.onnx"
    CONVERTED_MODEL_PATH = "./outputs/converted/onnx2tflite_int8_pytorch_model_automatic_0.9"
    TARGET_FRAMEWORK = ModelFramework.TENSORFLOW_LITE
    TARGET_DEVICE_NAME = DeviceName.RASPBERRY_PI_4B
    DATA_TYPE = DataType.INT8
    DATASET_PATH = "./examples/sample_datasets/20x64x64x3.npy"

    conversion_task = converter.convert_model(
        model_path=MODEL_PATH,
        target_framework=TARGET_FRAMEWORK,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        output_path=CONVERTED_MODEL_PATH,
        dataset_path=DATASET_PATH,
    )

    logger.info(conversion_task)

    ###
    # Benchmark available Devices with TFLite(int8) models(ModelFramework.TENSORFLOW_LITE)
    # DeviceName.RASPBERRY_PI_4B
    # DeviceName.RASPBERRY_PI_3B_PLUS
    # DeviceName.RASPBERRY_PI_ZERO_W
    # DeviceName.RASPBERRY_PI_ZERO_2W
    # DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    benchmarker = Benchmarker(user_session=session)
    benchmark_task = benchmarker.benchmark_model(
        model_path=CONVERTED_MODEL_PATH,
        target_framework=TARGET_FRAMEWORK,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
    )

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
