from loguru import logger
from netspresso.client import SessionClient
from netspresso.launcher import (
    ModelConverter,
    ModelBenchmarker,
    ModelFramework,
    DeviceName,
    BenchmarkTask,
    ConversionTask,
    DataType,
    HardwareType,
)


if __name__ == "__main__":
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    MODEL_PATH = "./examples/sample_models/yolox_nano.onnx"
    CONVERTED_MODEL_PATH = "./outputs/converted/converted_model.tflite"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = ModelConverter(user_session=session)

    ###
    # Available Target Frameworks for Conversion with ONNX Models
    #
    # ModelFramework.TENSORFLOW_LITE
    #

    ###
    # Available Devices for ModelFramework.TENSORFLOW_LITE and DataType.INT8 (target_framework)
    # DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    TARGET_DEVICE_NAME = DeviceName.RENESAS_RA8D1
    DATA_TYPE = DataType.INT8

    conversion_task: ConversionTask = converter.convert_model(
        model_path=MODEL_PATH,
        target_framework=ModelFramework.TENSORFLOW_LITE,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        output_path=CONVERTED_MODEL_PATH,
    )

    logger.info(conversion_task)

    ###
    # Benchmark available Devices with TFLite(int8) models(ModelFramework.TENSORFLOW_LITE)
    # DeviceName.ALIF_ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    benchmarker = ModelBenchmarker(user_session=session)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(
        model_path=CONVERTED_MODEL_PATH,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        hardware_type=HardwareType.HELIUM,
    )

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
