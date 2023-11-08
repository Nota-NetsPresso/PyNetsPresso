from loguru import logger
from netspresso.client import SessionClient
from netspresso.launcher import (
    ModelConverter,
    ModelBenchmarker,
    ModelFramework,
    DeviceName,
    BenchmarkTask,
    ConversionTask,
    Model,
    DataType,
    HardwareType,
)


if __name__ == "__main__":
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    CONVERTED_MODEL_PATH = "converted_model.tflite"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = ModelConverter(user_session=session)
    model: Model = converter.upload_model("./examples/sample_models/mobilenetv1.h5")

    ###
    # Available Target Frameworks for Conversion with Keras Models
    #
    # ModelFramework.TENSORFLOW_LITE
    #

    ###
    # Available Devices for ModelFramework.TENSORFLOW_LITE and DataType.INT8 (target_framework)
    # DeviceName.ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    TARGET_DEVICE_NAME = DeviceName.RENESAS_RA8D1
    DATA_TYPE = DataType.INT8

    conversion_task: ConversionTask = converter.convert_model(
        model=model,
        input_shape=model.input_shape,
        target_framework=ModelFramework.TENSORFLOW_LITE,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        wait_until_done=True,
    )

    logger.info(conversion_task)
    converter.download_converted_model(conversion_task, dst=CONVERTED_MODEL_PATH)

    ###
    # Benchmark available Devices with TFLite(int8) models(ModelFramework.TENSORFLOW_LITE)
    # DeviceName.ENSEMBLE_E7_DEVKIT_GEN2
    # DeviceName.RENESAS_RA8D1
    #

    benchmarker = ModelBenchmarker(user_session=session)
    benchmark_model: Model = benchmarker.upload_model(CONVERTED_MODEL_PATH)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(
        model=benchmark_model,
        target_device_name=TARGET_DEVICE_NAME,
        data_type=DATA_TYPE,
        HARDWARE_TYPE=HardwareType.HELIUM,
        wait_until_done=True
    )

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
