from loguru import logger
from netspresso.clients.auth import SessionClient
from netspresso.launcher import (
    Converter,
    Benchmarker,
    ModelFramework,
    DeviceName,
    BenchmarkTask,
    ConversionTask,
)


if __name__ == "__main__":
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    MODEL_PATH = "./examples/sample_models/mobilenetv1.h5"
    CONVERTED_MODEL_PATH = "./outputs/converted/keras2tflite"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = Converter(user_session=session)

    ###
    # Available Target Frameworks for Conversion with Keras Models
    #
    # ModelFramework.TENSORFLOW_LITE
    #

    ###
    # Available Devices for ModelFramework.TENSORFLOW_LITE (target_framework)
    # DeviceName.RASPBERRY_PI_4B
    # DeviceName.RASPBERRY_PI_3B_PLUS
    # DeviceName.RASPBERRY_PI_ZERO_W
    # DeviceName.RASPBERRY_PI_ZERO_2W
    #

    conversion_task: ConversionTask = converter.convert_model(
        model_path=MODEL_PATH,
        target_framework=ModelFramework.TENSORFLOW_LITE,
        target_device_name=DeviceName.RASPBERRY_PI_4B,
        output_path=CONVERTED_MODEL_PATH,
    )
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # conversion_task: ConversionTask = converter.convert_model(
    #     model_path=MODEL_PATH,
    #     target_framework=ModelFramework.TENSORFLOW_LITE,
    #     target_device_name=DeviceName.RASPBERRY_PI_4B,
    #     output_path=CONVERTED_MODEL_PATH,
    #     wait_until_done=False,
    # )

    # while conversion_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     conversion_task = converter.get_conversion_task(conversion_task)
    #     logger.info(f"conversion task status : {conversion_task.status}")
    #     time.sleep(1)

    # conversion_task = converter.get_conversion_task(conversion_task)
    logger.info(conversion_task)

    ###
    # Benchmark available Devices with TFLite models(ModelFramework.TENSORFLOW_LITE)
    # DeviceName.RASPBERRY_PI_4B
    # DeviceName.RASPBERRY_PI_3B_PLUS
    # DeviceName.RASPBERRY_PI_ZERO_W
    # DeviceName.RASPBERRY_PI_ZERO_2W
    #

    benchmarker = Benchmarker(user_session=session)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(
        model_path=CONVERTED_MODEL_PATH,
        target_framework=ModelFramework.TENSORFLOW_LITE,
        target_device_name=DeviceName.RASPBERRY_PI_4B,
    )
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # benchmark_task: BenchmarkTask = benchmarker.benchmark_model(
    #     model_path=CONVERTED_MODEL_PATH,
    #     target_framework=ModelFramework.TENSORFLOW_LITE,
    #     target_device_name=DeviceName.RASPBERRY_PI_4B,
    #     wait_until_done=False,
    # )

    # while benchmark_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     benchmark_task = benchmarker.get_benchmark_task(benchmark_task=benchmark_task)
    #     logger.info(f"benchmark task status : {benchmark_task.status}")
    #     time.sleep(1)

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
