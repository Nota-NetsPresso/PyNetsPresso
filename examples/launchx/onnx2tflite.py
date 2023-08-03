import time
from loguru import logger
from netspresso.client import SessionClient
from netspresso.launchx import ModelConverter, ModelBenchmarker
from netspresso.launchx.utils.devices import filter_devices_with_device_name
from netspresso.launchx.schemas import ModelFramework, TaskStatus, DeviceName
from netspresso.launchx.schemas.model import BenchmarkTask, ConversionTask, Model, TargetDevice


if __name__ == '__main__':
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    CONVERTED_MODEL_PATH = "converted_model.tflite"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = ModelConverter(session)
    model: Model = converter.upload_model("./test.onnx")

    available_devices: list[TargetDevice] = filter_devices_with_device_name(name=DeviceName.RASPBERRY_PI_4B,
                                                                            devices=model.available_devices)
    
    conversion_task: ConversionTask = converter.convert_model(model=model,
                                                              input_shape=model.input_shape,
                                                              target_framework=ModelFramework.TENSORFLOW_LITE,
                                                              target_device=available_devices[0],
                                                              wait_until_done=True)
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # conversion_task: ConversionTask = converter.convert_model(model=model,
    #                                                           input_shape=model.input_shape,
    #                                                           target_framework=ModelFramework.TENSORFLOW_LITE,
    #                                                           target_device=available_devices[0]
    #                                                           wait_until_done=False)

    # while conversion_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     conversion_task = converter.get_conversion_task(conversion_task)
    #     logger.info(f"conversion task status : {conversion_task.status}")
    #     time.sleep(1)

    # conversion_task = converter.get_conversion_task(conversion_task)
    logger.info(conversion_task)
    converter.download_converted_model(conversion_task, dst=CONVERTED_MODEL_PATH)

    
    benchmarker = ModelBenchmarker(session)
    benchmark_model: Model = benchmarker.upload_model(CONVERTED_MODEL_PATH)
    benchmark_available_devices: list[TargetDevice] = filter_devices_with_device_name(name=DeviceName.RASPBERRY_PI_4B,
                                                                                      devices=benchmark_model.available_devices)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(model=benchmark_model,
                                                                target_device=benchmark_available_devices[0],
                                                                wait_until_done=True)
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # benchmark_task: BenchmarkTask = benchmarker.benchmark_model(model=benchmark_model,
    #                                                             target_device=benchmark_available_devices[0],
    #                                                             wait_until_done=False)

    # while benchmark_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     benchmark_task = benchmarker.get_benchmark_task(benchmark_task=benchmark_task)
    #     logger.info(f"benchmark task status : {benchmark_task.status}")
    #     time.sleep(1)

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
