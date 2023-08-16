import time
from loguru import logger
from typing import List
from netspresso.client import SessionClient
from netspresso.launcher import ModelConverter, ModelBenchmarker
from netspresso.launcher.utils.devices import filter_devices_with_device_name
from netspresso.launcher.schemas import ModelFramework, TaskStatus, DeviceName
from netspresso.launcher.schemas.model import BenchmarkTask, ConversionTask, Model, TargetDevice


if __name__ == '__main__':
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    CONVERTED_MODEL_PATH = "converted_model.trt"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = ModelConverter(user_session=session)
    model: Model = converter.upload_model("./examples/sample_models/test.onnx")

    ###
    # Available Target Frameworks for Conversion with ONNX Models
    #
    # ModelFramework.TENSORRT <-- For NVIDIA Devices
    # ModelFramework.OPENVINO <-- For Intel CPUs
    # ModelFramework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
    # ModelFramework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
    #

    ###
    # Available Devices for ModelFramework.TENSORRT (target_framework)
    # DeviceName.JETSON_NANO
    # DeviceName.JETSON_TX2
    # DeviceName.JETSON_XAVIER
    # DeviceName.JETSON_NX
    # DeviceName.JETSON_AGX_ORIN
    # DeviceName.AWS_T4
    # DeviceName.JETSON_NANO
    #

    available_devices: List[TargetDevice] = filter_devices_with_device_name(name=DeviceName.JETSON_NANO,
                                                                            devices=model.available_devices)
    target_device = available_devices[0] # Jetson Nano - Jetpack 4.6
    conversion_task: ConversionTask = converter.convert_model(model=model,
                                                              input_shape=model.input_shape,
                                                              target_framework=ModelFramework.TENSORRT,
                                                              target_device=available_devices[0],
                                                              wait_until_done=True)
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # conversion_task: ConversionTask = converter.convert_model(model=model,
    #                                                           input_shape=model.input_shape,
    #                                                           target_framework=ModelFramework.TENSORRT,
    #                                                           target_device=available_devices[0],
    #                                                           wait_until_done=False)

    # while conversion_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     conversion_task = converter.get_conversion_task(conversion_task)
    #     logger.info(f"conversion task status : {conversion_task.status}")
    #     time.sleep(1)

    # conversion_task = converter.get_conversion_task(conversion_task)

    logger.info(conversion_task)
    converter.download_converted_model(conversion_task, dst=CONVERTED_MODEL_PATH)

    ###
    # !!WARNING!!
    # 
    # For NVIDIA GPUs and Jetson devices,
    # DeviceName and Software Version have to be matched with the target_device of the conversion.
    # TensorRT Model has strong dependency with the device type and its jetpack version.

    benchmarker = ModelBenchmarker(user_session=session)
    benchmark_model: Model = benchmarker.upload_model(CONVERTED_MODEL_PATH)
    benchmark_task: BenchmarkTask = benchmarker.benchmark_model(model=benchmark_model,
                                                                target_device=target_device,
                                                                wait_until_done=True)
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # benchmark_task: BenchmarkTask = benchmarker.benchmark_model(model=benchmark_model,
    #                                                             target_device=target_device,
    #                                                             wait_until_done=False)

    # while benchmark_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     benchmark_task = benchmarker.get_benchmark_task(benchmark_task=benchmark_task)
    #     logger.info(f"benchmark task status : {benchmark_task.status}")
    #     time.sleep(1)

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
    logger.info(f"model gpu memory footprint: {benchmark_task.memory_footprint_gpu} MB")
    logger.info(f"model cpu memory footprint: {benchmark_task.memory_footprint_cpu} MB")
