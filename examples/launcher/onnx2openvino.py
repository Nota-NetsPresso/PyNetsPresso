from loguru import logger
from netspresso.clients.auth import SessionClient
from netspresso.launcher import (
    Converter,
    Benchmarker,
    ModelFramework,
    DeviceName,
)


if __name__ == "__main__":
    EMAIL = "YOUR_EMAIL"
    PASSWORD = "YOUR_PASSWORD"
    MODEL_PATH = "./examples/sample_models/test.onnx"
    CONVERTED_MODEL_PATH = "./outputs/converted/onnx2openvino"
    session = SessionClient(email=EMAIL, password=PASSWORD)
    converter = Converter(user_session=session)

    ###
    # Available Target Frameworks for Conversion with ONNX Models
    #
    # ModelFramework.TENSORRT <-- For NVIDIA Devices
    # ModelFramework.OPENVINO <-- For Intel CPUs
    # ModelFramework.TENSORFLOW_LITE <-- For the devices like Raspberry Pi devices
    # ModelFramework.DRPAI <-- For Renesas Devices like RZ/V2M, RZ/V2L
    #

    ###
    # Available Devices for ModelFramework.OPENVINO (target_framework)
    #
    # DeviceName.Intel_XEON_W_2233
    #

    conversion_task = converter.convert_model(
        model_path=MODEL_PATH,
        target_framework=ModelFramework.OPENVINO,
        target_device_name=DeviceName.Intel_XEON_W_2233,
        output_path=CONVERTED_MODEL_PATH,
    )
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # conversion_task = converter.convert_model(
    #     model_path=MODEL_PATH,
    #     target_framework=ModelFramework.OPENVINO,
    #     target_device_name=DeviceName.Intel_XEON_W_2233,
    #     output_path=CONVERTED_MODEL_PATH,
    #     wait_until_done=False,
    # )

    # while conversion_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     conversion_task = converter.get_conversion_task(conversion_task)
    #     logger.info(f"conversion task status : {conversion_task.status}")
    #     time.sleep(1)

    # conversion_task = converter.get_conversion_task(conversion_task)
    logger.info(conversion_task)

    benchmarker = Benchmarker(user_session=session)
    benchmark_task = benchmarker.benchmark_model(
        model_path=CONVERTED_MODEL_PATH,
        target_framework=ModelFramework.OPENVINO,
        target_device_name=DeviceName.Intel_XEON_W_2233,
    )
    ########################
    # Asynchronous Procedure
    # If you wish to request conversion and retrieve the results later, please refer to the following code.
    ########################
    # benchmark_task = benchmarker.benchmark_model(
    #     model_path=CONVERTED_MODEL_PATH,
    #     target_framework=ModelFramework.OPENVINO,
    #     target_device_name=DeviceName.Intel_XEON_W_2233,
    #     wait_until_done=False,
    # )

    # while benchmark_task.status in [TaskStatus.IN_QUEUE, TaskStatus.IN_PROGRESS]:
    #     benchmark_task = benchmarker.get_benchmark_task(benchmark_task=benchmark_task)
    #     logger.info(f"benchmark task status : {benchmark_task.status}")
    #     time.sleep(1)

    logger.info(f"model inference latency: {benchmark_task.latency} ms")
