Benchmark Model
===============

.. autofunction:: netspresso.benchmarker.__init__.Benchmarker.benchmark_model


Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import DeviceName, SoftwareVersion


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    benchmarker = netspresso.benchmarker()
    benchmark_task = benchmarker.benchmark_model(
        input_model_path="./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1.trt",
        target_device_name=DeviceName.JETSON_AGX_ORIN,
        target_software_version=SoftwareVersion.JETPACK_5_0_1,
    )
