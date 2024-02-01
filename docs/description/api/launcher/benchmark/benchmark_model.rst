Benchmark Model
===============

.. autofunction:: netspresso.launcher.__init__.ModelBenchmarker.benchmark_model


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelBenchmarker, DeviceName


    benchmarker = ModelBenchmarker(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model: Model = benchmarker.upload_model("YOUR_MODEL_PATH")
    benchmark_task = benchmarker.benchmark_model(
        model=model,
        target_device_name=DeviceName.JETSON_NANO,
        target_software_version=SoftwareVersion.JETPACK_4_6,
        wait_until_done=True
    )
