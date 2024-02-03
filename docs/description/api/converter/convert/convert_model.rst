Convert Model
=============

.. autofunction:: netspresso.converter.__init__.Converter.convert_model


Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import DeviceName, Framework, SoftwareVersion


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    converter = netspresso.converter()
    conversion_task = converter.convert_model(
        input_model_path="./examples/sample_models/test.onnx",
        output_dir="./outputs/converted/TENSORRT_JETSON_AGX_ORIN_JETPACK_5_0_1",
        target_framework=Framework.TENSORRT,
        target_device_name=DeviceName.JETSON_AGX_ORIN,
        target_software_version=SoftwareVersion.JETPACK_5_0_1,
    )
