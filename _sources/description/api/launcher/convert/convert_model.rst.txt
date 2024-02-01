Convert Model
=============

.. autofunction:: netspresso.launcher.__init__.ModelConverter.convert_model


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelConverter, ModelFramework, DeviceName, SoftwareVersion


    converter = ModelConverter(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model = converter.upload_model("YOUR_MODEL_PATH")
    conversion_task = converter.convert_model(
        model=model,
        input_shape=model.input_shape,
        target_framework=ModelFramework.TENSORRT,
        target_device_name=DeviceName.JETSON_AGX_ORIN,
        target_software_version=SoftwareVersion.JETPACK_5_0_1,
        wait_until_done=True
    )
