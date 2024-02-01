Download Converted Model
========================

.. autofunction:: netspresso.launcher.__init__.ModelConverter.download_converted_model


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelConverter


    converter = ModelConverter(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model = converter.upload_model("YOUR_MODEL_PATH")
    converter.download_converted_model("YOUR_CONVERSION_TASK", dst="YOUR_CONVERTED_MODEL_PATH")
