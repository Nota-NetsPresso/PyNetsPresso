Upload Model
=============

.. autofunction:: netspresso.launcher.__init__.Launcher.upload_model


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelConverter, ModelBenchmarker


    converter = ModelConverter(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model = converter.upload_model("YOUR_MODEL_PATH")

    # or

    benchmarker = ModelBenchmarker(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model = benchmarker.upload_model("YOUR_MODEL_PATH")
