Download Model
==============

.. autofunction:: netspresso.compressor.__init__.ModelCompressor.download_model


Example
-------

.. code-block:: python

    from netspresso.compressor import ModelCompressor


    compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    compressor.download_model(
        model_id="YOUR_MODEL_ID",
        local_path="YOUR_LOCAL_PATH",  # ex) ./downloaded_model.h5
    )
