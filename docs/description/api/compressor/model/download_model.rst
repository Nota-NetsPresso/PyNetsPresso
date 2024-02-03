Download Model
==============

.. autofunction:: netspresso.compressor.__init__.Compressor.download_model


Example
-------

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressor.download_model(
        model_id="YOUR_MODEL_ID",
        local_path="YOUR_LOCAL_PATH",  # ex) ./downloaded_model.h5
    )
