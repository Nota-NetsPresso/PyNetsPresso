Get Compressed Models
=====================

.. autofunction:: netspresso.compressor.__init__.ModelCompressor.get_compressed_models


Details of Returns
------------------

.. autoclass:: netspresso.compressor.__init__.CompressedModel
    :show-inheritance:
    :noindex:


Example
~~~~~~~

.. code-block:: python

    from netspresso.compressor import ModelCompressor


    compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    compressed_models = compressor.get_compressed_models(model_id="YOUR_UPLOADED_MODEL_ID")

Output
~~~~~~

.. code-block:: bash

    >>> compressed_models
    [CompressedModel(
        model_id="8cbd8b0c-68ca-42ae-b84b-921e7462ba88",
        model_name="YOUR_COMPRESSED_MODEL_NAME",
        task="image_classification",
        framework="tensorflow_keras",
        input_shapes=[InputShape(batch=1, channel=3, dimension=[32, 32])],
        model_size=2.9439,
        flops=24.1811,
        trainable_parameters=0.6933,
        non_trainable_parameters=0.01,
        number_of_layers=0,
        compression_id="ce584e7f-b76e-43cc-83fe-d140fe476a58",
        original_model_id="YOUR_UPLOADED_MODEL_ID"
    )]