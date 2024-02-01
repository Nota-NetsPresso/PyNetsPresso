Get Uploaded Models
===================

.. autofunction:: netspresso.compressor.__init__.ModelCompressor.get_uploaded_models


Details of Returns
------------------

.. autoclass:: netspresso.compressor.__init__.Model
    :noindex:


Example
-------

.. code-block:: python

    from netspresso.compressor import ModelCompressor


    compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    uploaded_models = compressor.get_uploaded_models()

Output
~~~~~~

.. code-block:: bash

    >>> uploaded_models
    [Model(
        model_id="5eeb0edb-57d2-4a20-adf4-a6c05516015d",
        model_name="YOUR_MODEL_NAME",
        task="image_classification",
        framework="tensorflow_keras", 
        input_shapes=[InputShape(batch=1, channel=3, dimension=[32, 32])],
        model_size=12.9641,
        flops=92.8979,
        trainable_parameters=3.3095,
        non_trainable_parameters=0.0219,
        number_of_layers=0,
    )]