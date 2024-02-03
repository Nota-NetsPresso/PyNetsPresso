Get Model
=========

.. autofunction:: netspresso.compressor.__init__.Compressor.get_model


Details of Returns
------------------

.. autoclass:: netspresso.compressor.core.model.Model
    :noindex:

.. autoclass:: netspresso.compressor.core.model.CompressedModel
    :show-inheritance:
    :noindex:


Example
-------

If the model is uploaded model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    model = compressor.get_model(model_id="YOUR_UPLOADED_MODEL_ID")

Output
++++++

.. code-block:: bash

    >>> model
    Model(
        model_id="YOUR_MODEL_ID",
        model_name="YOUR_MODEL_NAME",
        task="image_classification",
        framework="tensorflow_keras", 
        input_shapes=[InputShape(batch=1, channel=3, dimension=[32, 32])],
        model_size=12.9641,
        flops=92.8979,
        trainable_parameters=3.3095,
        non_trainable_parameters=0.0219,
        number_of_layers=0,
    )


If the model is compressed model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    model = compressor.get_model(model_id="YOUR_COMPRESSED_MODEL_ID")

Output
++++++

.. code-block:: bash

    >>> model
    CompressedModel(
        model_id="YOUR_MODEL_ID",
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
        original_model_id="f482a0d3-0321-49a4-a8d2-bd88ac124230"
    )