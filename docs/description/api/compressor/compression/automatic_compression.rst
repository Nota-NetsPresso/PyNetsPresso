Automatic Compression
=====================

.. autofunction:: netspresso.compressor.__init__.ModelCompressor.automatic_compression


Details of Parameters
---------------------

Compression Ratio
~~~~~~~~~~~~~~~~~

.. note::
    - As the compression ratio increases, you can get more lighter and faster compressed models, but with greater lost of accuracy. 
    - Therefore, it is necessary to find an appropriate ratio for your requirements. It might require a few trials and errors.
    - The range of available values is as follows.

    .. raw:: html
        
        <div align="center">
            <img src="https://latex.codecogs.com/svg.image?0<&space;ratio&space;\leq&space;1" title="https://latex.codecogs.com/svg.image?0< ratio \leq 1" />
        </div>



Details of Returns
------------------

.. autoclass:: netspresso.compressor.__init__.CompressedModel
    :show-inheritance:
    :noindex:


Example
-------

.. code-block:: python

    from netspresso.compressor import ModelCompressor


    compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    compressed_model = compressor.automatic_compression(
        model_id="YOUR_UPLOADED_MODEL_ID",
        model_name="YOUR_COMPRESSED_MODEL_NAME",
        output_path="OUTPUT_PATH",  # ex) ./compressed_model.h5
        compression_ratio=0.5,
    )

Output
~~~~~~

.. code-block:: bash

    >>> compressed_model
    CompressedModel(
        model_id="78f65510-1f99-4856-99d9-60902373bd1d", 
        model_name="YOUR_COMPRESSED_MODEL_NAME", 
        task="image_classification", 
        framework="tensorflow_keras", 
        input_shapes=[InputShape(batch=1, channel=3, dimension=[32, 32])], 
        model_size=2.9439, 
        flops=24.1811, 
        trainable_parameters=0.6933, 
        non_trainable_parameters=0.01, 
        number_of_layers=0, 
        compression_id="b9feccee-d69e-4074-a225-5417d41aa572", 
        original_model_id="YOUR_UPLOADED_MODEL_ID"
    )
