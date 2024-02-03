Automatic Compression
=====================

.. autofunction:: netspresso.compressor.__init__.Compressor.automatic_compression


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


Example
-------

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressed_model = compressor.automatic_compression(
        input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
        input_model_path="./examples/sample_models/graphmodule.pt",
        output_dir="./outputs/compressed/pytorch_automatic_compression_1",
        compression_ratio=0.5,
    )
