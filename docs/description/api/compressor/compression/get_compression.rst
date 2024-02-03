Get Compression Information
===========================

.. autofunction:: netspresso.compressor.__init__.Compressor.get_compression


Details of Returns
------------------

.. autoclass:: netspresso.compressor.core.compression.CompressionInfo
    :noindex:


Example
-------

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressed_info = compressor.get_compression(compression_id="YOUR_COMPRESSION_ID")

Output
~~~~~~

.. code-block:: bash

    >>> compressed_info
    CompressionInfo(
        compressed_model_id="c65ca574-08ab-4a42-9e82-a222bd089b2c",
        compression_id="YOUR_COMPRESSION_ID",
        compression_method="PR_L2",
        available_layers=[
            AvailableLayer(name='conv1', values=[0.59375], channels=[32]), 
            AvailableLayer(name='layers.0.conv2', values=[0.25], channels=[64]), 
            AvailableLayer(name='layers.1.conv2', values=[0.25], channels=[128]), 
            AvailableLayer(name='layers.2.conv2', values=[0.34375], channels=[128]), 
            AvailableLayer(name='layers.3.conv2', values=[0.33203125], channels=[256]), 
            AvailableLayer(name='layers.4.conv2', values=[0.55859375], channels=[256]), 
            AvailableLayer(name='layers.5.conv2', values=[0.56640625], channels=[512]), 
            AvailableLayer(name='layers.6.conv2', values=[0.697265625], channels=[512]), 
            AvailableLayer(name='layers.7.conv2', values=[0.70703125], channels=[512]), 
            AvailableLayer(name='layers.8.conv2', values=[0.580078125], channels=[512]), 
            vailableLayer(name='layers.9.conv2', values=[0.51953125], channels=[512]), 
            AvailableLayer(name='layers.10.conv2', values=[0.517578125], channels=[512]), 
            AvailableLayer(name='layers.11.conv2', values=[0.7734375], channels=[1024]), 
            AvailableLayer(name='layers.12.conv2', values=[0.0234375], channels=[1024])
        ],
        options={'reshape_channel_axis': -1, 'policy': 'average', 'layer_norm': 'tss_norm', 'group_policy': 'average'}
        original_model_id=""
    )
