Manual Compression
==================

Select Compression Method
-------------------------

.. autofunction:: netspresso.compressor.__init__.Compressor.select_compression_method


Details of Parameters
~~~~~~~~~~~~~~~~~~~~~


Compression Method
++++++++++++++++++

.. autoclass:: netspresso.enums.__init__.CompressionMethod
    :noindex:


Available Compression Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
+------------+------------------------------+
| Name       | Description                  |
+============+==============================+
| PR_L2      | L2 Norm Pruning              |
+------------+------------------------------+
| PR_GM      | GM Pruning                   |
+------------+------------------------------+
| PR_NN      | Nuclear Norm Pruning         |
+------------+------------------------------+
| PR_ID      | Pruning By Index             |
+------------+------------------------------+
| FD_TK      | Tucker Decomposition         |
+------------+------------------------------+
| FD_SVD     | Singular Value Decomposition |
+------------+------------------------------+
| FD_CP      | CP Decomposition             |
+------------+------------------------------+

Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from netspresso.enums import CompressionMethod

    COMPRESSION_METHOD = CompressionMethod.PR_L2

.. warning::
    - Nuclear Norm is only supported in the Tensorflow-Keras Framework.

.. note::

    - Click on the link to learn more about the information. (:ref:`compression_method_heading`)

Options
~~~~~~~

.. autoclass:: netspresso.enums.__init__.Policy
    :noindex:

.. autoclass:: netspresso.enums.__init__.LayerNorm
    :noindex:

.. autoclass:: netspresso.enums.__init__.GroupPolicy
    :noindex:


Example
+++++++

.. code-block:: python

    from netspresso.enums import Policy, LayerNorm, GroupPolicy, Options

    OPTIONS = Options(
        policy=Policy.AVERAGE,
        layer_norm=LayerNorm.TSS_NORM,
        group_policy=GroupPolicy.COUNT,
        reshape_channel_axis=-1
    )

.. note::
    
    - Click the link for more information. (:ref:`pruning_options_heading`)

.. note::

    - This parameter applies only to the Pruning Method (PR_L2, PR_GM, PR_NN).

Details of Returns
~~~~~~~~~~~~~~~~~~


Example
+++++++

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import CompressionMethod, Policy, LayerNorm, GroupPolicy, Options


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compression_info = compressor.select_compression_method(
        model_id="YOUR_UPLOADED_MODEL_ID",
        compression_method=CompressionMethod.PR_L2,
        options=Options(
            policy=Policy.AVERAGE,
            layer_norm=LayerNorm.STANDARD_SCORE,
            group_policy=GroupPolicy.AVERAGE,
            reshape_channel_axis=-1,
        ),
    )

Output
^^^^^^

.. code-block:: bash

    >>> compression_info
    CompressionInfo(
        compression_method="PR_L2", 
        available_layers=[
            AvailableLayer(name='conv1', values=[""], channels=[32]), 
            AvailableLayer(name='layers.0.conv2', values=[""], channels=[64]), 
            AvailableLayer(name='layers.1.conv2', values=[""], channels=[128]), 
            AvailableLayer(name='layers.2.conv2', values=[""], channels=[128]), 
            AvailableLayer(name='layers.3.conv2', values=[""], channels=[256]), 
            AvailableLayer(name='layers.4.conv2', values=[""], channels=[256]), 
            AvailableLayer(name='layers.5.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.6.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.7.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.8.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.9.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.10.conv2', values=[""], channels=[512]), 
            AvailableLayer(name='layers.11.conv2', values=[""], channels=[1024]), 
            AvailableLayer(name='layers.12.conv2', values=[""], channels=[1024])
        ], 
        options={'reshape_channel_axis': -1, 'policy': 'average', 'layer_norm': 'tss_norm', 'group_policy': 'average'}
        original_model_id="YOUR_UPLOADED_MODEL_ID",
        compressed_model_id="", 
        compression_id="", 
    )


Set Compression Params
----------------------

Details of Parameters
~~~~~~~~~~~~~~~~~~~~~


Values of available layer
+++++++++++++++++++++++++

.. rst-class:: table

+--------------------+------------------+--------+---------------------------------------+
| Compression Method | Number of Values | Type   | Range                                 |
+====================+==================+========+=======================================+
| PR_L2              | 1                | Float  | 0.0 < ratio ≤ 1.0                     |
+--------------------+------------------+--------+---------------------------------------+
| PR_GM              | 1                | Float  | 0.0 < ratio ≤ 1.0                     |
+--------------------+------------------+--------+---------------------------------------+
| PR_NN              | 1                | Float  | 0.0 < ratio ≤ 1.0                     |
+--------------------+------------------+--------+---------------------------------------+
| PR_ID              | (Num of Out      | Int    | 0 < channels ≤ Num of Out Channels    |
|                    | Channels - 1)    |        |                                       |
+--------------------+------------------+--------+---------------------------------------+
| FD_TK              | 2                | Int    | 0 < rank ≤ (Num of In Channels or     |
|                    |                  |        | Num of Out Channels)                  |
+--------------------+------------------+--------+---------------------------------------+
| FD_CP              | 1                | Int    | 0 < rank ≤ min(Num of In Channels or  |
|                    |                  |        | Num of Out Channels)                  |
+--------------------+------------------+--------+---------------------------------------+
| FD_SVD             | 1                | Int    | 0 < rank ≤ min(Num of In Channels or  |
|                    |                  |        | Num of Out Channels)                  |
+--------------------+------------------+--------+---------------------------------------+


Example
+++++++

.. code-block:: python


   for available_layer in compression_info.available_layers:
      available_layer.values = [0.2]

Output
^^^^^^

.. code-block:: bash

      >>> compression_info
      CompressionInfo(
         compression_method="PR_L2", 
         available_layers=[
            AvailableLayer(name='conv1', values=[0.2], channels=[32]), 
            AvailableLayer(name='layers.0.conv2', values=[0.2], channels=[64]), 
            AvailableLayer(name='layers.1.conv2', values=[0.2], channels=[128]), 
            AvailableLayer(name='layers.2.conv2', values=[0.2], channels=[128]), 
            AvailableLayer(name='layers.3.conv2', values=[0.2], channels=[256]), 
            AvailableLayer(name='layers.4.conv2', values=[0.2], channels=[256]), 
            AvailableLayer(name='layers.5.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.6.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.7.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.8.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.9.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.10.conv2', values=[0.2], channels=[512]), 
            AvailableLayer(name='layers.11.conv2', values=[0.2], channels=[1024]), 
            AvailableLayer(name='layers.12.conv2', values=[0.2], channels=[1024])
         ], 
         options={'reshape_channel_axis': -1, 'policy': 'average', 'layer_norm': 'tss_norm', 'group_policy': 'average'}
         original_model_id="YOUR_UPLOADED_MODEL_ID",
         compressed_model_id="", 
         compression_id="", 
      )


Compress Model
--------------

.. autofunction:: netspresso.compressor.__init__.Compressor.compress_model


Example
+++++++

.. code-block:: python


    compressed_model = compressor.compress_model(
        compression=compression_info,
        output_dir="./outputs/compressed/graphmodule_manual",
    )
