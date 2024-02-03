Recommendation Compression
==========================

.. autofunction:: netspresso.compressor.__init__.Compressor.recommendation_compression


Details of Parameters
---------------------


Compression Method
~~~~~~~~~~~~~~~~~~

.. autoclass:: netspresso.enums.__init__.CompressionMethod
    :noindex:


Available Compression Method
++++++++++++++++++++++++++++
+------------+------------------------------+
| Name       | Description                  |
+============+==============================+
| PR_L2      | L2 Norm Pruning              |
+------------+------------------------------+
| PR_GM      | GM Pruning                   |
+------------+------------------------------+
| PR_NN      | Nuclear Norm Pruning         |
+------------+------------------------------+
| FD_TK      | Tucker Decomposition         |
+------------+------------------------------+
| FD_SVD     | Singular Value Decomposition |
+------------+------------------------------+

Example
+++++++

.. code-block:: python

    from netspresso.enums import CompressionMethod

    COMPRESSION_METHOD = CompressionMethod.PR_L2

.. warning::
    - Nuclear Norm is only supported in the Tensorflow-Keras Framework.

.. note::
    - Click on the link to learn more about the information. (:ref:`compression_method_heading`)

Recommendation Method
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: netspresso.enums.__init__.RecommendationMethod
    :noindex:


Available Recommendation Method
+++++++++++++++++++++++++++++++
+------------+---------------------------------------------------------------------+
| Name       | Description                                                         |
+============+=====================================================================+
| SLAMP      | Structured Layer-adaptive Sparsity for the Magnitude-based Pruning  |
+------------+---------------------------------------------------------------------+
| VBMF       | Variational Bayesian Matrix Factorization                           |
+------------+---------------------------------------------------------------------+

Example
+++++++

.. code-block:: python

    from netspresso.enums import RecommendationMethod

    RECOMMENDATION_METHOD = RecommendationMethod.SLAMP

.. note::
    - If you selected PR_L2, PR_GM, PR_NN for compression_method
        - The recommended_method available is **SLAMP**.
    - If you selected FD_TK, FD_SVD for compression_method
        - The recommended_method available is **VBMF**.


Recommendation Ratio
~~~~~~~~~~~~~~~~~~~~~

.. note::

    - SLAMP (Pruning ratio)

        - Remove corresponding amounts of the filters. (e.g. 0.2 removes 20% of the filters in each layer)

        - Available ranges

        .. raw:: html

            <div align="center" style="padding: 20px;">
                <img src="https://latex.codecogs.com/svg.image?0&space;<ratio&space;\leq&space;&space;1&space;" title="https://latex.codecogs.com/svg.image?0 <ratio \leq 1 " />
            </div>
        
        - Click the link for more information. (`SLAMP`_)

    - VBMF (Calibration ratio)

        - This function control compression level of model if the result of recommendation doesn't meet the compression level user wants. Remained rank add or subtract (removed rank x calibration ratio) according to calibration ratio range.

            .. image:: ../../../../../_static/compression/methods/calibration_ratio.png
                :width: 500
                :align: center


        - Available ranges

        .. raw:: html
                
            <div align="center" style="padding: 20px;">
                <img src="https://latex.codecogs.com/svg.image?-1&space;\leq&space;&space;ratio&space;\leq&space;&space;1&space;" title="https://latex.codecogs.com/svg.image?-1 \leq ratio \leq 1 " />
            </div>
        
        - Click the link for more information. (`VBMF`_)

.. _SLAMP : https://docs.netspresso.ai/docs/mc-structured-pruning#supported-functions
.. _VBMF: https://docs.netspresso.ai/docs/mc-filter-decomposition#recommendation-in-model-compressor


Options
~~~~~~~

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

Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import CompressionMethod, RecommendationMethod


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressed_model = compressor.recommendation_compression(
        compression_method=CompressionMethod.PR_L2,
        recommendation_method=RecommendationMethod.SLAMP,
        recommendation_ratio=0.5,
        input_model_path="./examples/sample_models/graphmodule.pt",
        output_dir="./outputs/compressed/graphmodule_recommend",
        input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
    )
