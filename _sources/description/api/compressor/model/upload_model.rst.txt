Upload Model
=============

.. autofunction:: netspresso.compressor.__init__.ModelCompressor.upload_model


Details of Parameters
---------------------

Task
~~~~

.. autoclass:: netspresso.compressor.__init__.Task
    :noindex:


Available Task
++++++++++++++
+-----------------------+-----------------------+
| Name                  | Description           |
+=======================+=======================+
| IMAGE_CLASSIFICATION  | Image Classification  |
+-----------------------+-----------------------+
| OBJECT_DETECTION      | Object Detection      |
+-----------------------+-----------------------+
| IMAGE_SEGMENTATION    | Image Segmentation    |
+-----------------------+-----------------------+
| SEMANTIC_SEGMENTATION | Semantic Segmentation |
+-----------------------+-----------------------+
| INSTANCE_SEGMENTATION | Instance Segmentation |
+-----------------------+-----------------------+
| PANOPTIC_SEGMENTATION | Panoptic Segmentation |
+-----------------------+-----------------------+
| OTHER                 | Other                 |
+-----------------------+-----------------------+

Example
+++++++

.. code-block:: python

    from netspresso.compressor import Task

    TASK = Task.IMAGE_CLASSIFICATION

Framework
~~~~~~~~~

.. autoclass:: netspresso.compressor.__init__.Framework
    :noindex:

Available Framework
+++++++++++++++++++
+------------------+----------------------+
| Name             | Description          |
+==================+======================+
| TENSORFLOW_KERAS | TensorFlow-Keras     |
+------------------+----------------------+
| PYTORCH          | PyTorch GraphModule  |
+------------------+----------------------+
| ONNX             | ONNX                 |
+------------------+----------------------+

Example
+++++++

.. code-block:: python

    from netspresso.compressor import Framework

    FRAMEWORK = Framework.TENSORFLOW_KERAS

.. note::
    - ONNX (.onnx)
        - Supported version: Pytorch >= 1.11.x, ONNX >= 1.10.x.
        - If a model is defined in Pytorch, it should be converted into the ONNX format before being uploaded.
        - `How-to-guide for the conversion of PyTorch into ONNX.`_

    - PyTorch GraphModule (.pt)
        - Supported version: Pytorch >= 1.11.x.
        - If a model is defined in Pytorch, it should be converted into the GraphModule before being uploaded.
        - The model must contain not only the status dictionary but also the structure of the model (do not use state_dict).
        - `How-to-guide for the conversion of PyTorch into GraphModule.`_

    - TensorFlow-Keras (.h5, .zip)
        - Supported version: TensorFlow 2.3.x ~ 2.8.x.
        - Custom layer must not be included in Keras H5 (.h5) format.
        - The model must contain not only weights but also the structure of the model (do not use save_weights).
        - If there is a custom layer in the model, please upload TensorFlow SavedModel format (.zip).

            .. image:: ../../../../_static/tf-keras.png
                :width: 250
                :align: center

.. _How-to-guide for the conversion of PyTorch into GraphModule.: https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/tree/main/models/torch#conversion-of-pytorch-into-graphmodule-torchfxgraphmodule
.. _How-to-guide for the conversion of PyTorch into ONNX.: https://github.com/Nota-NetsPresso/NetsPresso-Model-Compressor-ModelZoo/tree/main/models/torch#conversion-of-pytorch-into-onnx


Input Shapes
~~~~~~~~~~~~

.. note::
    - For input shapes, use the same values that you used to train the model.

        - If the input shapes of the model is **dynamic**, input shapes is **required**.
        - If the input shapes of the model is **static**, input shapes is **not required**.


    - For example, batch=1, channel=3, **height=768, width=1024**.

        .. code-block:: python

            input_shapes = [{"batch": 1, "channel": 3, "dimension": [768, 1024]}]

    - Currently, **only single input models** are supported.


Details of Returns 
------------------

.. autoclass:: netspresso.compressor.__init__.Model
    :noindex:


Example
-------

.. code-block:: python

    from netspresso.compressor import ModelCompressor, Task, Framework


    compressor = ModelCompressor(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    model = compressor.upload_model(
        model_name="YOUR_MODEL_NAME",
        task=Task.IMAGE_CLASSIFICATION,
        framework=Framework.TENSORFLOW_KERAS,
        file_path="YOUR_MODEL_PATH", # ex) ./model.h5
        input_shapes="YOUR_MODEL_INPUT_SHAPES",  # ex) [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
    )

Output
~~~~~~

.. code-block:: bash

    >>> model
    Model(
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
    )
