Upload Model
=============

.. autofunction:: netspresso.compressor.__init__.Compressor.upload_model


Details of Parameters
---------------------

Framework
~~~~~~~~~

.. autoclass:: netspresso.enums.__init__.Framework
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

    from netspresso.enums import Framework

    FRAMEWORK = Framework.PYTORCH

.. note::
    - ONNX (.onnx)
        - Supported version: PyTorch >= 1.11.x, ONNX >= 1.10.x.
        - If a model is defined in PyTorch, it should be converted into the ONNX format before being uploaded.
        - `How-to-guide for the conversion of PyTorch into ONNX.`_

    - PyTorch GraphModule (.pt)
        - Supported version: PyTorch >= 1.11.x.
        - If a model is defined in PyTorch, it should be converted into the GraphModule before being uploaded.
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

.. autoclass:: netspresso.compressor.core.model.Model
    :noindex:


Example
-------

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    model = compressor.upload_model(
        input_model_path="./examples/sample_models/mobilenetv1.h5",
        input_shapes=[{"batch": 1, "channel": 3, "dimension": [224, 224]}],
    )
