Delete Model
============

.. autofunction:: netspresso.compressor.__init__.Compressor.delete_model


Details of Parameters
---------------------

Recursive
~~~~~~~~~

.. warning::

    Deleting the model will also delete its compressed models. To proceed with the deletion,
    set the ``recursive`` parameter to True.

.. note::

    If the ``recursive`` parameter is set to True, the compressed model for that model will also be deleted.


Example
-------

If the model has a compressed model, and recursive is False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressor.delete_model(model_id="YOUR_MODEL_ID")

Output
++++++

.. code-block:: bash

    Deleting model...
    Deleting the model will also delete its compressed models. To proceed with the deletion, set the `recursive` parameter to True.


If the model has a compressed model, and recursive is True
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressor.delete_model(model_id="YOUR_MODEL_ID", recursive=True)

Output
++++++

.. code-block:: bash

    Deleting model...
    The compressed model for that model will also be deleted.
    Delete model successful.


If there is no compressed model in the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from netspresso import NetsPresso


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    compressor = netspresso.compressor()
    compressor.delete_model(model_id="YOUR_MODEL_ID")

Output
++++++

.. code-block:: bash

    Deleting model...
    The model will be deleted.
    Delete model successful.
