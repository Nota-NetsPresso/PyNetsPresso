Get Conversion Task
===================

.. autofunction:: netspresso.launcher.__init__.ModelConverter.get_conversion_task


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelConverter


    converter = ModelConverter(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    converter.get_conversion_task("YOUR_CONVERSION_TASK")
