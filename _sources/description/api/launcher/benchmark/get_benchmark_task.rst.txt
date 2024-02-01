Get Benchmark Task
==================

.. autofunction:: netspresso.launcher.__init__.ModelBenchmarker.get_benchmark_task


Example
-------

.. code-block:: python

    from netspresso.launcher import ModelBenchmarker


    benchmarker = ModelBenchmarker(email="YOUR_EMAIL", password="YOUR_PASSWORD")
    benchmark_task = benchmarker.get_benchmark_task(benchmark_task="YOUR_BEBCHMARK_TASK")
