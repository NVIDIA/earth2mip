Inference Workflows
===================

These python functions work on the :ref:`core objects <concepts>` of earth2mip.
Most are passed a :py:class:`earth2mip.initial_conditions.base.DataSource` and
an py:class:`earth2mip.time_loop.TimeLoop` object. These can be used from within
your own parallel scripts.

General Inference
-----------------

These routines combine many of the features of earth2mip, including ensemble initialization,
post processing,

.. autofunction:: earth2mip.inference_ensemble.run_inference

.. autofunction:: earth2mip.time_collection.run_over_initial_times

Scoring
-------

.. autofunction:: earth2mip.inference_medium_range.score_deterministic