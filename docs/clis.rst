.. _cli:

Command Line Utilities
======================

While one can certainly write their own script using the tools in the
:ref:`api`, this has overhead. For each script, one will need to
initialize the torch distributed group if running in parallel, and write a
submission script if running on the cluster. To simplify deployment, we also
include CLIs corresponding to the common APIs.

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - CLI
     - API
   * - ``python -m earth2mip.inference_medium_range``
     - :py:func:`earth2mip.inference_medium_range.score_deterministic`
   * - ``python -m earth2mip.inference_ensemble``
     - :py:func:`earth2mip.inference_ensemble.run_inference`
   * - ``python -m earth2mip.lagged_ensembles``
     - -
