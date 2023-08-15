Bring your own model
====================

To bring you own model, either implement

1. :py:class:`earth2mip.time_loop.TimeLoop`, or
2. wrap your :py:class:`torch.nn.Module` in a
   :py:class:`earth2mip.networks.Inference` object. This approach can be
   convenient if the you have a simple time-stepper torch module that requires
   standard normalization.

The ``TimeLoop`` method is more general, so we can demonstrate it here.
Let's begin by implementing a class for doing a persistence forecast. A
persistence forecast is one that always returns the initial condition. It is a
common baseline for a weather forecast. Here is an implementation::

    from earth2mip.time_loop import TimeLoop
    import datetime

    class Peristence(TimeLoop):
        time_step = datetime.timedelta(hours=12)

        def __call__(self, time, x, restart=None):
            while True:
                yield time, x, None
                time += self.time_step

Note that this time loop does not support restart capability.

On its own, ``Persistence`` can be passed to inference functions like
:py:func:`earth2mip.inference_medium_range.score_deterministic`, but often we
want to run one of these functions in a parallel job using one of the
:ref:`cli`.

Bring you own data
==================

Implement a custom data source like
:py:class:`earth2mip.initial_conditions.era5.HDF5DataSource`
