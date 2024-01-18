.. _concepts:

Concepts
========

Earth2 MIP has the following concepts:

* Layered abstractions for :ref:`wrapping machine learning models<model wrappers>`
* :ref:`Data Sources <data source>` for loading initial conditions and scoring against.
* :ref:`Python APIs <api>` work with these abstractions to do things like scoring and ensemble inference.
* :ref:`Model packages <model package>` allow reproducible loading of checkpoints from disk or the cloud.  that work with these abstractions.
* :ref:`Command line tools <cli>` that allow running scoring jobs in a parallel environment.

.. _model wrappers:
Model Wrappers
--------------

The core model-related abstractions are

#. Machine learning Module (e.g. ``torch.nn.Module``)
#. Time loop (:py:class:`earth2mip.time_loop.TimeLoop`).
#. Forecast (:py:class:`earth2mip.forecasts.Forecast`)

Each of these presents increasingly minimal interface.

We will demonstrate each of these interfaces for the persistence forecast.
A persistence forecast is one that always returns the initial condition. It is a
common baseline for a weather forecast.

A design philosphy we follow is that model wrappers should take plain torch
tensors as inputs and outputs and provide static metadata (e.g. grid, channel,
time) about how those tensors map onto the planet.
An alternative philosophy is to use some kind of metadata-aware container either
custom-built or off-the-shelf like xarray.
We avoid the latter approach since there is always a temptation to  implement all of
mathematics on some container type or hide the arrays under several layers of containers.
At the end of the day, most ML models ultimately take and receive a single multi
dimesional array of data.
Finally, ML
developers are all familiar with basic array-like data types.
We already need to wrap our models in a stack of abstractions to
provide a common interface, so we don't need to make the data more complex.

Module
^^^^^^

At the root is a machine learning model with inputs/outputs that correspond to
two-dimensional fields defined on the planet. These fields are reified as some
array-like data structure, the names of the fields/channels, and grid object
(:py:class:`earth2mip.grid.LatLongGrid`). To use a Module, fcn-mip needs to be provided
metadata about the Model's inputs/outputs (see. :py:class:`earth2mip.schema.Model`).

Here as how to implement the persistence forecast as module::

    import torch.nn

    class PersistenceModule(torch.nn.Module):
        def forward(self, x):
            return x

This is a very simple, but has no metadata about the input or output, and from
an outside perspective, can be difficult to use if ``x`` has more semantic
meaning or requirements.

Time Stepper
^^^^^^^^^^^^

.. autoclass:: earth2mip.time_loop.TimeStepper
    :members:

Time Loop
^^^^^^^^^

A Time Loop is a higher level interface encapsulating the Module, but also
timestepping logic, data preprocessing, and output.
:py:class:`earth2mip.time_loop.TimeLoop` defines this interface, and
:py:class:`earth2mip.networks.Inference` is implements this interface and can
turn a Module into a TimeLoop.

Here is a TimeLoop of the persistence forecast::

    from earth2mip.time_loop import TimeLoop
    from earth2mip.schema import Grid
    import datetime

    class PeristenceTimeLoop(TimeLoop):
        time_step = datetime.timedelta(hours=12)
        # 1 history level = only the current time as input
        n_history_levels = 1
        in_channel_names = ["a", "b", "c"]
        out_channel_names = ["a", "b", "c"]
        grid = Grid.grid_721x1440

        def __call__(self, time, x, restart=None):
            b, h, c, w, h == x.shape

            assert b == 1
            assert h == self.n_history_levels
            assert c == len(self.in_channel_names)
            assert (w, h) == self.grid.shape

            while True:
                yield time, x, None
                time += self.time_step

This encapsulates the time stepping, and exposes other needed metadata.

.. note::
    This time loop does not support restart capability.

Forecast
--------

Many scoring algorithms are most easily expressed as operations over 2D array of
states that we call a Forecast Array. The rows of this array correspond to
initial times, and the columns to lead times. The size of this array may be
unbounded.
For example, computing a lead time dependent metrics, such as RMSE
corresponds to averaging the square difference of Forecast Arrays of
observations and forecasts, and then averaging over the row dimension.
This is defined by the :py:class:`earth2mip.forecasts.Forecast` interface.
Compared to a TimeLoop, a Forecast encapsulates any time handling and initialization logic.
One advantage is that an archive of forecasts on disk can be represented as a Forecast
(see :py:class:`earth2mip.forecasts.XarrayForecast`).
This allows using the same code to score both static and streaming forecasts.

Finally, here is a :py:class:`earth2mip.forecasts.Forecast` implementation, for
a persistence forecast beginning on Jan 1, 2018 and producing ICs every 12
hours and sampling forecasts every 12 hours::

    from earth2mip.forecasts import Forecast
    import datetime

    class PeristenceForecast(Forecast):
        # only corresponds to out_channel_names
        channel_names = ["a", "b", "c"]

        def __init__(self, initial_data: Mapping[datetime.datetime, np.ndarray]):
            self.initial_data = initial_data

        def __getitem__(self, i):
            initial_time = datetime.datetime(2018, 1, 1)
            lead_dt = init_dt = datetime.timedelta(hours=12)

            time = initial_time + init_dt * i
            x = self.initial_data[time]
            while True:
                yield x
we can see that ``PeristenceForecast`` encapsulates the initialization, time and
other logic.


Translating between Model Wrappers
----------------------------------

earth2mip provides implementations that translate between :ref:`model wrappers`.

To create a TimeLoop from a Module, use
:py:class:`earth2mip.networks.Inference`::

    from earth2mip.networks import Inference

    model = PersistenceModule()

    # work around to not do any normalization
    center = np.zeros([3])
    scale = np.ones([3])
    time_loop = Inference(
        model,
        center=center,
        scale=scale,
        grid=Grid.grid_721x1440,
        time_step=datetime.timedelta(hours=12),
        # note n_history_levels == n_history + 1
        n_history=0,
        channel_names=["a", "b", "c"],
    )

To create a forecast from a TimeLoop, you can use
:py:class:`earth2mip.forecasts.TimeLoopForecast`::

    from earth2mip.forecasts import TimeLoopForecast

    forecast = TimeLoopForecast(
        time_loop,
        initial_data={
            datetime.datetime(2018, 1, 1): np.zeros([1, 1, 3, 721, 1440])
        },
    )


.. _data source:

Data Source
-----------

:py:class:`earth2mip.initial_conditions.hdf5.DataSource`


.. _model package:

Model package
-------------

A model package is a directory containing a ``metadata.json`` file following
:py:ref:`this schema <earth2mip.schema.Model>` and any other static data
required to load the model.  The model package typically contains model
parameters, normalization constants, etc.
