Bring your own model
====================

fcn-mip provides two options for bringing your own model/data. The first, is
you can build your own script using one of the  abstractions in
:ref:`concepts` and calling one of the inference functions (TODO add
cross reference).  See (examples/pangu_24.py).

However, sometimes it is more convenient to use the :ref:`CLIs <cli>`,
especially when deploying in a cluster environment. Doing so requires creating a
:ref:`model package`. Under the hood, all the CLIs use
:py:func:`earth2mip.networks.get_model`.

One can modify the behavior of :py:func:`earth2mip.networks.get_model` by either:

#. specifying an ``entrypoint`` which returns TimeLoop.
#. specifying an ``archicture_entrypoint`` which returns a ``torch.nn.Module`` that will be wrapped with :py:func:`earth2mip.networks.Inference`

See :ref:`concepts` for more on these methods, but the ``TimeLoop`` method is
more general and often easier to implement. Frequently, your own model training
code base will already have some time looping logic that can be repurposed.

Module Interface
^^^^^^^^^^^^^^^^

This is a lower level interface. It is selected using the
``architecture_entrypoint`` key of ``metadata.json`` file, and providing the
other metadata.

For example, to implement a plugin for a persistence forecast you would create a file ``my_package/plugin`` containing::

    # my_package/plugin.py
    import torch

    class Persistence(torch.nn.Module):
        def forward(self, x):
            return x


    def load_persistence_module(package, pretrained=True, device=None):
        """
        load a model package

        Args:
            package: a fcn_mip.model_registry.Package, has package.get(path), and
                package.metadata() methods.
        """
        # use package.get to open weights etc if needed.
        return Persistence()


Then, you can create a model package in ``/abs/path/to/persistence/``

``/abs/path/to/persistence/metadata.json`` should contain::

    {
        "architecture_entrypoint": "my_package.plugin:load_persistence_module"
        },
        "grid": "721x1440",
        "in_channels": [0, 1, 2],
        "out_channels": [0, 1, 2],
    }

.. note::
    The values of ``entrypoint`` and ``architecture_entrypoint`` are loaded
    using `Python entrypoints
    <https://docs.python.org/3/library/importlib.metadata.html#entry-points>`_.
    They have the form `package.module:function`.

Finally, you should be able to load the model like this in python::

    from earth2mip.networks import get_model
    time_loop = get_model("file://abs/path/to/persistence")

Or if you have an environment variable ``MODEL_REGISTRY=/abs/path/to``::

    time_loop = get_model("persistence")

Once this is working you can pass the model package to one of the :ref:`cli`::

    torchrun --nproc_per_node <ngpu> -m earth2mip.inference_medium_range -n 56 file://abs/path/to/persistence scores.nc

.. warning::

    This currently only supports loading data from the var73 and var34 channel sets.


Time Loop Interface
^^^^^^^^^^^^^^^^^^^

This project contains examples of TimeLoop loaders for Pangu:


Python code:

.. literalinclude:: ../earth2mip/networks/pangu.py
  :pyobject: load

the corresponding ``metadata.json`` file::

    {
      "entrypoint": {
        "name": "earth2mip.networks.pangu:load",
        "kwargs": {
          "time_step_hours": 6
        }
      }
    }

Loading an interleaved time loop:

.. literalinclude:: ../earth2mip/networks/pangu.py
    :pyobject: load_24substep6

the corresponding ``metadata.json`` file::

    {"entrypoint": {"name": "earth2mip.networks.pangu:load_24substep6"}}

Bring you own data
==================

Implement a custom data source like
:py:class:`earth2mip.initial_conditions.hdf5.DataSource`
