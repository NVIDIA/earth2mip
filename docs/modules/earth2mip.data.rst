Data Sources
========================

Data sources used for downloading, caching and reading different weather / climate data
APIs into `Xarray data arrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.
Used for fetching initial conditions for inference and validation data for scoring.

.. note ::

    Each data source may have its own respective license. We encourage users to
    familiarize themselves with each and the limitations it may impose on their use
    case.

earth2mip.data.CDS
------------------------

.. autoclass:: earth2mip.data.CDS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.data.ERA5H5
----------------------------------

.. autoclass:: earth2mip.data.ERA5H5
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.data.GFS
-----------------------------------------

.. autoclass:: earth2mip.data.GFS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.data.IFS
-----------------------------------------

.. autoclass:: earth2mip.data.IFS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.data.Random
-----------------------------------------

.. autoclass:: earth2mip.data.Random
   :members: __call__
   :undoc-members:
   :show-inheritance:

earth2mip.data.DataArrayFile
-----------------------------------------

.. autoclass:: earth2mip.data.DataArrayFile
   :members: __call__
   :undoc-members:
   :show-inheritance:

earth2mip.data.DataSetFile
-----------------------------------------

.. autoclass:: earth2mip.data.DataSetFile
   :members: __call__
   :undoc-members:
   :show-inheritance: