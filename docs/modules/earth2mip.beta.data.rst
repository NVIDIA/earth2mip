Data Sources
============

.. warning ::

    Experimental APIs, here be dragons.

Data sources used for downloading, caching and reading different weather / climate data
APIs into `Xarray data arrays <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_.
Used for fetching initial conditions for inference and validation data for scoring.

.. note ::

    Each data source may have its own respective license. We encourage users to
    familiarize themselves with each and the limitations it may impose on their use
    case.

earth2mip.beta.data.CDS
------------------------

.. autoclass:: earth2mip.beta.data.CDS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.ERA5H5
----------------------------------

.. autoclass:: earth2mip.beta.data.ERA5H5
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.GFS
-----------------------------------------

.. autoclass:: earth2mip.beta.data.GFS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.IFS
-----------------------------------------

.. autoclass:: earth2mip.beta.data.IFS
   :members: __call__, available
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.Random
-----------------------------------------

.. autoclass:: earth2mip.beta.data.Random
   :members: __call__
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.DataArrayFile
-----------------------------------------

.. autoclass:: earth2mip.beta.data.DataArrayFile
   :members: __call__
   :undoc-members:
   :show-inheritance:

earth2mip.beta.data.DataSetFile
-----------------------------------------

.. autoclass:: earth2mip.beta.data.DataSetFile
   :members: __call__
   :undoc-members:
   :show-inheritance: