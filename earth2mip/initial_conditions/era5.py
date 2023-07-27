# flake8: noqa
import os
from earth2mip.datasets import era5
from earth2mip import config
import xarray
import datetime
import s3fs
import json
from earth2mip import schema
from modulus.utils import filesystem
import logging

__all__ = ["open_era5_xarray"]

logger = logging.getLogger(__name__)
# TODO move to earth2mip/datasets/era5?

# def _get_path(path: str, time) -> str:
#     filename = time.strftime("%Y.h5")
#     h5_files = filesystem.glob(os.path.join(path, "*/*.h5"))
#     files = {os.path.basename(f): f for f in h5_files}
#     return files[filename]


def open_era5_xarray(
    time: datetime.datetime, channel_set: schema.ChannelSet
) -> xarray.DataArray:

    root = config.get_data_root(channel_set)
    raise NotImplementedError(
        "Need to update era5 method to get explicit file name (replace glob)"
    )
    # path = _get_path(root, time)  # TODO: REMOVE/REPLACE

    logger.debug(f"Opening {path} for {time}.")

    if path.endswith(".h5"):
        if path.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
            f = fs.open(path)
        else:
            f = None
        if channel_set == schema.ChannelSet.var34:
            ds = era5.open_34_vars(path, f=f)
        else:
            metadata_path = os.path.join(config.ERA5_HDF5_73, "data.json")
            metadata_path = filesystem.download_cached(metadata_path)
            with open(metadata_path) as mf:
                metadata = json.load(mf)
            ds = era5.open_hdf5(path=path, f=f, metadata=metadata)

    elif path.endswith(".nc"):
        ds = xarray.open_dataset(path).fields

    return ds
