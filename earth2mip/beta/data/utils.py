from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import xarray as xr


def prep_data_array(
    da: xr.DataArray, device: Optional[torch.device] = "cpu"
) -> tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
    """Prepares a data array from a data source for inference workflows by converting
    the data array to a torch tensor and the coordinate system to an OrderedDict.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"

    Returns
    -------
    tuple[torch.Tensor, OrderedDict[str, np.ndarray]]
        Tuple containing output tensor and coordinate OrderedDict
    """
    out = torch.Tensor(da.values).to(device)

    out_coords = OrderedDict()
    for dim in da.coords.dims:
        out_coords[dim] = np.array(da.coords[dim])

    return out, out_coords
