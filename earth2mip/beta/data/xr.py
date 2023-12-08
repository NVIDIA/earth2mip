# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from typing import Any, Dict, List, Union

import xarray as xr


class DataArrayFile:
    """A local data array file data source. This file should be compatable with xarray.
    For example, a netCDF file.

    Parameters
    ----------
    file_path : str
        Path to xarray data array compatible file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def da(self, xr_args: Dict[str, Any] = {}) -> xr.DataArray:
        return xr.open_dataarray(self.file_path, **xr_args)

    def __call__(
        self,
        time: Union[datetime.datetime, List[datetime.datetime]],
        channel: Union[str, List[str]],
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        t : datetime.datetime or List[datetime.datetime]
            Timestamps to return data for.
        channel : str or List[str]
            Strings or list of strings that refer to the
            channel/variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        return self.da.sel(time=time, channel=channel)


class DataSetFile:
    """A local data array file data source. This file should be compatable with xarray.
    For example, a netCDF file.

    Parameters
    ----------
    file_path : str
        Path to xarray dataset compatible file.
    variable_name : str
        Data array name in xarray dataset
    """

    def __init__(self, file_path: str, variable_name: str):
        self.file_path = file_path
        self.variable_name = variable_name

    @property
    def da(self, xr_args: Dict[str, Any] = {}) -> xr.DataArray:
        return xr.open_dataset(self.file_path, **xr_args)[self.variable_name]

    def __call__(
        self,
        time: Union[datetime.datetime, List[datetime.datetime]],
        channel: Union[str, List[str]],
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        t : datetime.datetime or List[datetime.datetime]
            Timestamps to return data for.
        channel : str or List[str]
            Strings or list of strings that refer to the
            channel/variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        return self.da.sel(time=time, channel=channel)
