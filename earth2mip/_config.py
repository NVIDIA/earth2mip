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

from typing import List
from pydantic import BaseSettings, Field
from earth2mip import schema
import os


def _default_local_cache():
    return os.path.join(os.environ["HOME"], ".cache", "earth2mip")


class Settings(BaseSettings):
    # not needed anymore
    # AFNO_26_WEIGHTS: Optional[str] = None
    # AFNO_26_MEAN: str
    # AFNO_26_SCALE: str

    # only used in earth2mip.diagnostics
    # TODO add defaults (maybe scope in that module)
    MEAN: str = ""
    SCALE: str = ""

    # Key configurations
    ERA5_HDF5_34: str = ""
    ERA5_HDF5_73: str = ""
    MODEL_REGISTRY: str = ""
    LOCAL_CACHE: str = Field(default_factory=_default_local_cache)

    # used for scoring (score-ifs.py, inference-medium-range)
    TIME_MEAN: str = ""
    TIME_MEAN_73: str = ""

    # used in score-ifs.py
    # TODO refactor to a command line argument of that script
    IFS_ROOT: str = None

    # only used in test suite
    # TODO add a default option.
    TEST_DIAGNOSTICS: List[str] = ()

    # where to store regridding files
    MAP_FILES: str = ""

    class Config:
        env_file = ".env"

    def get_data_root(self, channel_set: schema.ChannelSet) -> str:
        if channel_set == schema.ChannelSet.var34:
            val = self.ERA5_HDF5_34
            if not val:
                raise ValueError(
                    "Please configure ERA5_HDF5_34 to point to the 34 channel data."  # noqa
                )
            return val
        elif channel_set == schema.ChannelSet.var73:
            val = self.ERA5_HDF5_73
            if not val:
                raise ValueError("Please configure ERA5_HDF5_73.")
        else:
            raise NotImplementedError(channel_set)

        return val

    def get_time_mean(self, channel_set: schema.ChannelSet) -> str:
        return {
            schema.ChannelSet.var34: self.TIME_MEAN,
            schema.ChannelSet.var73: self.TIME_MEAN_73,
        }[channel_set]
