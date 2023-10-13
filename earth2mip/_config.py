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
from pydantic import BaseSettings


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
    ERA5_HDF5: str = ""
    MODEL_REGISTRY: str = ""
    LOCAL_CACHE: str = ""

    # used for scoring (score-ifs.py, inference-medium-range)
    TIME_MEAN: str = ""

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
