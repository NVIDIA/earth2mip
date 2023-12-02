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

import os
from typing import List

from pydantic import BaseSettings, Field


def _default_local_cache():
    return os.path.join(os.path.expanduser("~"), ".cache", "earth2mip")


def _default_model_registry():
    path = os.path.join(_default_local_cache(), "models")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:  # noqa
        pass
    return path


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
    MODEL_REGISTRY: str = Field(default_factory=_default_model_registry)
    LOCAL_CACHE: str = Field(default_factory=_default_local_cache)

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

    # End point for s3 commands
    S3_ENDPOINT: str = "https://pbss.s8k.io"

    class Config:
        env_file = ".env"
