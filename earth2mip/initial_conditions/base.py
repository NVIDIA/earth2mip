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
# limitations under the License.from typing import Protocol, List, runtime_checkable

import datetime
from typing import List, Protocol, runtime_checkable

import numpy as np

from earth2mip import grid


@runtime_checkable
class DataSource(Protocol):

    grid: grid.LatLonGrid

    @property
    def channel_names(self) -> List[str]:
        pass

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        """

        Returns:
            data at ``time``. shape is (len(channel_names), *grid.shape)
        """
        pass
