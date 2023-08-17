# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

from earth2mip import schema
import json


def test_model():
    obj = schema.Model(
        architecture="some_arch",
        n_history=0,
        channel_set=schema.ChannelSet.var34,
        grid=schema.Grid.grid_720x1440,
        in_channels=[0, 1],
        out_channels=[0, 1],
    )
    loaded = json.loads(obj.json())
    assert loaded["channel_set"] == obj.channel_set.var34.value
