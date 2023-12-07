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
# TODO add graphcast license
import datetime

import torch

import earth2mip.grid
from earth2mip.time_loop import GeoTensorInfo, TimeStepperLoop


def test_time_stepper_loop():
    # Create a dummy TimeStepper object
    grid = earth2mip.grid.equiangular_lat_lon_grid(4, 8)

    class DummyTimeStepper:
        def __init__(self):
            self.input_info = GeoTensorInfo(["a", "b"], grid=grid)
            self.output_info = GeoTensorInfo(["a", "c"], grid=grid)
            self.time_step = datetime.timedelta(minutes=1)
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        def initialize(self, x, time):
            return x

        def step(self, state):
            state = state + 1
            return state, state[:, 0]

    # Create a TimeStepperLoop object
    stepper = DummyTimeStepper()
    time_loop = TimeStepperLoop(stepper)

    # Test properties
    assert time_loop.in_channel_names == ["a", "b"]
    assert time_loop.out_channel_names == ["a", "c"]
    assert time_loop.grid == grid
    assert time_loop.n_history_levels == 1
    assert time_loop.time_step == stepper.time_step
    assert time_loop.device == stepper.device
    assert time_loop.dtype == stepper.dtype

    # Test __call__ method
    time = datetime.datetime(2023, 1, 1)
    b, t, c, y, x = 1, 1, 2, *grid.shape
    x = torch.zeros([b, t, c, y, x])
    restart = None
    iterator = time_loop(time, x, restart)

    def assert_value_good(value, step_num):
        new_time, data, y = value
        assert new_time == time + time_loop.time_step * step_num

        if step_num == 0:
            assert torch.all(data[0, 0] == 0)
            # "d" is not present in input so needs to be filled in with nan in
            # first step
            assert torch.all(torch.isnan(data[0, 1]))
        else:
            assert torch.all(data == step_num)

        assert torch.all(y == step_num)

    assert_value_good(next(iterator), 0)
    assert_value_good(next(iterator), 1)
    assert_value_good(next(iterator), 2)
