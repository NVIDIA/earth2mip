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

"""
Pangu Weather adapter

adapted from https://raw.githubusercontent.com/ecmwf-lab/ai-models-panguweather/main/ai_models_panguweather/model.py

# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
"""
# %%
from typing import List
import logging
import os
import datetime
import torch

import numpy as np
import onnxruntime as ort
import dataclasses

from earth2mip import registry, schema, networks, config, initial_conditions, geometry


class PanguWeather:
    # Download
    download_url = (
        "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/{file}"
    )
    download_files = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]
    param_sfc = ["msl", "u10m", "v10m", "t2m"]
    param_level_pl = (
        ["z", "q", "t", "u", "v"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )
    # Output
    expver = "pguw"
    # providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, path):
        self.path = path
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1

        pangu_weather = self.path
        # That will trigger a FileNotFoundError
        self.device_index = torch.cuda.current_device()

        os.stat(self.path)
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": self.device_index,
                },
            )
        ]

        self.ort_session = ort.InferenceSession(
            self.path,
            sess_options=options,
            providers=providers,
        )

    def __call__(self, fields_pl, fields_sfc):
        assert fields_pl.dtype == torch.float32
        assert fields_sfc.dtype == torch.float32
        # from https://onnxruntime.ai/docs/api/python/api_summary.html
        binding = self.ort_session.io_binding()

        def bind_input(name, x):
            x = x.contiguous()

            binding.bind_input(
                name=name,
                device_type="cuda",
                device_id=self.device_index,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )

        def bind_output(name, like):
            x = torch.empty_like(like).contiguous()
            binding.bind_output(
                name=name,
                device_type="cuda",
                device_id=self.device_index,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            return x

        bind_input("input", fields_pl)
        bind_input("input_surface", fields_sfc)
        output = bind_output("output", like=fields_pl)
        output_sfc = bind_output("output_surface", like=fields_sfc)
        self.ort_session.run_with_iobinding(binding)
        return output, output_sfc


class PanguStacked:
    def __init__(self, model: PanguWeather):
        self.model = model

    def channel_names(self):
        variables, levels = self.model.param_level_pl
        names = []
        for v in variables:
            for lev in levels:
                names.append(v + str(lev))

        for v in self.model.param_sfc:
            names.append(v)
        return names

    def __call__(self, x):
        return self.forward(x)

    def to(self):
        pass

    def forward(self, x):
        assert x.shape[0] == 1
        assert x.shape[1] == len(self.channel_names())
        pl_shape = (5, 13, 721, 1440)
        nchan = pl_shape[0] * pl_shape[1]
        pl = x[:, :nchan]
        surface = x[:, nchan:]
        pl = pl.resize(*pl_shape)
        sl = surface[0]
        plo, slo = self.model(pl, sl)
        return torch.cat(
            [
                plo.resize(1, nchan, 721, 1440),
                slo.resize(1, x.size(1) - nchan, 721, 1440),
            ],
            dim=1,
        )


class PanguInference(torch.nn.Module):
    n_history_levels = 1
    time_step = datetime.timedelta(hours=6)

    def __init__(self, model_6: PanguStacked, model_24: PanguStacked):
        super().__init__()
        self.model_6 = model_6
        self.model_24 = model_24
        self.channels = None

    def to(self, device):
        return self

    def cuda(self, device=None):
        return self

    @property
    def in_channel_names(self):
        return self.channel_names

    @property
    def out_channel_names(self):
        return self.channel_names

    @property
    def grid(self):
        return schema.Grid.grid_721x1440

    @property
    def channel_set(self):
        return schema.ChannelSet.var_pangu

    @property
    def channel_names(self):
        return schema.ChannelSet.var_pangu.list_channels()

    @property
    def n_history(self):
        return 0

    def normalize(self, x):
        # No normalization for pangu
        return x

    def run_steps_with_restart(self, x, n, normalize=True, time=None):
        """Yield (time, unnormalized data, restart) tuples

        restart = (time, unnormalized data)
        """
        assert normalize, "normalize=False not supported"
        # do not implement restart capability
        restart_data = None

        for k, data in enumerate(self(time, x)):
            yield data
            if k == n:
                break

        yield from self.__call__(time, x)

    def __call__(self, time, x, restart=None):
        """Yield (time, unnormalized data, restart) tuples

        restart = (time, unnormalized data)
        """
        if restart:
            raise NotImplementedError("Restart capability not implemented.")
        # do not implement restart capability
        restart_data = None

        with torch.no_grad():
            x0 = x[:, -1].clone()
            yield time, x0, restart_data
            time0 = time
            while True:
                x1 = x0
                time1 = time0
                for i in range(3):
                    time1 += datetime.timedelta(hours=6)

                    x1 = self.model_6(x1)
                    yield time1, x1, restart_data

                time0 += datetime.timedelta(hours=24)
                x0 = self.model_24(x0)
                yield time0, x0, restart_data


def load(package, *, pretrained=True, device="doesn't matter"):
    """Load the sub-stepped pangu weather inference"""
    assert pretrained

    p6 = package.get("pangu_weather_6.onnx")
    p24 = package.get("pangu_weather_24.onnx")

    model_6 = PanguStacked(PanguWeather(p6))
    model_24 = PanguStacked(PanguWeather(p24))
    return PanguInference(model_6, model_24)


def load_single_model(
    package, *, time_step_hours: int, pretrained=True, device="cuda:0"
):
    """Load a single time-step pangu weather"""
    assert pretrained
    with torch.cuda.device(device):
        p = package.get("model.onnx")
        model = PanguStacked(PanguWeather(p))
        channel_names = model.channel_names()
        center = np.zeros([len(channel_names)])
        scale = np.ones([len(channel_names)])
        grid = schema.Grid.grid_721x1440
        dt = datetime.timedelta(hours=time_step_hours)
        inference = networks.Inference(
            model,
            channels=None,
            center=center,
            scale=scale,
            grid=grid,
            channel_names=channel_names,
            channel_set=schema.ChannelSet.var_pangu,
            time_step=dt,
        )
        inference.to(device)
        return inference
