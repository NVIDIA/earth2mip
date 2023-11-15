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
import os

import pandas as pd

from earth2mip import forecast_metrics_io


def test_read_metric_empty(tmpdir):
    series = forecast_metrics_io.read_metrics(tmpdir)
    assert series.empty


def test_write_and_read_metric(tmpdir):
    # Write a metric to the file object
    initial_time = datetime.datetime(2022, 1, 1, 0, 0, 0)
    lead_time = datetime.timedelta(hours=24)
    channel = "t2m"
    metric = "rmse"
    value = 25.6

    # Read the metric from the directory
    directory = tmpdir.mkdir("metrics")
    filename = "test_metric.csv"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        forecast_metrics_io.write_metric(
            f, initial_time, lead_time, channel, metric, value
        )
    metrics = forecast_metrics_io.read_metrics(directory)

    # Check that the metric was written correctly
    expected_index = pd.MultiIndex.from_tuples(
        [(initial_time, lead_time, channel, metric)],
        names=["initial_time", "lead_time", "channel", "metric"],
    )
    expected_value = pd.Series([value], index=expected_index, name="value")
    pd.testing.assert_series_equal(metrics, expected_value)
