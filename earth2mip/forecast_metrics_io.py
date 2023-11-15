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
"""Routines for reading and writing forecast metrics to a directory of csv files.

The csv files contain the records::

    initial_time_iso, lead_time_hours, channel, metric, value
    2022-01-01T00:00:00,24,t2m,rmse,25.6

"""
import csv
import datetime
import os
from typing import IO

import pandas as pd


def read_metrics(directory: str) -> pd.Series:
    """
    Reads all csv files in the given directory and returns a pandas Series
    containing all the metric values.
    """

    dfs = []
    # Loop through all csv files in the directory
    csv_files = [
        filename for filename in os.listdir(directory) if filename.endswith(".csv")
    ]

    if len(csv_files) == 0:
        return pd.Series()

    for filename in csv_files:
        filepath = os.path.join(directory, filename)
        # Read the csv file into a pandas DataFrame
        df = pd.read_csv(
            filepath,
            header=None,
            names=[
                "initial_time_iso",
                "lead_time_hours",
                "channel",
                "metric",
                "value",
            ],
        )
        out = pd.DataFrame()
        out["initial_time"] = pd.to_datetime(df["initial_time_iso"])
        out["lead_time"] = pd.to_timedelta(df["lead_time_hours"], unit="h")
        out["channel"] = df["channel"]
        out["metric"] = df["metric"]
        out["value"] = df["value"]
        dfs.append(out)

    df = pd.concat(dfs, axis=0)
    df.set_index(["initial_time", "lead_time", "channel", "metric"], inplace=True)
    return df["value"]


def write_metric(
    f: IO[str],
    initial_time: datetime.datetime,
    lead_time: datetime.timedelta,
    channel: str,
    metric: str,
    value: float,
) -> None:
    """
    Writes a single metric value to the given file object in csv format.
    """
    # Write the metric value to the file object in csv format
    writer = csv.writer(f)
    writer.writerow(
        [
            initial_time.isoformat(),
            lead_time.total_seconds() // 3600,
            channel,
            metric,
            value,
        ]
    )
