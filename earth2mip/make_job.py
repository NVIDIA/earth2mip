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
import json
import os
import pathlib

import typer


def get_times_2018():
    nsteps = 728
    times = [
        datetime.datetime(2018, 1, 1) + k * datetime.timedelta(hours=12)
        for k in range(nsteps)
    ]
    return times


def get_times_s2s_test():
    time = datetime.datetime(2020, 1, 2)
    dt = datetime.timedelta(days=7)
    while time.year < 2021:
        yield time
        time += dt


def get_time_s2s_calibration():
    times_file = pathlib.Path(__file__).parent / "times" / "calibration.txt"
    with times_file.open() as f:
        for line in f:
            line = line.strip()
            time = datetime.datetime.fromisoformat(line)
            yield time


get_times = {
    "2018": get_times_2018,
    "s2s_test": get_times_s2s_test,
    "s2s_calibration": get_time_s2s_calibration,
}


def get_time(times):
    if isinstance(times, list):
        times = [datetime.datetime.fromisoformat(s) for s in times]
    else:
        times = get_times[times]()
        # convert generator to list
        times = list(times)
    return times


def main(
    model: str,
    config: str,
    output: str,
):
    os.makedirs(output, exist_ok=True)

    with open(config) as f:
        protocol_config = json.load(f)

    config = {"protocol": protocol_config, "model": model}
    times = get_time(protocol_config["times"])
    config["protocol"]["times"] = [time.isoformat() for time in times]

    config_path = os.path.join(output, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    typer.run(main)
