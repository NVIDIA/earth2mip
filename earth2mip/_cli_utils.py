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
import argparse
import datetime
from typing import List

from earth2mip import networks, schema


class TimeRange:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--start-time", default="2018-01-01", type=datetime.datetime.fromisoformat
        )
        parser.add_argument(
            "--time-step",
            default="12",
            type=lambda h: datetime.timedelta(hours=int(h)),
            help="time step in hours between times.",
        )
        parser.add_argument(
            "--end-time",
            default="2018-12-01",
            help="final time (inclusive).",
            type=datetime.datetime.fromisoformat,
        )

    @staticmethod
    def from_args(args) -> List[datetime.datetime]:
        """parse the command line arguments and return a TimeRange"""
        return get_times(args.start_time, args.end_time, args.time_step)


def get_times(start_time: datetime, end_time: datetime, step: datetime.timedelta):
    # the IFS data Jaideep downloaded only has 668 steps (up to end of november 2018)
    times = []
    time = start_time
    while time <= end_time:
        times.append(time)
        time += step
    return times


def add_model_args(parser: argparse.ArgumentParser, required=False):
    if required:
        parser.add_argument("model", type=str)
    else:
        parser.add_argument("--model", type=str)
    parser.add_argument(
        "--model-metadata",
        type=str,
        help="metadata.json file. Defaults to the metadata.json in the model package.",
        default="",
    )


def model_from_args(args, device):
    if args.model_metadata:
        with open(args.model_metadata) as f:
            metadata = schema.Model.parse_raw(f.read())
    else:
        metadata = None

    return networks.get_model(args.model, device=device, metadata=metadata)
