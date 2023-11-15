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

import numpy as np
import pytz


def convert_to_datetime(time) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(np.datetime_as_string(time, "s"))
    return dt


def datetime_to_timestamp(time: datetime.datetime) -> float:
    if time.tzinfo is None:
        time = time.replace(tzinfo=pytz.utc)
    return time.timestamp()
