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
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class InitialConditionSource(Enum):
    ifs: str = "ifs"
    era5: str = "era5"
    cds: str = "cds"
    gfs: str = "gfs"
    hrmip: str = "hrmip"


# https://docs.pydantic.dev/usage/types/#discriminated-unions-aka-tagged-unions
class WeatherEventProperties(BaseModel):
    """
    Attributes:
        netcdf: load the initial conditions from this path if given

    """

    name: str
    start_time: Optional[datetime.datetime]
    initial_condition_source: InitialConditionSource = InitialConditionSource.era5
    netcdf: str = ""
    # TODO do not require IC other than restart (currently loads data w/ normal mechanisms regardless) # noqa
    restart: str = ""


class Diagnostic(BaseModel):
    type: str
    function: str = ""
    channels: List[str]
    nbins: int = 10


class Window(BaseModel):
    type: Literal["Window"] = "Window"
    name: str
    lat_min: float = -90
    lat_max: float = 90
    lon_min: float = 0
    lon_max: float = 360
    diagnostics: List[Diagnostic]


class CWBDomain(BaseModel):
    type: Literal["CWBDomain"]
    name: str
    path: str = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
    diagnostics: List[Diagnostic]


class MultiPoint(BaseModel):
    type: Literal["MultiPoint"]
    name: str
    lat: List[float]
    lon: List[float]
    diagnostics: List[Diagnostic]


Domain = Union[Window, CWBDomain, MultiPoint]


class WeatherEvent(BaseModel):
    properties: WeatherEventProperties
    domains: List[Domain]


def _read():
    with open("weather_events.json") as f:
        return json.load(f)


def list_():
    events = _read()
    return list(events)


def read(forecast_name: str) -> WeatherEvent:
    weather_events = _read()
    weather_event = weather_events[forecast_name]

    for domain in weather_event["domains"]:
        if domain["name"] == "global":
            domain["type"] = "Window"
            domain["lat_min"] = -90
            domain["lat_max"] = 90
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "northern_hemisphere":
            domain["lat_min"] = 0
            domain["lat_max"] = 90
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "southern_hemisphere":
            domain["lat_min"] = -90
            domain["lat_max"] = 0
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "CWBDomain":
            if len(domain["diagnostics"]) > 1:
                print("CWBDomain only supports one diagnostic")
                domain["diagnostics"] = domain["diagnostics"][0]
    event = WeatherEvent.parse_obj(weather_event)
    return event
