# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable


class LexiconType(type):
    def __getitem__(cls, val: str) -> tuple[str, Callable]:
        return cls.get_item(val)  # type: ignore[attr-defined]


E2MIP_VOCAB = {
    "u10m": "u-component of wind at 10 m",
    "v10m": "v-component of wind at 10 m",
    "u100m": "u-component of wind at 100 m",
    "v100m": "v-component of wind at 100 m",
    "t2m": "temperature at 2m",
    "sp": "surface pressure",
    "msl": "mean sea level pressure",
    "tcwv": "total column water vapor",
    "tp": "total precipitation",
    "fg10m": "maximum 10 m wind gust since previous post-processing",
    "u50": "u-component of wind at 50 hPa",
    "u100": "",
    "u150": "",
    "u200": "",
    "u250": "",
    "u300": "",
    "u400": "",
    "u500": "",
    "u600": "",
    "u700": "",
    "u850": "",
    "u925": "",
    "u1000": "",
    "v50": "",
    "v100": "",
    "v150": "",
    "v200": "",
    "v250": "",
    "v300": "",
    "v400": "",
    "v500": "",
    "v600": "",
    "v700": "",
    "v850": "",
    "v925": "",
    "v1000": "",
    "z50": "",
    "z100": "",
    "z150": "",
    "z200": "",
    "z250": "",
    "z300": "",
    "z400": "",
    "z500": "",
    "z600": "",
    "z700": "",
    "z850": "",
    "z925": "",
    "z1000": "",
    "t50": "",
    "t100": "",
    "t150": "",
    "t200": "",
    "t250": "",
    "t300": "",
    "t400": "",
    "t500": "",
    "t600": "",
    "t700": "",
    "t850": "",
    "t925": "",
    "t1000": "",
    "r50": "relative humidity at 50 hPa",
    "r100": "relative humidity at 100 hPa",
    "r150": "relative humidity at 150 hPa",
    "r200": "relative humidity at 200 hPa",
    "r250": "relative humidity at 250 hPa",
    "r300": "relative humidity at 300 hPa",
    "r400": "relative humidity at 400 hPa",
    "r500": "relative humidity at 500 hPa",
    "r600": "relative humidity at 600 hPa",
    "r700": "relative humidity at 700 hPa",
    "r850": "relative humidity at 850 hPa",
    "r925": "relative humidity at 925 hPa",
    "r1000": "relative humidity at 1000 hPa",
    "q50": "specific humidity at 50 hPa",
    "q100": "specific humidity at 100 hPa",
    "q150": "specific humidity at 150 hPa",
    "q200": "specific humidity at 200 hPa",
    "q250": "specific humidity at 250 hPa",
    "q300": "specific humidity at 300 hPa",
    "q400": "specific humidity at 400 hPa",
    "q500": "specific humidity at 500 hPa",
    "q600": "specific humidity at 600 hPa",
    "q700": "specific humidity at 700 hPa",
    "q850": "specific humidity at 850 hPa",
    "q925": "specific humidity at 925 hPa",
    "q1000": "specific humidity at 1000 hPa",
}
