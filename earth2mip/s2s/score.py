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

import altair as alt
import numpy as np
import pandas as pd
import xarray
import xskillscore


def score(terciles: xarray.Dataset, truth: xarray.Dataset) -> pd.DataFrame:
    """area-weighted RPSS scores

    Regions: global, northern hemisphere, and southern hemisphere

    Args:
        terciles: predicted terciles. must have `longitude`, `latitude`, `forecast_time` coordinates/dims.
        truth: true terciles. same format as terciles

    Returns:
        dataframe with scores for different regions and climatology. Example::

                lead_time forecast_time       t2m valid_time  week source         region        tp
            0        21.0    2018-01-02  1.455134 2018-01-23   1.0   sfno         Global       NaN
            1        35.0    2018-01-02  1.357457 2018-02-06   1.0   sfno         Global       NaN
            2        21.0    2018-01-02  1.308716 2018-01-23   NaN   clim         Global  1.310107
            3        35.0    2018-01-02  1.306281 2018-02-06   NaN   clim         Global  1.312259
            4        21.0    2018-01-02  1.331612 2018-01-23   1.0   sfno  Northern Hem.       NaN
            5        35.0    2018-01-02  1.211101 2018-02-06   1.0   sfno  Northern Hem.       NaN
            6        21.0    2018-01-02  1.184829 2018-01-23   NaN   clim  Northern Hem.  1.237482
            7        35.0    2018-01-02  1.180459 2018-02-06   NaN   clim  Northern Hem.  1.241959
            8        21.0    2018-01-02  1.575785 2018-01-23   1.0   sfno  Southern Hem.       NaN
            9        35.0    2018-01-02  1.497714 2018-02-06   1.0   sfno  Southern Hem.       NaN
            10       21.0    2018-01-02  1.431765 2018-01-23   NaN   clim  Southern Hem.  1.381933
            11       35.0    2018-01-02  1.430871 2018-02-06   NaN   clim  Southern Hem.  1.381818

    """  # noqa

    sfno = xarray.open_dataset(terciles)
    obs = xarray.open_dataset(truth)

    sfno, obs = xarray.align(sfno, obs)
    assert obs.sizes["forecast_time"] > 0  # noqa

    clim = xarray.ones_like(obs) / 3.0

    # %%
    masks = {}
    cos_lat = np.cos(np.deg2rad(obs.latitude))
    masks["Global"] = cos_lat
    masks["Northern Hem."] = cos_lat.where(obs.latitude > 0, 0.0)
    masks["Southern Hem."] = cos_lat.where(obs.latitude < 0, 0.0)

    scores = []
    for mask in masks:
        cos_lat = masks[mask]

        iscores = {}
        iscores["sfno"] = xskillscore.rps(
            sfno,
            obs,
            category_edges=None,
            input_distributions="p",
            dim=["latitude", "longitude"],
            weights=cos_lat,
        )
        iscores["clim"] = xskillscore.rps(
            clim,
            obs,
            category_edges=None,
            input_distributions="p",
            dim=["latitude", "longitude"],
            weights=cos_lat,
        )

        for key in iscores:
            v = iscores[key]
            scores.append(v.assign(source=key, region=mask))

    df = pd.concat([d.to_dataframe() for d in scores])
    df = df.reset_index()
    df["lead_time"] = df.lead_time + pd.Timedelta("7D")
    df["valid_time"] = df.lead_time + df.forecast_time
    df["lead_time"] = df.lead_time / pd.Timedelta("1D")
    return df


def plot(df, output, timeUnit="yearmonthdate"):
    import vl_convert

    alt.data_transformers.disable_max_rows()

    # https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
    wong_palette = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]

    # altair

    def my_theme():
        return {"config": {"range": {"category": wong_palette}}}

    alt.themes.register("my_theme", my_theme)
    alt.themes.enable("my_theme")
    c = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(field="valid_time", timeUnit=timeUnit),
            y=alt.Y("mean(t2m)", title="RPSS T2M"),
            column="region",
            color="source",
            row=alt.Row("lead_time", type="ordinal", title="Lead Time (days)"),
        )
        .properties(width=200, height=200 / 1.61)
    )

    png_data = vl_convert.vegalite_to_png(vl_spec=c.to_json(), scale=3)
    with open(output, "wb") as f:
        f.write(png_data)
