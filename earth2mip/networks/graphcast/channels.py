# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128
from typing import List
from earth2mip.initial_conditions import cds
import numpy as np
import dataclasses
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from graphcast.graphcast import TaskConfig
import pandas as pd
import graphcast.data_utils
import pandas
import xarray


CODE_TO_GRAPHCAST_NAME = {
    167: "2m_temperature",
    151: "mean_sea_level_pressure",
    166: "10m_v_component_of_wind",
    165: "10m_u_component_of_wind",
    260267: "total_precipitation_6hr",
    212: "toa_incident_solar_radiation",
    130: "temperature",
    129: "geopotential",
    131: "u_component_of_wind",
    132: "v_component_of_wind",
    135: "vertical_velocity",
    133: "specific_humidity",
    162051: "geopotential_at_surface",
    172: "land_sea_mask",
}

sl_inputs = {
    "2m_temperature": 167,
    "mean_sea_level_pressure": 151,
    "10m_v_component_of_wind": 166,
    "10m_u_component_of_wind": 165,
    "toa_incident_solar_radiation": 212,
}

pl_inputs = {
    "temperature": 130,
    "geopotential": 129,
    "u_component_of_wind": 131,
    "v_component_of_wind": 132,
    "vertical_velocity": 135,
    "specific_humidity": 133,
}

static_inputs = {
    "geopotential_at_surface": 162051,
    "land_sea_mask": "172",
}

levels = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

time_dependent = {
    "toa_incident_solar_radiation": None,
    "year_progress_sin": None,
    "year_progress_cos": None,
    "day_progress_sin": None,
    "day_progress_cos": None,
}


channels = []
code_to_channel = dict(zip(cds.CHANNEL_TO_CODE.values(), cds.CHANNEL_TO_CODE.keys()))


def yield_channels():
    for v in pl_inputs:
        code = pl_inputs[v]
        for level in levels:
            yield code_to_channel[code] + str(level)

    for v in sl_inputs:
        code = sl_inputs[v]
        yield code_to_channel[code]


def yield_channels_ecmwf():
    for v in pl_inputs:
        code = pl_inputs[v]
        for level in levels:
            yield code_to_channel[code], code, level

    for v in sl_inputs:
        code = sl_inputs[v]
        yield code_to_channel[code], code, None


def unpack(x, codes, pressure_level_codes, single_level_codes, levels):
    """retrieve variable from x

    Args:
        x: array (b, h, c, y, x)
        channels: names of c dim
        levels: levels to get
        pressure_level_codes_to_get
        single_level_codes_to_get
    """
    idx = pandas.Index(codes)
    output = xarray.Dataset()
    for code in pressure_level_codes:
        name = CODE_TO_GRAPHCAST_NAME[code]
        indexer = idx.get_indexer(
            [cds.PressureLevelCode(code, level) for level in levels]
        )
        output[name] = ["batch", "time", "level", "lat", "lon"], x[:, :, indexer]
    for code in single_level_codes:
        name = CODE_TO_GRAPHCAST_NAME[code]
        indexer = codes.index(cds.SingleLevelCode(code))
        output[name] = ["batch", "time", "lat", "lon"], x[:, :, indexer]
    return output.assign_coords(level=levels)


def unpack_all(x, codes):

    p_codes = set()
    s_codes = set()
    levels = set()

    for c in codes:
        if isinstance(c, cds.SingleLevelCode):
            s_codes.add(c.id)
        elif isinstance(c, cds.PressureLevelCode):
            p_codes.add(c.id)
            levels.add(c.level)

    return unpack(x, codes, sorted(p_codes), sorted(s_codes), levels=sorted(levels))


def assign_grid_coords(ds, grid):
    coords = {
        "lat": grid.lat[::-1],
        "lon": grid.lon,
    }
    return ds.assign_coords(coords)


def pack(ds: xarray.Dataset, codes) -> np.ndarray:
    """retrieve variable from x

    Args:
        x: array (b, h, c, y, x)
        channels: names of c dim
        levels: levels to get
        pressure_level_codes_to_get
        single_level_codes_to_get
    """
    idx = pandas.Index(codes)
    shape = (1, ds.sizes["time"], len(codes), ds.sizes["lat"], ds.sizes["lon"])
    x = np.zeros(shape=shape)
    for k, code in enumerate(codes):
        name = CODE_TO_GRAPHCAST_NAME[code.id]
        if isinstance(code, cds.SingleLevelCode):
            x[:, :, k] = ds[name]
        elif isinstance(code, cds.PressureLevelCode):
            x[:, :, k] = ds[name].sel(level=code.level)
    return x


def get_codes(variables: List[str], levels: List[int], n_history=2):
    lookup_code = cds.keys_to_vals(CODE_TO_GRAPHCAST_NAME)
    output = []
    for v in sorted(variables):
        if v in time_dependent:
            for history in range(n_history):
                output.append((history, v))
        elif v in static_inputs:
            output.append(v)
        elif v in lookup_code:
            code = lookup_code[v]
            if v in pl_inputs:
                for history in range(n_history):
                    for level in levels:
                        output.append(
                            (history, cds.PressureLevelCode(code, level=level))
                        )
            else:
                for history in range(n_history):
                    output.append((history, cds.SingleLevelCode(code)))
        else:
            raise NotImplementedError(v)
    return output


def toa_incident_solar_radiation(time, lat, lon):
    # TODO validate this code against the ECWMF data
    solar_constant = 1361  #  W/m²
    z = cos_zenith_angle(time, lon, lat)
    return np.maximum(0, z) * solar_constant


def add_dynamic_vars(ds, time):
    if "batch" in ds.dims:
        assert ds.sizes["batch"] == 1, "haven't vectorized cos zenith yet over time"
    ds.coords["init_time"] = xarray.Variable(["batch"], [pd.Timestamp(time)])
    ds.coords["datetime"] = ds.init_time + ds.time
    graphcast.data_utils.add_derived_vars(ds)
    del ds["year_progress"]
    del ds["day_progress"]

    toas = []
    for dt in ds.time.values.flat:
        tt = pd.Timestamp(time) - pd.Timedelta(dt)
        datetime = tt.to_pydatetime()
        toas.append(toa_incident_solar_radiation(time, ds.lat, ds.lon))
    ds["toa_incident_solar_radiation"] = xarray.concat(toas, dim="time").expand_dims(
        "batch"
    )
    del ds.coords["datetime"]
    del ds.coords["init_time"]


if __name__ == "__main__":
    from earth2mip.initial_conditions import cds
    import datetime
    import xarray

    channels = list(yield_channels())
    client = cds.Client()
    ds = cds.DataSource(channels, client=client)
    time = datetime.datetime(2018, 1, 1)

    try:
        ds = xarray.open_dataset("output.nc")
    except FileNotFoundError:
        ds = ds[time]
        ds.rename("fields").to_netcdf("output.nc")
