# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128
from typing import List
from earth2mip.initial_conditions import cds
import numpy as np
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from graphcast.graphcast import TaskConfig
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


def is_3d(name):
    return name in pl_inputs


def get_codes(variables: List[str], levels: List[int], time_levels: List[int]):
    lookup_code = cds.keys_to_vals(CODE_TO_GRAPHCAST_NAME)
    output = []
    for v in sorted(variables):
        if v in time_dependent:
            for history in time_levels:
                output.append((history, v))
        elif v in static_inputs:
            output.append(v)
        elif v in lookup_code:
            code = lookup_code[v]
            if v in pl_inputs:
                for history in time_levels:
                    for level in levels:
                        output.append(
                            (history, cds.PressureLevelCode(code, level=level))
                        )
            else:
                for history in time_levels:
                    output.append((history, cds.SingleLevelCode(code)))
        else:
            raise NotImplementedError(v)
    return output


def toa_incident_solar_radiation(time, lat, lon):
    # TODO validate this code against the ECWMF data
    solar_constant = 1361  #  W/m²
    z = cos_zenith_angle(time, lon, lat)
    return np.maximum(0, z) * solar_constant


def get_data_for_code_scalar(code, scalar):
    match code:
        case _, cds.PressureLevelCode(id, level):
            arr = scalar[CODE_TO_GRAPHCAST_NAME[id]].sel(level=level).values
        case _, cds.SingleLevelCode(id):
            arr = scalar[CODE_TO_GRAPHCAST_NAME[id]].values
        case "land_sea_mask":
            arr = scalar[code].values
        case "geopotential_at_surface":
            arr = scalar[code].values
        case _, str(s):
            arr = scalar[s].values
    return arr


def get_codes_from_task_config(task_config: TaskConfig):
    x_codes = get_codes(
        task_config.input_variables,
        levels=task_config.pressure_levels,
        time_levels=[0, 1],
    )
    f_codes = get_codes(
        task_config.forcing_variables,
        levels=task_config.pressure_levels,
        time_levels=[2],
    )
    t_codes = get_codes(
        task_config.target_variables,
        levels=task_config.pressure_levels,
        time_levels=[0],
    )
    return x_codes + f_codes, t_codes


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
