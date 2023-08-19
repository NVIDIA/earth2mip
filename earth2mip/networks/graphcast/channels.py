# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128
from earth2mip.initial_conditions import cds
import numpy as np
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
import graphcast.data_utils 
import pandas
import xarray

tisr =     "toa_incident_solar_radiation"

sl_inputs = {
     "2m_temperature": 167 ,
    "mean_sea_level_pressure": 151,
    "10m_v_component_of_wind": 166,
    "10m_u_component_of_wind": 165,
    "total_precipitation_6hr": 260267,
tisr: 212,
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
    "year_progress_sin": None,
    "year_progress_cos": None,
    "day_progress_sin": None,
    "day_progress_cos": None
}


channels = []
cds.CHANNEL_TO_CODE['tp6'] = sl_inputs['total_precipitation_6hr']
cds.CHANNEL_TO_CODE['tisr'] = sl_inputs['toa_incident_solar_radiation']
cds.CHANNEL_TO_CODE['zs'] = static_inputs['geopotential_at_surface']
cds.CHANNEL_TO_CODE['w'] = pl_inputs['vertical_velocity']

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

def keys_to_vals(d):
    return dict(zip(d.values(), d.keys()))


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
        name = keys_to_vals(pl_inputs)[code]
        indexer = idx.get_indexer([cds.PressureLevelCode(code, level) for level in levels])
        output[name] = ["batch", "time", "level", "lat", "lon"], x[:, :, indexer]
    for code in single_level_codes:
        name = keys_to_vals(sl_inputs)[code]
        indexer = codes.index(cds.SingleLevelCode(code))
        output[name] = ["batch", "time", "lat", "lon"], x[:, :, indexer]
    return output.assign_coords(level=levels)


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
    shape = (1, ds.sizes['time'], len(codes), ds.sizes['lat'], ds.sizes['lon'])
    x = np.zeros(shape=shape)
    pl_name_by_code = keys_to_vals(pl_inputs)
    sl_name_by_code = keys_to_vals(sl_inputs)
    for k, code in enumerate(codes):
        if isinstance(code, cds.SingleLevelCode):
            x[:, :, k] = ds[sl_name_by_code[code.id]]
        elif isinstance(code, cds.PressureLevelCode):
            x[:, :, k] = ds[pl_name_by_code[code.id]].sel(level=code.level)
    return x
    

def toa_incident_solar_radiation(time, lat, lon):
    solar_constant = 1361  #  W/m²
    z = cos_zenith_angle(time, lon, lat)
    return np.maximum(0, z) * solar_constant


def add_derived_vars(ds):
    graphcast.data_utils.add_derived_vars(ds)
    ds[tisr] = toa_incident_solar_radiation(ds.time, ds.lat, ds.lon)



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




