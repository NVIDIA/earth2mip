# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128
from earth2mip.initial_conditions import cds

sl_inputs = {
     "2m_temperature": 167 ,
    "mean_sea_level_pressure": 151,
    "10m_v_component_of_wind": 166,
    "10m_u_component_of_wind": 165,
    "total_precipitation_6hr": 260267,
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
    "toa_incident_solar_radiation": 212,
    "year_progress_sin": None,
    "year_progress_cos": None,
    "day_progress_sin": None,
    "day_progress_cos": None
}


channels = []
cds.CHANNEL_TO_CODE['tp6'] = sl_inputs['total_precipitation_6hr']
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

for v, code, level in yield_channels_ecmwf():
    print(v, code, level)
