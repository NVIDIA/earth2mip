from earth2mip.networks.graphcast import channels
from earth2mip.initial_conditions.cds import (
    parse_channel,
    _parse_files,
    _get_cds_requests,
)
import glob
import datetime

channels = list(channels.yield_channels())
codes = [parse_channel(c) for c in channels]
for n, req in _get_cds_requests(
    codes, time=datetime.datetime(2018, 1, 1), format="grib"
):
    print(req)
files = glob.glob("*.grib")
_parse_files(codes, files)
