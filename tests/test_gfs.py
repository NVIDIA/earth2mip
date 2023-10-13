# coding: utf-8
from earth2mip.initial_conditions import gfs
import datetime
import pytest


@pytest.mark.slow
def test_gfs():
    t = datetime.datetime.today() - datetime.timedelta(days=1)
    ds = gfs.DataSource(["t850"])
    out = ds[t]
    assert out.shape == (1, 1, 721, 1440)
