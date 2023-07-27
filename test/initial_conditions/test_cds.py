from earth2mip.initial_conditions import get
from earth2mip import schema
import datetime

import pytest


@pytest.mark.slow
@pytest.mark.xfail
def test_get():
    # uses I/O and old ICs are not available forever.
    time = datetime.datetime(2023, 3, 10, 0)
    dataset = get(
        0, time, schema.ChannelSet.var34, source=schema.InitialConditionSource.cds
    )

    # check dims
    correct_dims = {"time": 1, "channel": 34, "lat": 721, "lon": 1440}
    assert dataset.dims == tuple(correct_dims.keys())
    assert dataset.shape == tuple(correct_dims.values())
