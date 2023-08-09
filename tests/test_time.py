from earth2mip.time import convert_to_datetime
import numpy as np
import datetime


def test_convert_to_datetime():
    time = np.datetime64("2021-01-01T00:00:00")
    expected = datetime.datetime(2021, 1, 1)
    assert convert_to_datetime(time) == expected
