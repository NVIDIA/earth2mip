from earth2mip.networks.graphcast.channels import yield_channels, pl_inputs, levels, unpack, pack
from earth2mip.initial_conditions import cds
import numpy as np


def test_unpack():
    levels = [10, 100]
    t2m = cds.parse_channel("t2m")
    t10 = cds.parse_channel("t10")
    t100 = cds.parse_channel("t100")

    codes = [t2m, t10, t100]

    # b, t, c, y, x
    x = np.random.uniform(size=[1, 1, len(codes), 2, 3])
    y = unpack(x, codes, levels=levels, pressure_level_codes=[t100.id], single_level_codes=[t2m.id])
    assert y["2m_temperature"].shape == (1, 1, 2, 3)
    assert y["temperature"].shape == (1, 1, len(levels), 2, 3)

    np.testing.assert_array_equal(x, pack(y, codes))
