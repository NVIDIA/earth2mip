import numpy as np
import properscoring
import torch

from earth2mip.crps import crps_from_empirical_cdf


def test_crps_cdf():
    n = 10
    x = torch.randn((10, n))
    y = torch.randn((n,))
    out = crps_from_empirical_cdf(y, x)
    reference = properscoring.crps_ensemble(y.numpy(), x.numpy(), axis=0)
    np.testing.assert_allclose(out.numpy(), reference, rtol=1e-6)
