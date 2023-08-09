from earth2mip.networks import pangu
import datetime
import torch


class MockPangu(pangu.PanguWeather):
    def __init__(self):
        pass

    def __call__(self, pl, sl):
        return pl, sl


def test_pangu():
    model_6 = pangu.PanguStacked(MockPangu())
    model_24 = pangu.PanguStacked(MockPangu())
    inference = pangu.PanguInference(model_6, model_24)
    t0 = datetime.datetime(2018, 1, 1)
    dt = datetime.timedelta(hours=6)
    x = torch.ones((1, 1, len(inference.in_channel_names), 721, 1440))
    n = 5

    times = []
    for k, (time, y, _) in enumerate(inference(t0, x)):
        if k > n:
            break
        assert y.shape == x.shape[1:]
        assert torch.all(y == x[0])
        times.append(time)

    assert times == [t0 + k * dt for k in range(n + 1)]
