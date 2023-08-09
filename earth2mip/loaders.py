import torch
from typing import Protocol


class LoaderProtocol(Protocol):
    def __call__(self, package, pretrained=True) -> None:
        return


def pickle(package, pretrained=True):
    """
    load a checkpoint into a model
    """
    assert pretrained
    p = package.get("weights.tar")
    return torch.load(p)


def torchscript(package, pretrained=True):
    """
    load a checkpoint into a model
    """
    p = package.get("scripted_model.pt")
    import json

    config = package.get("config.json")
    with open(config) as f:
        config = json.load(f)

    model = torch.jit.load(p)

    if config["add_zenith"]:
        from earth2mip.networks import CosZenWrapper
        import numpy as np

        lat = 90 - np.arange(721) * 0.25
        lon = np.arange(1440) * 0.25
        model = CosZenWrapper(model, lon, lat)

    return model
