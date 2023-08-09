# import this before torch to avoid GLIBC error
import xarray
import os

import pytest
from earth2mip import config
import torch
import os


import torch

def get_gpu_count():
    return torch.cuda.device_count()


@pytest.fixture()
def has_registry():
    if not config.MODEL_REGISTRY:
        pytest.skip("MODEL_REGISTRY not configured.")


@pytest.fixture()
def dist():
    from modulus.distributed.manager import DistributedManager
    DistributedManager.initialize()
    return DistributedManager()

@pytest.fixture()
def ngpu():
    return get_gpu_count()
