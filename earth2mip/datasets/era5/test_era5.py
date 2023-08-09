import pathlib
import h5py

from earth2mip.datasets import era5

import pytest


@pytest.mark.slow
def test_open_34_vars(tmp_path: pathlib.Path):
    path = tmp_path / "1979.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fields", shape=[1, 34, 721, 1440])

    ds = era5.open_34_vars(path)
    # ensure that data can be grabbed
    ds.mean().compute()

    assert set(ds.coords) == {"time", "channel", "lat", "lon"}


@pytest.mark.slow
def test_open_all_hdf5(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    path = folder / "1979.h5"

    shape = [1, 34, 721, 1440]
    with h5py.File(path, "w") as f:
        f.create_dataset("fields", shape=shape)

    with era5.open_all_hdf5(tmp_path.as_posix()) as array:
        assert array.shape == (1, *shape)
