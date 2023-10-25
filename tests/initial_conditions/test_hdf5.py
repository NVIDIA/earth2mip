import pathlib
import datetime
import json
from earth2mip.initial_conditions import hdf5
from earth2mip import grid
import h5py
import pytest


@pytest.mark.parametrize("year", [2018, 1980, 2017])
def test__get_path(year):
    root = pathlib.Path(__file__).parent / "mock_data"
    root = root.as_posix()
    dt = datetime.datetime(year, 1, 2)
    path = hdf5._get_path(root, dt)
    assert pathlib.Path(path).name == dt.strftime("%Y.h5")
    assert pathlib.Path(path).exists()


def test__get_path_key_error():

    with pytest.raises(KeyError):
        hdf5._get_path(".", datetime.datetime(2040, 1, 2))


def test_hdf_data_source(tmp_path: pathlib.Path):
    h5_var_name = "fields"

    data_json = {
        "attrs": {},
        "coords": {"lat": [0, 1], "lon": [0, 1], "channel": ["a", "b"]},
        "dims": ["time", "channel", "lat", "lon"],
        "h5_path": h5_var_name,
    }

    data_json_path = tmp_path / "data.json"
    data_json_path.write_text(json.dumps(data_json))

    h5_path = tmp_path / "validation" / "2018.h5"
    h5_path.parent.mkdir()
    with h5py.File(h5_path.as_posix(), mode="w") as f:
        f.create_dataset(h5_var_name, shape=(10, 2, 2, 2), dtype="<f")

    ds = hdf5.DataSource.from_path(tmp_path.as_posix())
    time = datetime.datetime(2018, 1, 1)
    array = ds[time]
    assert array.shape == (2, 2, 2)
    assert isinstance(ds.grid, grid.LatLonGrid)
