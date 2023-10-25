import pathlib
import datetime
import json
from earth2mip.initial_conditions.era5 import HDF5DataSource
import h5py


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

    ds = HDF5DataSource.from_path(tmp_path.as_posix())
    time = datetime.datetime(2018, 1, 1)
    array = ds[time]
    assert array.shape == (2, 2, 2)
    assert array.grid
