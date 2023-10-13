from earth2mip import initial_conditions, schema
from earth2mip.initial_conditions.base import DataSource
from earth2mip import config
import pytest


@pytest.mark.parametrize("source", list(schema.InitialConditionSource))
def test_get_data_source(
    source: schema.InitialConditionSource, monkeypatch: pytest.MonkeyPatch
):

    if (
        source
        in [schema.InitialConditionSource.era5, schema.InitialConditionSource.hrmip]
        and not config.ERA5_HDF5
    ):
        pytest.skip(f"Need HDF5 data to test {source}")

    ds = initial_conditions.get_data_source(
        n_history=0,
        channel_names=["t850", "t2m"],
        grid=schema.Grid.grid_721x1440,
        initial_condition_source=source,
    )
    assert isinstance(ds, DataSource)
