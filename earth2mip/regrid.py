import netCDF4 as nc
import einops
import torch
import pathlib
from earth2mip.schema import Grid
from earth2mip._config import Settings


class TempestRegridder(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        dataset = nc.Dataset(file_path)
        self.lat = dataset["latc_b"][:]
        self.lon = dataset["lonc_b"][:]

        i = dataset["row"][:] - 1
        j = dataset["col"][:] - 1
        M = dataset["S"][:]

        i = i.data
        j = j.data
        M = M.data

        self.M = torch.sparse_coo_tensor((i, j), M, [max(i) + 1, max(j) + 1]).float()

    def to(self, device):
        self.M = self.M.to(device)
        return self

    def forward(self, x):
        xr = einops.rearrange(x, "b c x y -> b c (x y)")
        yr = xr @ self.M.T
        y = einops.rearrange(
            yr, "b c (x y) -> b c x y", x=self.lat.size, y=self.lon.size
        )
        return y


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _get_tempest_regridder(src: Grid, dest: Grid) -> TempestRegridder:
    # TODO map data needs to be available for S2S scoring
    config = Settings()

    # TODO add instructions for how to make the tempest map file
    map_file = (
        pathlib.Path(config.MAP_FILES) / src.value / dest.value / "tempest_map.nc"
    )
    return TempestRegridder(map_file.as_posix())


def get_regridder(src: Grid, dest: Grid):
    if src == dest:
        return Identity()
    else:
        return _get_tempest_regridder(src, dest)
    raise NotImplementedError()
