from typing import List
import numpy as np
import dataclasses
from earth2mip import schema


@dataclasses.dataclass(frozen=True)
class LatLonGrid:

    lat: List[float]
    lon: List[float]

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))


def regular_lat_lon_grid(
    nlat: int, nlon: int, includes_south_pole: bool = True
) -> LatLonGrid:
    """A regular lat-lon grid

    Lat is ordered from 90 to -90. Includes -90 and only if if includes_south_pole is True.
    Lon is ordered from 0 to 360. includes 0, but not 360.

    """  # noqa
    lat = np.linspace(90, -90, nlat, endpoint=includes_south_pole)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return LatLonGrid(lat.tolist(), lon.tolist())


def from_enum(grid_enum: schema.Grid) -> LatLonGrid:
    if grid_enum == schema.Grid.grid_720x1440:
        return regular_lat_lon_grid(720, 1440, includes_south_pole=False)
    elif grid_enum == schema.Grid.grid_721x1440:
        return regular_lat_lon_grid(721, 1440)
    elif grid_enum == schema.Grid.s2s_challenge:
        return regular_lat_lon_grid(181, 360)
    else:
        raise ValueError(f"Unknown grid {grid_enum}")
