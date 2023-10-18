import torch
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase

class WindSpeed(DiagnosticBase):
    """Computes the wind speed at a given level

    Example
    -------
    >>> windspeed = WindSpeed('10m', Grid.grid_721x1440)
    >>> x = np.randn(1, 2, 721, 1440)
    >>> out = windspeed(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """
    def __init__(self, level:str, grid:Grid):
        super().__init__()
        self.grid = grid

        self._in_channels = [f'u{level}', f'v{level}']
        self._out_channels = [f'ws{level}']

    @property
    def in_channels(self) -> list[str]:
        return self._in_channels

    @property
    def out_channels(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> Grid:
        return self.grid

    @property
    def out_grid(self) -> Grid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x[:,0:1,...]**2 + x[:,1:2,...]**2)

    @classmethod
    def load_diagnostic(cls, level:str, grid:Grid):
        return cls(level, grid)