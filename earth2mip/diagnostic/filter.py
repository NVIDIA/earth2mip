import torch
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase

class Filter(DiagnosticBase):
    """Simply filters channels of a given input array.

    Note
    ----
    If this errors its likely because you have a mismatch of channels in the Diagnostic
    factory class.

    Example
    -------
    >>> dfilter = Filter(['u10m','t2m'], ['u10m'], Grid.grid_721x1440)
    >>> x = np.randn(1, 2, 721, 1440)
    >>> out = dfilter(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """
    def __init__(self, in_channels:str, out_channels:str, grid:Grid):
        super().__init__()
        self.grid = grid

        self._in_channels = in_channels
        self._out_channels = out_channels

        self.indexes = []
        try:
            for channel in self._out_channels:
                self.indexes.append(self._in_channels.index(channel))
            self.indexes = torch.IntTensor(self.indexes)
        except ValueError as e:
            raise ValueError(f"Looks like theres a mismatch between input and " +
                "requested channels. {e}")

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
    def load_diagnostic(cls, in_channels:str, out_channels:str, grid:Grid):
        return cls(in_channels, out_channels, grid)