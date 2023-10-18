import torch
from typing import Literal
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase


class Filter(DiagnosticBase):
    """Simply filters channels of a given input array. Probably shouldn't be using this
    manually unless you know what you are doing.

    Note
    ----
    If this errors its likely because you have a mismatch of channels

    Example
    -------
    >>> dfilter = Filter(['u10m','t2m'], ['u10m'], Grid.grid_721x1440)
    >>> x = torch.randn(1, 2, 721, 1440)
    >>> out = dfilter(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """

    def __init__(self, in_channels: list[str], out_channels: list[str], grid: Grid):
        super().__init__()
        self.grid = grid

        self._in_channels = in_channels
        self._out_channels = out_channels

        indexes_list = []
        try:
            for channel in self._out_channels:
                indexes_list.append(self._in_channels.index(channel))
            self.register_buffer("indexes", torch.IntTensor(indexes_list))
        except ValueError as e:
            raise ValueError(
                f"Looks like theres a mismatch between input and "
                + "requested channels. {e}"
            )

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
        print(x.device, self.indexes.device)
        return torch.index_select(x, 1, self.indexes)

    @classmethod
    def load_diagnostic(
        cls,
        in_channels: list[str],
        out_channels: list[str],
        grid: Grid,
        device: str = "cuda:0",
    ):
        return cls(in_channels, out_channels, grid).to(device)


class FilterConfig(DiagnosticConfigBase):

    type: Literal["Filter"] = "Filter"
    in_channels: list[str]
    out_channels: list[str]
    grid: Grid = Grid.grid_721x1440

    def initialize(self):
        dm = DistributedManager()
        return Filter.load_diagnostic(
            self.in_channels, self.out_channels, self.grid, device=dm.device
        )
