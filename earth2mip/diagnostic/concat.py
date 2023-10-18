import torch
from typing import Literal
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase
from earth2mip.diagnostic.filter import Filter


class Concat(DiagnosticBase):
    """Provide a pimative that concats the fields of two diagnostic functions.
    This is primarly implemented for limited use with configs. If you are doing
    something complicated, odds are you should be using Python APIs to manually
    implement stuff.

    This only accepts essential fields of the two diagnostics.

    TODO: Concept atm

    Example
    -------
    >>> dfilter = Filter(['u10m','t2m'], ['u10m'], Grid.grid_721x1440)
    >>> x = torch.randn(1, 2, 721, 1440)
    >>> out = dfilter(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """

    def __init__(
        self, diagnostic_1: DiagnosticBase, diagnostic_2: DiagnosticBase, axis: int = 1
    ):
        super().__init__()
        assert (
            diagnostic_1.out_grid == diagnostic_2.out_grid
        ), "Grid mismathch in concat function"
        assert (
            diagnostic_1.in_grid == diagnostic_2.in_grid
        ), "Grid mismathch in concat function"
        self.in_grid = diagnostic_1.in_grid
        self.out_grid = diagnostic_1.out_grid

        self.axis = axis

        self.diagnostic_1 = diagnostic_1
        self.diagnostic_2 = diagnostic_2
        self._in_channels = list(
            set(diagnostic_1.in_channels + diagnostic_2.in_channels)
        )
        self._out_channels = diagnostic_1.out_channels + diagnostic_2.out_channels
        self.filter_1 = Filter.load_diagnostic(
            self._in_channels, diagnostic_1.in_channels, diagnostic_1.in_grid
        )
        self.filter_2 = Filter.load_diagnostic(
            self._in_channels, diagnostic_2.in_channels, diagnostic_2.in_grid
        )

    @property
    def in_channels(self) -> list[str]:
        return self._in_channels

    @property
    def out_channels(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> Grid:
        return self.in_grid

    @property
    def out_grid(self) -> Grid:
        return self.out_grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        output_1 = self.diagnostic_1(self.filter_1(x))
        output_2 = self.diagnostic_2(self.filter_2(x))
        return torch.cat([output_1, output_2], dim=self.axis)

    @classmethod
    def load_diagnostic(
        cls,
        diagnostic_1: DiagnosticBase,
        diagnostic_2: DiagnosticBase,
        axis: int = 1,
        device: str = "cuda:0",
    ):
        return cls(diagnostic_1, diagnostic_2, axis).to(device)


class FilterConfig(DiagnosticConfigBase):

    type: Literal["Concat"] = "Concat"
    diagnostic_1: DiagnosticBase
    diagnostic_2: DiagnosticBase
    axis: int = 1

    def initialize(self):
        dm = DistributedManager()
        return Concat.load_diagnostic(
            self.diagnostic_1, self.diagnostic_2, axis, device=dm.device
        )
