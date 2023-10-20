import torch
from typing import Literal, Optional, List
from modulus.distributed import DistributedManager
from earth2mip.schema import Grid
from earth2mip.model_registry import Package
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

    def __init__(self, diagnostics: list[DiagnosticBase], axis: int = 1):
        super().__init__()
        assert all(
            diag.in_grid == diagnostics[0].in_grid for diag in diagnostics
        ), "Grid mismatch in concat function"
        assert all(
            diag.out_grid == diagnostics[0].out_grid for diag in diagnostics
        ), "Grid mismathch in concat function"
        self._in_grid = diagnostics[0].in_grid
        self._out_grid = diagnostics[0].out_grid

        self.axis = axis

        self.diagnostics = diagnostics
        self._in_channels = []
        self._out_channels = []
        for diag in self.diagnostics:
            self._in_channels = self._in_channels + diag.in_channels
            self._out_channels = self._out_channels + diag.out_channels
        # For input make sure we only get the unique
        # WARNING: No garentees on ordering
        self._in_channels = list(set(self._in_channels))

        self.filters = []
        for diagnostic in self.diagnostics:
            self.filters.append(
                Filter.load_diagnostic(
                    None, self._in_channels, diagnostic.in_channels, diagnostic.in_grid
                )
            )

    @property
    def in_channels(self) -> list[str]:
        return self._in_channels

    @property
    def out_channels(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> Grid:
        return self._in_grid

    @property
    def out_grid(self) -> Grid:
        return self._out_grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for dfilter, diagnostic in zip(self.filters, self.diagnostics):
            outputs.append(diagnostic(dfilter(x)))
        return torch.cat(outputs, dim=self.axis)

    @classmethod
    def load_diagnostic(
        cls,
        package: Optional[Package],
        diagnostics: list[DiagnosticBase],
        axis: int = 1,
        device: str = "cuda:0",
    ):
        return cls(diagnostics, axis).to(device)

    @classmethod
    def load_config_type(cls):
        return ConcatConfig


class ConcatConfig(DiagnosticConfigBase):

    type: Literal["Concat"] = "Concat"
    diagnostics: list[DiagnosticConfigBase]
    axis: int = 1

    def initialize(self):
        dm = DistributedManager()
        package = Concat.load_package()
        return Concat.load_diagnostic(package, self.diagnostics, self.axis, device=dm.device)
