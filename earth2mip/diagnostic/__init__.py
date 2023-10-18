import torch
from typing import Union
from earth2mip.schema import Grid
from earth2mip.geo_function import GeoFunction
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase
from earth2mip.diagnostic.identity import IdentityConfig, Identity
from earth2mip.diagnostic.filter import FilterConfig, Filter
from earth2mip.diagnostic.wind_speed import WindSpeedConfig
from earth2mip.diagnostic.wind_gust import WindGustConfig

DiagnosticTypes = Union[
    IdentityConfig,
    FilterConfig,
    WindSpeedConfig,
    WindGustConfig,
]


class Diagnostic(GeoFunction):
    """List of diagnostic functions that are executed in sequential order. This is
    useful for building compositions of functions to create complex outputs.

    TODO: Add concat feature

    Note
    ----
    Presently this executes and refines channels greedly. In other words only the output
    channel of a function is ever preserved. Expanding with concat functions is likely
    for the future.

    Args:
        in_channels (list[str]): Input channels
        in_grid (Grid): Input grid
    """

    def __init__(self, in_channels: list[str], in_grid: Grid, device="cuda:0"):
        self.diagnostics = [Identity(in_channels, in_grid)]
        self.device = device

    def add(self, diagfunc: DiagnosticBase):
        """Add

        Args:
            diagfunc (DiagnosticBase): Diagnostic function
        """
        # Create filter to transition between diagnostics
        dfilter = Filter.load_diagnostic(
            in_channels=self.diagnostics[-1].out_channels,
            out_channels=diagfunc.in_channels,
            grid=diagfunc.in_grid,
            device=self.device,
        )

        self.diagnostics.append(dfilter)
        self.diagnostics.append(diagfunc)

    def from_config(self, cfg: DiagnosticConfigBase):
        """Adds a diagnostic function from a config object

        Args:
            cfg (DiagnosticConfigBase): Diagnostic config
        """
        diagnostic = cfg.initialize()
        self.add(diagnostic)

    @property
    def in_channels(self) -> list[str]:
        if len(self.diagnostics) > 0:
            return self.diagnostics[0].in_channels

    @property
    def out_channels(self) -> list[str]:
        if len(self.diagnostics) > 0:
            return self.diagnostics[-1].out_channels

    @property
    def in_grid(self) -> Grid:
        if len(self.diagnostics) > 0:
            return self.diagnostics[0].in_grid

    @property
    def out_grid(self) -> Grid:
        if len(self.diagnostics) > 0:
            return self.diagnostics[-1].out_grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for diag in self.diagnostics:
            x = diag(x)
        return x
