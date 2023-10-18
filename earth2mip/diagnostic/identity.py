import torch
from typing import Literal
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase


class Identity(DiagnosticBase):
    """Idenity function. You probably don't need to use this unless you know what you
    are doing. Primarly used by the factory.
    """

    def __init__(self, in_channels: str, grid: Grid):
        super().__init__()
        self.grid = grid
        self.channels = in_channels

    @property
    def in_channels(self) -> list[str]:
        return self.channels

    @property
    def out_channels(self) -> list[str]:
        return self.channels

    @property
    def in_grid(self) -> Grid:
        return self.grid

    @property
    def out_grid(self) -> Grid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def load_diagnostic(cls, in_channels: list[str], grid: Grid):
        return cls(in_channels, grid)


class IdentityConfig(DiagnosticConfigBase):

    type: Literal["Identity"] = "Identity"
    in_channels: list[str]
    grid: Grid = Grid.grid_721x1440

    def initialize(self):
        return WindSpeed.load_diagnostic(self.in_channels, self.grid)
