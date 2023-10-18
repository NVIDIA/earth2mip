import torch
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase

class Identity(DiagnosticBase):
    """Idenity function. You probably don't need to use this unless you know what you
    are doing. Primarly used by the factory.
    """
    def __init__(self, channels:str, grid:Grid):
        super().__init__()
        self.grid = grid
        self.channels = channels

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
    def load_diagnostic(cls, channels:str, grid:Grid):
        return cls(channels, grid)