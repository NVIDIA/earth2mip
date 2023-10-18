import torch
from earth2mip.schema import Grid
from earth2mip.geo_function import GeoFunction
from earth2mip.diagnostic.base import DiagnosticBase
from earth2mip.diagnostic.identity import Identity
from earth2mip.diagnostic.filter import Filter
from earth2mip.diagnostic.wind_speed import WindSpeed

class Diagnostic(GeoFunction):
    """Diagnostic factory class. This is useful for constructing a chain of diagnostic
    operations either by Python or from configs.

    Note
    ----
    This requires new diagnostic functions to be added with a construction function
    here, as well as config support (TODO)

    Args:
        in_channels (list[str]): Input channels
        in_grid (Grid): Input grid
    """

    def __init__(self, in_channels: list[str], in_grid: Grid):
        self.diaglist = DiagnosticList(in_channels, in_grid)


    def wind_speed(self, level:str, grid:Grid):
        WindSpeed.load_package()
        


class DiagnosticList(GeoFunction):
    """List of diagnostic functions that are executed in sequential order. This is
    useful for building compositions of functions to create complex outputs.

    Note
    ----
    Presently this executes and refines channels greedly. In other words only the output
    channel of a function is ever preserved. Expanding with concat functions is likely
    for the future.

    Args:
        in_channels (list[str]): Input channels
        in_grid (Grid): Input grid
    """


    def __init__(self, in_channels: list[str], in_grid: Grid):
        self.diagnostics = [Identity(in_channels, in_grid)]


    def add(self, diagfunc: DiagnosticBase):
        """Add 

        Args:
            diagfunc (DiagnosticBase): _description_
        """
        # Create filter to transition between diagnostics
        dfilter = Filter(self.diagnostics[-1].out_channels, diagfunc.in_channels)

        self.diagnostics.append(dfilter)
        self.diagnostics.append(diagfunc)


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

