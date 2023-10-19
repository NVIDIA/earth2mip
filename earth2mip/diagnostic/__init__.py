import sys
import torch
from typing import Union
from earth2mip.schema import Grid
from earth2mip.model_registry import Package
from earth2mip.geo_function import GeoFunction
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase
from earth2mip.diagnostic.identity import Identity
from earth2mip.diagnostic.filter import Filter
from earth2mip.diagnostic.wind_speed import WindSpeed

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


DIAGNOSTIC_REGISTY = {"identity": Identity, "filter": Filter, "windspeed": WindSpeed}


def get_config_types():
    """Method for getting a list of diagnostic config types for DiagnosticTypes
    which should be used in down stream
    """
    diagnostic_types = []
    # Add known diagnostic models
    for value in DIAGNOSTIC_REGISTY.values():
        try:
            diagnostic_types.append(value.load_config_type())
        except NotImplementedError:
            pass

    # Add any entrypointed
    group = "earth2mip.diagnostic"
    entrypoints = entry_points(group=group)
    for entry_point in entrypoints:
        if entry_point.name not in DIAGNOSTIC_REGISTY:
            try:
                config_type = entry_point.load().load_config_type()
                diagnostic_types.append(config_type)
            except NotImplementedError:
                pass


DIAGNOSTIC_TYPES = Union[get_config_types()]


def get_package(name: str) -> Package:
    """Fetch diagnostic package (if needed)

    Parameters
    ----------
    name : str
        Diagnostic function name / entrypoint
    """
    if name in DIAGNOSTIC_REGISTY:
        return DIAGNOSTIC_REGISTY[name].load_package()

    group = "earth2mip.diagnostic"
    entrypoints = entry_points(group=group)
    for entry_point in entrypoints:
        print(entry_point.name)
        if entry_point.name not in DIAGNOSTIC_REGISTY:
            return entry_point.load().load_package()

    raise ValueError(f"Diagnostic {name} not found")


def get_diagnostic(name: str, *args, **kwargs):
    """Fetch diagnostic object

    Parameters
    ----------
    name : str
        Diagnostic function name / entrypoint
    """
    if name in DIAGNOSTIC_REGISTY:
        return DIAGNOSTIC_REGISTY[name].load_diagnostic(*args, **kwargs)

    group = "earth2mip.diagnostic"
    entrypoints = entry_points(group=group)
    for entry_point in entrypoints:
        if entry_point.name not in DIAGNOSTIC_REGISTY:
            return entry_point.load().load_diagnostic(*args, **kwargs)

    raise ValueError(f"Diagnostic {name} not found")


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
