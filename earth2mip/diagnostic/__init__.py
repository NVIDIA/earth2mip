import sys
from typing import Union
from earth2mip.model_registry import Package
from earth2mip.diagnostic.wind_speed import WindSpeed
from earth2mip.diagnostic.climate_net import ClimateNet

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


DIAGNOSTIC_REGISTY = {
    "windspeed": WindSpeed,
    "climatenet": ClimateNet,
}


def _get_config_types():
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
    # Returns tuple here so its compatable with the typing.Union[]
    return tuple(diagnostic_types)


# Build config type dynamically
DIAGNOSTIC_TYPES = Union[_get_config_types()]


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
