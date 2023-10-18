import torch
from typing import Literal
from pydantic import BaseModel, Extra
from earth2mip.geo_function import GeoFunction
from earth2mip.model_registry import Package


class DiagnosticBase(torch.nn.Module, GeoFunction):
    """Diagnostic model base class"""

    @classmethod
    def load_package(cls, *args, **kwargs) -> Package:
        """Class function used to create the diagnostic model package (if needed).
        This should be where any explicit download functions should be orcastrated
        """
        pass

    @classmethod
    def load_diagnostic(cls, *args, **kwargs):
        """Class function used to load the diagnostic model onto device memory and
        create an instance of the diagnostic for use
        """
        pass


class DiagnosticConfigBase(BaseModel):
    """Diagnostic model config base class"""

    # Used to discrimate between config classes, sub classes should overwrite
    type: Literal["DiagnosticBase"] = "DiagnosticBase"

    def initialize(self) -> DiagnosticBase:
        package = DiagnosticBase.load_package()
        return DiagnosticBase.load_diagnostic(package)

    class Config:
        # Don't allow any extra params in diagnostic configs, be strict
        extra = Extra.forbid
