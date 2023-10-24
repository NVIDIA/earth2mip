import torch
from typing import Protocol
from abc import abstractmethod
from earth2mip.schema import Grid


class GeoFunction(Protocol):
    """Geo Function

    This is the most primative functional of Earth-2 MIP which represents a functional
    operations on geographical data to produce geographical data. This implies the
    following two requirements:
        1) The function must define in and out channel variables representing the fields
           in the input/output arrays.
        2) The function must define the in and out grid schemas.

    Many auto-gressive models can be represented as a GeoFunction and can maintain a
    internal state. Diagnostic models must be a GeoFunction by definition.

    Warning
    -------
    Geo Function is a concept not full adopted in Earth-2 MIP and is being adopted
    progressively.
    """

    @property
    def in_channel_names(self) -> list[str]:
        pass

    @property
    def out_channel_names(self) -> list[str]:
        pass

    @property
    def in_grid(self) -> Grid:
        pass

    @property
    def out_grid(self) -> Grid:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
