import torch
import numpy as np
from typing import Literal
from modulus.distributed import DistributedManager
from earth2mip import config
from earth2mip.schema import Grid
from earth2mip.model_registry import Package
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase
from earth2mip.model_registry import ModelRegistry

# Temp custom package for the model
try:
    from e2mipgust import restore_checkpoint
    from e2mipgust.network.precip_sfno import PercipSFNO  # TODO: Eliminate
except:
    restore_checkpoint = None

IN_CHANNELS = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
]

OUT_CHANNELS = ["10fg"]


class WindGust(DiagnosticBase):
    """Wind gust diagnostic model

    TODO: This can likely be generalized to a general network diagnostic model but
    probably not worth it.

    Example
    -------
    >>> package = WingGust.load_package()
    >>> windgust = WindGust.load_diagnostic(package)
    >>> x = torch.randn(1, 73, 721, 1440)
    >>> out = windgust(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
    ):
        super().__init__()
        self.grid = Grid.grid_721x1440

        self._in_channels = IN_CHANNELS
        self._out_channels = OUT_CHANNELS

        self.model = model
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)

    @property
    def in_channels(self) -> list[str]:
        return self._in_channels

    @property
    def out_channels(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> Grid:
        return self.grid

    @property
    def out_grid(self) -> Grid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        x = (x - self.in_center) / self.in_scale
        out = self.model(x)
        out = self.out_scale * out + self.out_center
        print(out.shape)
        return out

    @classmethod
    def load_package(cls, registry: str = "s3://earth2_server/diagnostics") -> Package:
        registry = ModelRegistry("s3://earth2_server/diagnostics")
        return registry.get_model("gustnet")

    @classmethod
    def load_diagnostic(cls, package: Package, device="cuda:0"):

        if restore_checkpoint is None:
            raise ImportError("Failed to import e2mipgust package")

        model = PercipSFNO().to(
            device
        )  # TODO: Eliminate with 1 method for instan and loading of weights
        model = restore_checkpoint(package.get("best_ckpt_mp0.tar"), model)
        input_center = torch.Tensor(np.load(package.get("global_means.npy")))
        input_scale = torch.Tensor(np.load(package.get("global_stds.npy")))
        # Todo: change this so its not a dict and is uniform with inputs
        diag_norms = np.load(package.get("diagnostic_norms.npy"), allow_pickle=True)[()]
        out_center = torch.Tensor([diag_norms["fg10"]["mean"]])[:, None, None]
        out_std = torch.Tensor([diag_norms["fg10"]["std"]])[:, None, None]

        return cls(model, input_center, input_scale, out_center, out_std).to(device)


class WindGustConfig(DiagnosticConfigBase):

    type: Literal["WindGust"] = "WindGust"

    def initialize(self):
        dm = DistributedManager()
        package = WindGust.load_package()
        return WindGust.load_diagnostic(package, device=dm.device)
