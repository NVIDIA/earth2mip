import modulus
import torch
import torch.nn.functional as F
from modulus.models.afno import AFNO


class PeriodicPad2d(torch.nn.Module):
    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


class PrecipNet(modulus.Module):
    def __init__(
        self,
        inp_shape,
        in_channels,
        out_channels,
        patch_size=(8, 8),
        embed_dim=768,
    ):
        super().__init__()
        self.backbone = AFNO(
            inp_shape=inp_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            num_blocks=8,
        )
        self.ppad = PeriodicPad2d(1)
        self.conv = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.act = torch.nn.ReLU()
        self.eps = 1e-5

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x
