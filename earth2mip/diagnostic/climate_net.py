# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from earth2mip import config, grid
from earth2mip.diagnostic.base import DiagnosticBase
from earth2mip.model_registry import ModelRegistry, Package

IN_CHANNELS = [
    "tcwv",
    "u850",
    "v850",
    "msl",
]

OUT_CHANNELS = [
    "climnet_bg",  # Background
    "climnet_tc",  # Tropical Cyclone
    "climnet_ar",  # Atmospheric River
]


# =========================== TODO Move somehwere ============================
# ============================================================================
# Architecture: https://arxiv.org/pdf/1811.08201.pdf
class Wrap(torch.nn.Module):
    """Climate net wrapper for padding."""

    def __init__(self, padding):
        super(Wrap, self).__init__()
        self.p = padding

    def forward(self, x):
        # creating the circular padding
        return F.pad(x, (self.p,) * 4, mode="circular")


class ConvBNPReLU(nn.Module):
    """Convolutional Net with Batch Norm and PreLU."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    """Batch Norm with PReLU layer."""

    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):
    """Convolutional Layer with Batch Norm."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        output = self.bn(output)
        return output


class Conv(nn.Module):
    """Ordinary Convolutional Layer."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output


class ChannelWiseConv(nn.Module):
    """Channel Wise convolutional layer."""

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output


class DilatedConv(nn.Module):
    """Convolutional Layer with Dilation."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride, bias=False, dilation=d
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output


class ChannelWiseDilatedConv(nn.Module):
    """Channel-wise Convolutional Layer with Dilation."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False, dilation=d
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.padding(input)
        output = self.conv(output)
        return output


class FGlo(nn.Module):
    """The FGlo class is employed to refine the joint feature of both local feature and
    surrounding context."""

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """The size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)"""

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  # size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    """Context Guided Block."""

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


class InputInjection(nn.Module):
    """Inject Input with pooling."""

    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(Wrap(padding=1))
            self.pool.append(nn.AvgPool2d(3, stride=2))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class CGNetModule(nn.Module):
    """
    CGNet (Wu et al, 2018: https://arxiv.org/pdf/1811.08201.pdf) implementation.
    This is taken from their implementation, we do not claim credit for this.
    """

    def __init__(self, classes=19, channels=4, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(
            channels, 32, 3, 2
        )  # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + channels)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32 + channels, 64, dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.bn_prelu_2 = BNPReLU(128 + channels)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128 + channels, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find("ConvTranspose2d") != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.interpolate(
            classifier, input.size()[2:], mode="bilinear", align_corners=False
        )  # Upsample score map, factor=8
        return out


# ============================================================================
# ============================================================================


class ClimateNet(DiagnosticBase):
    """Climate Net Diagnostic model, built into Earth-2 MIP. This model can be used to
    create prediction labels for tropical cyclones and atmopheric rivers. Produces
    non-standard output channels climnet_bg, climnet_tc and climnet_ar representing
    background label, tropical cyclone and atmopheric river labels.

    Note:
        This model and checkpoint are from Prabhat et al. 2021
        https://doi.org/10.5194/gmd-14-107-2021
        https://github.com/andregraubner/ClimateNet

    Example:
        >>> package = ClimateNet.load_package()
        >>> model = ClimateNet.load_diagnostic(package)
        >>> x = torch.randn(1, 4, 721, 1440)
        >>> out = model(x)
        >>> out.shape
        (1, 3, 721, 1440)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
    ):
        super().__init__()
        self.grid = grid.equiangular_lat_lon_grid(721, 1440)

        self._in_channels = IN_CHANNELS
        self._out_channels = OUT_CHANNELS

        self.model = model
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)

    @property
    def in_channel_names(self) -> list[str]:
        return self._in_channels

    @property
    def out_channel_names(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> grid.LatLonGrid:
        return self.grid

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # noqa
        x = (x - self.in_center) / self.in_scale
        out = self.model(x)
        return torch.softmax(out, 1)  # Softmax channels

    @classmethod
    def load_package(
        cls, registry: str = os.path.join(config.MODEL_REGISTRY, "diagnostics")
    ) -> Package:
        registry = ModelRegistry(registry)
        return registry.get_model("e2mip://climatenet")

    @classmethod
    def load_diagnostic(cls, package: Package, device="cuda:0"):

        model = CGNetModule(
            channels=len(IN_CHANNELS),
            classes=len(OUT_CHANNELS),
        )
        weights_path = package.get("weights.tar")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        input_center = torch.Tensor(np.load(package.get("global_means.npy")))[
            :, None, None
        ]
        input_scale = torch.Tensor(np.load(package.get("global_stds.npy")))[
            :, None, None
        ]

        return cls(model, input_center, input_scale).to(device)
