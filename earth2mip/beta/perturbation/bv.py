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

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch

from earth2mip.beta.perturbation.base import PerturbationMethod
from earth2mip.beta.perturbation.brown import Brown


class BredVector:
    """Bred Vector perturbation method, a classical technique for pertubations in
    ensemble forecasting.

    Parameters
    ----------
    model : Callable[[torch.Tensor], torch.Tensor]
        Dynamical model, typically this is the prognostic AI model.
        TODO: Update to prognostic looper
    noise_amplitude : float, optional
        Noise amplitude, by default 0.05
    integration_steps : int, optional
        Number of integration steps to use in forward call, by default 20
    ensemble_perturb : bool, optional
        Perturb the ensemble in an interacting fashion, by default False
    seeding_perturbation_method : PerturbationMethod, optional
        Method to seed the Bred Vector perturbation, by default Brown Noise

    Note
    ----
    For additional information:

    - https://journals.ametsoc.org/view/journals/bams/74/12/1520-0477_1993_074_2317_efantg_2_0_co_2.xml
    - https://en.wikipedia.org/wiki/Bred_vector
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, OrderedDict[str, np.ndarray]],
            tuple[torch.Tensor, OrderedDict[str, np.ndarray]],
        ],
        noise_amplitude: float = 0.05,
        integration_steps: int = 20,
        ensemble_perturb: bool = False,
        seeding_perturbation_method: PerturbationMethod = Brown(),
    ):
        self.model = model
        self.noise_amplitude = noise_amplitude
        self.ensemble_perturb = ensemble_perturb
        self.integration_steps = integration_steps
        self.seeding_perturbation_method = seeding_perturbation_method

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> torch.Tensor:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : OrderedDict[str, np.ndarray]
            Ordered dict representing coordinate system that discribes the tensor

        Returns
        -------
        torch.Tensor
            Perturbation noise tensor
        """
        dx = self.seeding_perturbation_method(x, coords)

        xd = torch.clone(x)
        xd, _ = self.model(xd, coords)
        # Run forward model
        for k in range(self.integration_steps):
            x1 = x + dx
            x2, _ = self.model(x1, coords)
            if self.ensemble_perturb:
                dx1 = x2 - xd
                dx = dx1 + self.noise_amplitude * (dx - dx.mean(dim=0))
            else:
                dx = x2 - xd

        gamma = torch.norm(x) / torch.norm(x + dx)

        return dx * self.noise_amplitude * gamma
