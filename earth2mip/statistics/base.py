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

import torch
from typing import List, Protocol, Union, runtime_checkable

Tensor = torch.Tensor 

@runtime_checkable
class Statistic(Protocol):
    """ Statistics interface. """

    def __call__(self, x: Tensor, dim: Union[int, List[int]] = 0) -> Union[Tensor, List[Tensor]]:
        """ Compute the statistics of x, over the specified dimensions [dim]. 
        
        Parameters
        ----------
        x : Tensor
            Input data to take statistic of.
        dim : Union[int, List[int]], Optional = 0
            Dimension over which to do the statistical reduction.

        Returns
        -------
        Union[Tensor, List[Tensor]]
            Statistical reduction of x.    
        """
        pass
    
    def __update__(self, x: Tensor, *old_stats: Tensor, dim: Union[int, List[int]] = 0) -> Union[Tensor, List[Tensor]]:
        """ Update old statistics with new inputs, x, to compute updated statistics. 

        Parameters
        ----------
        x : Tensor
            Input data to take statistic of.
        old_stats : Tensor
            Arbitrary list of old statistic tensors.
        dim : Union[int, List[int]], Optional = 0
            Dimension over which to do the statistical reduction.

        Returns
        -------
        Union[Tensor, List[Tensor]]
            Updated statistics after reducing over x.
        """