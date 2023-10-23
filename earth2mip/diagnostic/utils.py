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


def filer_channels(
    input: torch.Tensor, in_channels: list[str], out_channels: list[str]
) -> torch.Tensor:
    """Utility function used for selecting a sub set of channels from

    Parameters
    ----------
    input : torch.Tensor
        _description_
    in_channels : list[str]
        _description_
    out_channels : list[str]
        _description_

    Returns
    -------
    torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    indexes_list = []
    try:
        for channel in out_channels:
            indexes_list.append(in_channels.index(channel))
    except ValueError as e:
        raise ValueError(
            "Looks like theres a mismatch between input and "
            + f"requested channels. {e}"
        )
    indexes = torch.IntTensor(indexes_list).to(input.device)
    return torch.index_select(input, 1, indexes)
