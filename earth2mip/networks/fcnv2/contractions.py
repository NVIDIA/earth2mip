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

# Helper routines for FNOs


@torch.jit.script
def compl_contract2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,kixyr->srbkxy", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res


@torch.jit.script
def compl_contract2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bixy,kixy->bkxy", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def compl_contract_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bins,kinr->srbkn", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res


@torch.jit.script
def compl_contract_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("bin,kin->bkn", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def compl_ttc1_c_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    res = torch.einsum("jt,bct->jbct", ac, bc)
    return torch.view_as_real(res)


@torch.jit.script
def compl_ttc2_c_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    res = torch.einsum("oi,icj,jbct->bot", ac, bc, cc)
    return torch.view_as_real(res)


def contract_tt(x, w):
    y = compl_ttc1_c_fwd(w[2], x)
    return compl_ttc2_c_fwd(w[0], w[1], y)


# Helper routines for spherical MLPs
@torch.jit.script
def compl_mul1d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixs,ior->srbox", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res


@torch.jit.script
def compl_mul1d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    res = compl_mul1d_fwd(a, b) + c
    return res


@torch.jit.script
def compl_muladd1d_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul1d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the real-valued case:


@torch.jit.script
def compl_mul1d_fwd_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bix,io->box", a, b)
    return res


@torch.jit.script
def compl_muladd1d_fwd_r(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmp = compl_mul1d_fwd_r(a, b)
    return tmp + c


# Helper routines for FFT MLPs


@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ior->srboxy", a, b)
    res = torch.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
    )
    return res


@torch.jit.script
def compl_mul2d_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    res = compl_mul2d_fwd(a, b) + c
    return res


@torch.jit.script
def compl_muladd2d_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmpcc = torch.view_as_complex(compl_mul2d_fwd_c(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the real-valued case:
@torch.jit.script
def compl_mul2d_fwd_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixy,io->boxy", a, b)
    return res


@torch.jit.script
def compl_muladd2d_fwd_r(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    tmp = compl_mul2d_fwd_c(a, b)
    return torch.view_as_real(tmp + c)
