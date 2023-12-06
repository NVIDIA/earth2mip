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

import sys

from earth2mip.networks import depends_on_time

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import EntryPoint

import pytest


def test_upstream_entrypoint():
    ep = EntryPoint(name=None, group=None, value="sys")
    assert ep.load() == sys

    # refer to an attribute with ":"
    ep = EntryPoint(name=None, group=None, value="sys:argv")
    assert ep.load() == sys.argv

    # if you don't use : it will give an error
    with pytest.raises(ModuleNotFoundError):
        ep = EntryPoint(name=None, group=None, value="sys.argv")
        ep.load()


def test_inspect_for_time():
    def f(x, time):
        pass

    def g(x):
        pass

    assert depends_on_time(f)
    assert not depends_on_time(g)
