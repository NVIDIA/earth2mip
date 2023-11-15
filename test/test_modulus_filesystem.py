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

from pathlib import Path

import pytest
from modulus.utils import filesystem


@pytest.fixture
def pyfile_name():
    return "test_filesystem.py"


def test_modulus_filesystem_local(pyfile_name):
    # Check if this test file is seen in a Fsspec local file system
    file_path = Path(__file__).parent.resolve()
    fssystem = filesystem._get_fs("file")
    assert pyfile_name in [Path(file).name for file in fssystem.ls(file_path)]
