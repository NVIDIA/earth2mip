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

from earth2mip import filesystem


def test_glob(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    # use file:// protocol to ensure handling is correct
    (f,) = filesystem.glob(f"file://{tmp_path.as_posix()}/*.txt")
    assert f == f"file://{a.as_posix()}"


def test_glob_no_scheme(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    (f,) = filesystem.glob(f"{tmp_path.as_posix()}/*.txt")
    assert f == a.as_posix()


def test__to_url():
    assert (
        filesystem._to_url("s3", "sw_climate_fno/a.txt") == "s3://sw_climate_fno/a.txt"
    )
    assert filesystem._to_url("", "sw_climate_fno/a.txt") == "sw_climate_fno/a.txt"
