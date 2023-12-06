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

import random
import time
from concurrent.futures import ThreadPoolExecutor


def test_thread_pool_always_returns_same_order():

    pool = ThreadPoolExecutor(4)

    def func(x):
        # ensure that threads all finish at different times
        time.sleep(random.uniform(0, 0.01))
        return x

    items = list(range(10))

    for i in range(4):
        assert list(pool.map(func, items)) == items
