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

import builtins
import hashlib
import logging
import os
import urllib.request
from typing import List

import fsspec
import fsspec.implementations.cached
import s3fs

logger = logging.getLogger(__name__)

LOCAL_CACHE = os.path.expanduser("~") + "/.cache/earth2mip"


def _cache_fs(fs):
    return fsspec.implementations.cached.CachingFileSystem(
        fs=fs, cache_storage=LOCAL_CACHE
    )


def _get_fs(path):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url="https://pbss.s8k.io"))
    else:
        return fsspec.filesystem("file")


def open(path, mode="r"):
    if path.startswith("s3://"):
        fs = _get_fs(path)
        cached_fs = _cache_fs(fs)
        return cached_fs.open(path, mode)
    else:
        return builtins.open(path, mode)


def download_cached(path: str, recursive: bool = False) -> str:
    sha = hashlib.sha256(path.encode())
    filename = sha.hexdigest()
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    cache_path = os.path.join(LOCAL_CACHE, filename)

    url = urllib.parse.urlparse(path)

    # TODO watch for race condition here
    if not os.path.exists(cache_path):
        logger.debug("Downloading %s to cache: %s", path, cache_path)
        if path.startswith("s3://"):
            fs = _get_fs(path)
            fs.get(path, cache_path)
        elif url.scheme == "http" or url.scheme == "https":
            # TODO: Check if this supports directory fetches
            urllib.request.urlretrieve(path, cache_path)
        elif url.scheme == "file":
            path = os.path.join(url.netloc, url.path)  # noqa
            return path
        elif url.scheme:
            fs = fsspec.filesystem(url.scheme)
            fs.get(path, cache_path, recursive=recursive)
        else:
            return path

    else:
        logger.debug("Opening from cache: %s", cache_path)

    return cache_path


def pipe(dest, value):
    """Save string to dest"""
    fs = _get_fs(dest)
    fs.pipe(dest, value)


def _to_url(scheme, path):
    if scheme:
        return scheme + "://" + path
    else:
        return path


def glob(pattern: str, maxdepth=1) -> List[str]:
    fs = _get_fs(pattern)
    url = urllib.parse.urlparse(pattern)
    return [_to_url(url.scheme, path) for path in fs.glob(pattern, maxdepth=maxdepth)]


def ls(path):
    fs = _get_fs(path)
    return fs.ls(path)
