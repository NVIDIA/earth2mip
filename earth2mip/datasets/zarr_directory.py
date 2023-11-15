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

import json
import os
from collections.abc import MutableMapping

import xarray


class NestedDirectoryStore(MutableMapping):
    """Reads data like this::

        {root}/a/{group}/{variable}
        {root}/b/{group}/{variable}

    The data are assumed to have identical shape and chunking and require a
    .zmetadata file. This store maps

        {root}/a/{group}/{variable}/0 -> {group}/{variable}/0.0

    """

    def __init__(
        self,
        map,
        directories,
        group,
        concat_dim="initial_time",
        static_coords=(),
        dim_rename=None,
    ):
        """
        Args:
            map: a mutable mapping to base off
            directories: a list of directories containing identical data
            concat_dim: ``directories`` will be put in a dimension named ``concat_dim``
            static_coords: A list of variables that should not be
                concated...read from the first example.
            dim_rename: if provided rename the dimensions of the source data

        """

        self._map = map
        self.concat_dim = concat_dim
        self.group = group
        self.dim_rename = dim_rename or {}
        self.static_coords = static_coords
        self._local = {}
        self.directories = directories
        ds = xarray.Dataset()
        ds[self.concat_dim] = [self.concat_dim], directories
        ds.to_zarr(self._local)

    def _get_new_key_chunk(self, k):
        chunk = k.split("/")[-1]
        variable = "/".join(k.split("/")[:-1])
        index, *sub_chunks = chunk.split(".")
        time = self.directories[int(index)]
        new_chunk = ".".join(sub_chunks)
        full_path = f"{time}/{self.group}/{variable}/{new_chunk}"
        return full_path

    def _get_new_key(self, k):
        if os.path.basename(k) == ".zarray":
            time = self.directories[0]
            # TODO parameterize "mean.zarr" or map it to a "group"
            full_path = f"{time}/{self.group}/{os.path.dirname(k)}/.zarray"
            return full_path
        elif os.path.basename(k) == ".zattrs":
            time = self.directories[0]
            full_path = f"{time}/{self.group}/{os.path.dirname(k)}/.zarray"
            return full_path
        elif os.path.basename(k) == ".zgroup":
            time = self.directories[0]
            full_path = f"{time}/{self.group}/.zgroup"
            return full_path
        elif os.path.basename(k) == ".zmetadata":
            return k
        else:
            return self._get_new_key_chunk(k)

    def _modify_zarray(self, v):
        config = json.loads(v)
        config["chunks"] = [1, *config["chunks"]]
        config["shape"] = [len(self.directories), *config["shape"]]
        return json.dumps(config)

    def _modify_zattrs(self, v):
        config = json.loads(v)
        xarray_dim_name = "_ARRAY_DIMENSIONS"
        dims = config.get(xarray_dim_name, [])
        dims_renamed = [self.dim_rename.get(dim, dim) for dim in dims]
        config[xarray_dim_name] = [self.concat_dim, *dims_renamed]
        return json.dumps(config)

    def __getitem__(self, k):
        if k.startswith(self.concat_dim):
            return self._local[k]

        key = self._get_new_key(k)
        if os.path.basename(k) == ".zarray":
            return self._modify_zarray(self._map[key])
        elif os.path.basename(k) == ".zgroup":
            return self._map[key]
        elif os.path.basename(k) == ".zmetadata":
            return json.dumps(self._get_metadata())
        else:
            return self._map[key]

    def __delitem__(self, k):
        k = self._get_new_key(self, k)
        del self._map[k]

    def __iter__(self):
        raise NotImplementedError()

    def __contains__(self, k):
        return (
            (self._get_new_key(k) in self._map)
            or (os.path.basename(k) == ".zmetadata")
            or (k.startswith(self.concat_dim) and k in self._local)
        )

    def __len__(self, k):
        raise NotImplementedError()

    def __setitem__(self, k, v):
        k = self._get_new_key(k)
        self._map[k] = v

    def _get_metadata(self):
        meta = self._map[os.path.join(self.directories[0], self.group, ".zmetadata")]
        meta = json.loads(meta)
        if not meta["zarr_consolidated_format"] == 1:
            raise ValueError("zarr_consolidated_format must be 1")

        metadata_dict = meta["metadata"]
        # use same class to modify the .zarray and other data
        new_meta = {}
        for k in self._local:
            if os.path.basename(k) == ".zarray":
                new_meta[k] = self._local[k].decode()
            elif os.path.basename(k) == (".zattrs"):
                new_meta[k] = self._local[k].decode()

        for key in metadata_dict:
            if os.path.dirname(key) in self.static_coords:
                continue

            if os.path.basename(key) == ".zarray":
                new_meta[key] = self._modify_zarray(json.dumps(metadata_dict[key]))
            elif os.path.basename(key) == (".zattrs"):
                new_meta[key] = self._modify_zattrs(json.dumps(metadata_dict[key]))
            else:
                new_meta[key] = json.dumps(metadata_dict[key])
        return {"metadata": new_meta, "zarr_consolidated_format": 1}
