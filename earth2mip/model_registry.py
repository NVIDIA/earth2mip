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

"""Create-read-update-delete (CRUD) operations for the FCN model registry

The location of the registry is configured using `config.MODEL_REGISTRY`. Both
s3:// and local paths are supported.

The top-level structure of the registry is like this::

    afno_26ch_v/
    baseline_afno_26/
    gfno_26ch_sc3_layers8_tt64/
    hafno_baseline_26ch_edim512_mlp2/
    modulus_afno_20/
    sfno_73ch/
    tfno_no-patching_lr5e-4_full_epochs/


The name of the model is the folder name. Each of these folders has the
following structure::

    sfno_73ch/about.txt            # optional information (e.g. source path)
    sfno_73ch/global_means.npy
    sfno_73ch/global_stds.npy
    sfno_73ch/weights.tar          # model checkpoint
    sfno_73ch/metadata.json


The `metadata.json` file contains data necessary to use the model for forecasts::

    {
        "architecture": "sfno_73ch",
        "n_history": 0,
        "grid": "721x1440",
        "in_channels": [
            0,
            1
        ],
        "out_channels": [
            0,
            1
        ]
    }

Its schema is provided by the :py:class:`earth2mip.schema.Model`.

The checkpoint file `weights.tar` should have a dictionary of model weights and
parameters in the `model_state` key. For backwards compatibility with FCN
checkpoints produced as of March 1, 2023 the keys should include prefixed
`module.` prefix. This checkpoint format may change in the future.


Scoring FCNs under active development
-------------------------------------

One can use fcn-mip to score models not packaged in fcn-mip using a metadata
file like this::

    {
        "architecture": "pickle",
        ...
    }

This will load ``weights.tar`` using `torch.load`. This is not recommended for
long-time archival of model checkpoints but does allow scoring models under
active development. Once a reasonable skill is achieved the model's source code
can be stabilized and packaged within fcn-mip for long-term archival.

"""
import json
import logging
import os
import urllib
import zipfile
from typing import Literal

from earth2mip import filesystem, schema

logger = logging.getLogger(__file__)

METADATA = "metadata.json"


class Package:
    """A model package

    Simple file system operations and quick metadata access

    """

    def __init__(self, root: str, seperator: str):
        self.root = root
        self.seperator = seperator

    def get(self, path, recursive: bool = False):
        return filesystem.download_cached(self._fullpath(path), recursive=recursive)

    def _fullpath(self, path):
        return self.root + self.seperator + path

    def metadata(self) -> schema.Model:
        metadata_path = self._fullpath(METADATA)
        local_path = filesystem.download_cached(metadata_path)
        with open(local_path) as f:
            return schema.Model.parse_raw(f.read())


# TODO: Replace with some NGC file system scheme
def download_ngc_package(root: str, url: str, zip_file: str):
    model_registry = os.path.dirname(root)
    if not os.path.isdir(root):
        os.makedirs(model_registry, exist_ok=True)  # Make sure registry folder exists
        logger.info("Downloading NGC model checkpoint, this may take a bit")
        urllib.request.urlretrieve(
            url,
            f"{model_registry}/{zip_file}",
        )
        # Unzip
        with zipfile.ZipFile(f"{model_registry}/{zip_file}", "r") as zip_ref:
            zip_ref.extractall(model_registry)
        # Clean up zip
        os.remove(f"{model_registry}/{zip_file}")
    else:
        logger.debug("Package already found, skipping download")


def FCNPackage(root: str, seperator: str):
    download_ngc_package(
        root=root,
        url="https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcn/"
        + "versions/v0.2/files/fcn.zip",
        zip_file="fcn.zip",
    )
    return Package(root, seperator)


def DLWPPackage(root: str, seperator: str):
    download_ngc_package(
        root=root,
        url="https://api.ngc.nvidia.com/v2/models/nvidia/modulus/"
        + "modulus_dlwp_cubesphere/versions/v0.2/files/dlwp_cubesphere.zip",
        zip_file="dlwp_cubesphere.zip",
    )
    return Package(root, seperator)


def FCNv2Package(root: str, seperator: str):
    download_ngc_package(
        root=root,
        url="https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/"
        + "versions/v0.2/files/fcnv2_sm.zip",
        zip_file="fcnv2_sm.zip",
    )
    return Package(root, seperator)


def NGCDiagnosticPackage(
    root: str, seperator: str, name: Literal["precipitation_afno", "climatenet"]
):
    download_ngc_package(
        root=root,
        url="https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_diagnostics/"
        + f"versions/v0.1/files/{name}.zip",
        zip_file=f"{name}.zip",
    )
    return Package(root, seperator)


def PanguPackage(root: str, seperator: str):

    name = root.split(seperator)[-1]
    if not os.path.isdir(root):
        logger.info(
            "Downloading Pangu 6hr / 24hr model checkpoints, this may take a bit"
        )
        os.makedirs(root, exist_ok=True)
        # Wget onnx files
        if name == "pangu" or name == "pangu_24":
            urllib.request.urlretrieve(
                "https://get.ecmwf.int/repository/test-data/ai-models/"
                + "pangu-weather/pangu_weather_24.onnx",
                f"{root}/pangu_weather_24.onnx",
            )
        if name == "pangu" or name == "pangu_6":
            urllib.request.urlretrieve(
                "https://get.ecmwf.int/repository/test-data/ai-models/"
                + "pangu-weather/pangu_weather_6.onnx",
                f"{root}/pangu_weather_6.onnx",
            )
        # For completeness and compatability
        if name == "pangu":
            entry_point = "earth2mip.networks.pangu:load"
        elif name == "pangu_24":
            entry_point = "earth2mip.networks.pangu:load_24"
        else:
            entry_point = "earth2mip.networks.pangu:load_6"

        with open(os.path.join(root, "metadata.json"), "w") as outfile:
            json.dump(
                {"entrypoint": {"name": entry_point}},
                outfile,
                indent=2,
            )
    else:
        logger.info("Pangu package already found, skipping download")

    return Package(root, seperator)


class ModelRegistry:
    SEPERATOR: str = "/"

    def __init__(self, path: str):
        self.path = path

    def list_models(self):
        return [os.path.basename(f) for f in filesystem.ls(self.path)]

    def get_model(self, name: str):
        if name.startswith("e2mip://"):
            return self.get_builtin_model(name)

        return Package(self.get_path(name), seperator=self.SEPERATOR)

    def get_builtin_model(self, name: str):
        """Built in models that have globally buildable packages"""
        # TODO: Add unique name prefix for built in packages?
        name = name.replace("e2mip://", "")
        if name == "fcn":
            return FCNPackage(self.get_path(name), seperator=self.SEPERATOR)
        if name == "fcnv2_sm":
            return FCNv2Package(self.get_path(name), seperator=self.SEPERATOR)
        elif name == "dlwp":
            return DLWPPackage(self.get_path(name), seperator=self.SEPERATOR)
        elif name == "pangu" or name == "pangu_24" or name == "pangu_6":
            return PanguPackage(self.get_path(name), seperator=self.SEPERATOR)
        elif name == "precipitation_afno":
            return NGCDiagnosticPackage(
                self.get_path(name), seperator=self.SEPERATOR, name=name
            )
        elif name == "climatenet":
            return NGCDiagnosticPackage(
                self.get_path(name), seperator=self.SEPERATOR, name=name
            )
        elif name.startswith("graphcast"):
            return Package("gs://dm_graphcast", seperator="/")
        else:
            raise ValueError(f"Model {name} not registered in e2mip package registry")

    def get_path(self, name, *args):
        return self.SEPERATOR.join([self.path, name, *args])

    def get_model_path(self, name: str):
        return self.get_path(name)

    def get_weight_path(self, name: str):
        return self.get_path(name, "weights.tar")

    def get_scale_path(self, name: str):
        return self.get_path(name, "global_stds.npy")

    def get_center_path(self, name: str):
        return self.get_path(name, "global_means.npy")

    def put_metadata(self, name: str, metadata: schema.Model):
        metadata_path = self.get_path(name, METADATA)
        filesystem.pipe(metadata_path, metadata.json().encode())

    def get_metadata(self, name: str) -> schema.Model:
        return self.get_model(name).metadata()
