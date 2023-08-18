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

from earth2mip import networks, schema
import argparse


def add_model_args(parser: argparse.ArgumentParser, required=False):
    if required:
        parser.add_argument("model", type=str)
    else:
        parser.add_argument("--model", type=str)
    parser.add_argument(
        "--model-metadata",
        type=str,
        help="metadata.json file. Defaults to the metadata.json in the model package.",
        default="",
    )


def model_from_args(args, device):
    if args.model_metadata:
        with open(args.model_metadata) as f:
            metadata = schema.Model.parse_raw(f.read())
    else:
        metadata = None

    return networks.get_model(args.model, device=device, metadata=metadata)
