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
from typing import Literal, Type
from pydantic import BaseModel, Extra
from earth2mip.geo_function import GeoFunction
from earth2mip.model_registry import Package


class DiagnosticBase(torch.nn.Module, GeoFunction):
    """Diagnostic model base class"""

    @classmethod
    def load_package(cls, *args, **kwargs) -> Package:
        """Class function used to create the diagnostic model package (if needed).
        This should be where any explicit download functions should be orcastrated
        """
        pass

    @classmethod
    def load_diagnostic(cls, *args, **kwargs):
        """Class function used to load the diagnostic model onto device memory and
        create an instance of the diagnostic for use
        """
        pass

    @classmethod
    def load_config_type(cls, *args, **kwargs) -> Type[BaseModel]:
        """Class function used access the Pydantic config class of the diagnostic if
        one has been implemented. Note this returns a class reference, not instantiated
        object.
        """
        raise NotImplementedError("This diagnostic does not have a config implemented")
        pass


class DiagnosticConfigBase(BaseModel):
    """Diagnostic model config base class"""

    # Used to discrimate between config classes, sub classes should overwrite
    type: Literal["DiagnosticBase"] = "DiagnosticBase"

    def initialize(self) -> GeoFunction:
        package = DiagnosticBase.load_package()
        return DiagnosticBase.load_diagnostic(package)

    class Config:
        # Don't allow any extra params in diagnostic configs, be strict
        extra = Extra.forbid