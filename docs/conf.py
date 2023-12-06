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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import pathlib
import sys
from importlib.metadata import version

import dotenv
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# -- Load environment vairs -----------------------------------------------------
# Note: To override, use environment variables (e.g. PLOT_GALLERY=True make html)
# Defaults will build API docs for
dotenv.load_dotenv()
doc_version = os.getenv("DOC_VERSION", "main")
plot_gallery = os.getenv("PLOT_GALLERY", False)
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", False)
print(doc_version, plot_gallery, run_stale_examples)

root = pathlib.Path(__file__).parent
modulus = root.parent / "third_party" / "modulus"
release = version("earth2mip")

sys.path.insert(0, root.parent.as_posix())
sys.path.insert(0, modulus.as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
version = ".".join(release.split(".")[:2])
project = "Earth-2 MIP"
copyright = "2023, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "announcement": "Earth-2 MIP is in Beta! Expect potential API instability!",
    "logo": {
        "text": "Earth-2 MIP",
        "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
        "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    },
    "navbar_align": "content",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/NVIDIA/earth2mip/gh-pages/_static/switcher.json",
        "version_match": doc_version,  # Set DOC_VERSION env variable to change
    },
    "external_links": [
        {
            "name": "Changelog",
            "url": "https://github.com/NVIDIA/earth2mip/blob/main/CHANGELOG.md",
        },
    ],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/NVIDIA/earth2mip",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}
favicons = ["favicon.ico"]

# https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples/",
    "gallery_dirs": "examples",
    "plot_gallery": plot_gallery,
    "image_srcset": ["2x"],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": "/[0-9]+.*.py",  # Run python files that start with number
    "run_stale_examples": run_stale_examples,
}
