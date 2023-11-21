# Earth-2 MIP Contribution Guide

(Under construction)

## Contribute to Earth-2 MIP

### Pull Requests

Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo)
the [upstream](https://github.com/NVIDIA/Earth2-MIP) Earth-2 MIP repository.

2. Git clone the forked repository and push changes to the personal fork.

3. Once the code changes are staged on the fork and ready for review, a
[Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR)
can be [requested](https://help.github.com/en/articles/creating-a-pull-request)
to merge the changes from a branch of the fork into a selected branch of upstream.

    - Exercise caution when selecting the source and target branches for the PR.
    - Ensure that you update the [`CHANGELOG.md`](CHANGELOG.md) to reflect your contributions.
    - Creation of a PR creation kicks off CI and a code review process.
    - Atleast one Nvidia engineer will be assigned for the review.

4. The PR will be accepted and the corresponding issue closed after adequate review and
testing has been completed. Note that every PR should correspond to an open issue and
should be linked on Github.

### Notebooks

Do not commit the outputs of notebooks.
Run the following command to remove output cells / meta data from notebooks before
committing it to git history:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --log-level=ERROR <notebook.ipynb>
```

### CI
Linting. Manually run the linting:

    make lint

Run it before every commit:

- (optional) install pre-commit hooks and git lfs with `make setup-ci`.
  After this command is run, you will need to fix any lint errors before
  commiting. This needs to be done once per local clone of this repository.


To run the test suite:

    pytest

To run quick tests (takes 10 seconds):

  pytest -m 'not slow'

To reset the regression data:

  pytest --regtest-reset

and then check in the changes.


### Licensing Information

All source code files should start with this paragraph:

```bash
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
```

### Doc Strings

Earth-2 MIP uses [google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) doc strings.
If you are a VSCode user, using the [auto-doc string extension](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) is suggested.


### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the
contribution is your original work, or you have rights to submit it under the same
license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license
    document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.

  ```

## Building Documentation

There are two commands for building the docs:

```bash
# Build the core and API docs
make docs

# Build the core, API and example docs
# This will require all examples to get executed and assumed the correct run env
make docs-full
```

### Adding a New Docs Version

To build a release version of docs the following steps need to be taken:

1. Update the `docs/_static/switcher.json` to include your new docs version under the main branch.
For example, for release 1.0.0 one would add the following:

```json
{
        "name": "1.0.0",
        "version": "1.0.0",
        "url": "https://nvidia.github.io/earth2mip/v/1.0.0/"
}
```

2. Update the `docs/.env` documentation version variable to the respective version (it
should match the version string you set in the JSON).
3. Run `make docs`, this should build the html file. Verify the build looks okay. The
version drop down won't appear the be updated because its not updated on Github.
4. You now need to manually move the contents of `docs/_build/html` into the `gh-pages`
branch of the repo under `v/<version number>`. You'll see other versions inside the `v`
folder, create your own.
5. Commit this change with the message `Documentation version <version number> upload`
and push to the `gh-pages` branch.
6. Back in `main` branch revert `docs/.env` and commit the updated `switcher.json` plus
anything else.
7. Open a PR for merging into main. Upon merge the CI will build for main branch and
linking should then be set up for the new version.

