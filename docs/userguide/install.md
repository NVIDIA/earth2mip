
# Installation

Earth-2 MIP is a Python package and can be installed from source using pip.
However, we suggest setting up a dedicated run environment for Earth-2 MIP.

## Install from Source (Recommended)

To install Earth-2 MIP from source:

```bash
git clone git@github.com:NVIDIA/earth2mip.git

cd earth2mip && pip install .
```

## Install from Release Wheel

Built Python wheels can be accessed in Earth-2 MIPs releases on Github:

``` bash
curl -L https://github.com/NVIDIA/earth2mip/releases/download/v0.1.0/earth2mip-0.1.0-py3-none-any.whl > earth2mip-0.1.0-py3-none-any.whl

pip install earth2mip-0.1.0-py3-none-any.whl
```

## Installing from PyPi

Presently Earth-2 MIP is not on PyPi. Install from source in the mean time.

# Earth-2 MIP Environments

When running Earth-2 MIP, it is highly encouraged to do so in a dedicated environment to
mitigate the chance of unforseen package conflicts.
Docker is the preferred environment, but if you are not familiar with docker or cannnot
run a docker container Conda is typically a suitable alternative.

## Conda Environment

Start with installing [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
/ [Anaconda](https://docs.continuum.io/free/anaconda/).
Once installed we will create a new conda enviorment for Earth-2 MIP:

```bash
conda create --name earth2mip python=3.10
conda activate earth2mip
```

Next we can install Earth-2 MIP using one of the listed methods above. E.g.

```bash
git clone git@github.com:NVIDIA/earth2mip.git

cd earth2mip && pip install .
```

If you are interested in running Jupyter Notebooks (such as the provided examples),
create a iPython kernel using the environment to use in [Jupyter lab](https://jupyterlab.readthedocs.io/en/stable/user/running.html):

```bash
pip install ipykernel
python -m ipykernel install --user --name=earth2mip-kernel
```

## Docker Environment

For the most consistent environment, we suggest using the Modulus docker container
that is avaiable on the Nvidia container registry.
Launch the Modulus docker container with port mapping as below:

```bash
docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -p 8888:8888 -it nvcr.io/nvidia/modulus/modulus:23.11
```

```{note}
The above command maps the port `8888` inside the docker container to the port `8888` on
local machine.
This is a requirement needed when using Jupyter Lab to run the notebook examples.
```

Once inside the container install Earth-2 MIP, using your preferred method.
Note you will have to pip install every time you run the container.
Alternatively, you can not use `--rm` in the run command and keep the stopped
container to then [start again](https://docs.docker.com/engine/reference/commandline/start/)
in the future.

## Singularity / Apptainer Environment

On many systems, singularity/apptainer is the containerize environment solution of choice,
typically due to better security.
The modulus docker container can be converted into a singularity container with the
following definition file being placed in the root of the Earth-2 MIP
repository:

```dockerfile
Bootstrap: docker
FROM: nvcr.io/nvidia/modulus/modulus:23.11

%files
    earth2mip/* /workspace/earth2mip/earth2mip/
    MANIFEST.in /workspace/earth2mip/MANIFEST.in
    setup.cfg /workspace/earth2mip/setup.cfg
    setup.py /workspace/earth2mip/setup.py
    README.md /workspace/earth2mip/README.md
    LICENSE.txt /workspace/earth2mip/LICENSE.txt
    examples/README.md /workspace/earth2mip/examples/README.md

%post
    cd /workspace/earth2mip && pip3 install .
    pip3 install cartopy

%environment
    export HOME=/workspace/earth2mip/

%runscript
    cd ~

%labels
    AUTHOR NVIDIA Earth-2 and Modulus Team
```

A Singularity Image Format (`sif`) file can get built with the following command:

```bash
singularity build --fakeroot --sandbox earth2mip.sif earth2mip.def
```

This can then be ran using the following execute command:

```bash
singularity exec -B ${PWD}:/workspace/earth2mip --nv earth2mip.sif bash -c 'cd ~'
```

```{note}
The following translates easily to apptainer where the commands / definition file can be
translated easily. See [migration guide](https://apptainer.org/docs/admin/main/singularity_migration.html)
for additional details.
```

```{note}
Singularity / Apptainer containers are not recommended for running Jupyter lab.
Instead consider a conda environment.
```

# Optional Dependicies

Earth-2 MIP has a number of optional dependencies that can be installed depending on the
use case.

## Models

Install optional dependencies for Pangu weather:

```bash
pip install .[pangu]
```

Install optional dependencies for Graphcast:

```bash
pip install .[graphcast]
pip install -r requirements.txt
```

## Development

Install development packages for linting/formating/etc:

```bash
pip install .[dev]
```

Install packages for building documentation:

```bash
pip install .[docs]
```
