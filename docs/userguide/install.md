
# Installation

Earth-2 MIP is a Python package and can be installed from source using pip.
However, we suggest setting up a dedicated run environment for Earth-2 MIP.

## Installing from PyPi

Presently Earth-2 MIP is not on PyPi. Install from source in the mean time.

## Install from Release Wheel (Coming soon)

Built Python wheels can be access in Earth-2 MIPs releases on Github.
Navigate to the latest release of Earth-2 MIP and download the latest `.whl`.
Install using pip:

``` bash
pip install wheel_file.whl
```

## Install from Source

To install Earth-2 MIP from source:

```bash
git clone git@github.com:NVIDIA/earth2mip.git

cd earth2mip && pip install .
```

# Earth-2 MIP Environments

When running Earth-2 MIP, it is highly encourage to do so in a dedicated environment to
mitigate the chance of unforseen package conflicts.
Docker is the prefered environment, but if you are not familiar with docker or cannnot
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
docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -p 8888:8888 -it nvcr.io/nvidia/modulus/modulus:23.08
```

```{note}
The above command maps the port `8888` inside the docker container to the port `8888` on
local machine.
This is a requirement needed when using Jupyter Lab to run the notebook examples.
```

Once inside the container install Earth-2 MIP, using your prefered method.

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
