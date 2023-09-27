# Earth-2 MIP Examples

Earth-2 MIP provide both Jupyter notebook and python workflow examples.
Generally speaking, new users are encouraged to use the notebooks which contain in depth information
on how to use Earth-2 MIP and the type of analysis that can be done out of the box.
The workflow scripts provide more advanced use cases which can be used to more involved analysis, extension and customization
in Earth-2 MIP.

## Running Notebooks

While Earth-2 MIP can be installed in a Python/Conda enviroment, its suggested that
users run Earth 2 MIP inside the Modulus base container provided on NGC.
The following instructions can be used to run the examples notebooks using either docker
container, apptainer/singularity image or conda environment.

### Docker

Start with cloniong the repository to your local machine (if you haven't already) use
the following commands:

```bash
git clone https://github.com/NVIDIA/earth2mip.git
cd earth2mip
```

Launch the Modulus docker container with port mapping as below:

```bash
docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -p 8888:8888 -v ${PWD}:/earth2mip -it nvcr.io/nvidia/modulus/modulus:23.08
```

The above command maps the port `8888` inside the docker container to the port `8888` on local machine.

Once inside the container, navigate to the mounted folder `/earth2mip/`, install
Earth-2 MIP package and launch the `jupyter lab` as shown below.

```bash
cd /earth2mip
pip install .
cd examples
jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password=''
```

This will start the jupyter lab server inside the docker container.
The Jupyter lab enviroment will be hosted at [http://localhost:8888/](http://localhost:8888/).

### Singularity (Apptainer)

For systems/compute environments that do not allow docker directly, a
Singularity/Apptainer definition file is also provided to build an image.
This will build a `sif` file from the Modulus docker container and set up Jupyter lab inside.
Starting in the root directory of this prepository, to build the enviroment run the
following command:

```bash
singularity build --fakeroot --sandbox earth2mip.sif examples/earth2mip.def
```

This should create `earth2mip.sif` in the root of the repository folder.
The following command can now be used to bind the Earth-2 MIP repo into the singularity
container, install an edittable version of Earth-2 MIP and run Jupyter lab.

```bash
singularity run -B ${PWD}:/workspace/earth2mip earth2mip.sif
```

If your system requires specific set up for Jupyter lab, the following command can be
modified below.

```bash
singularity exec -B ${PWD}:/workspace/earth2mip --nv earth2mip.sif bash -c 'cd ~
    jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=examples/'
```

The Jupyter lab enviroment will be hosted at [http://localhost:8888/](http://localhost:8888/).

For a development environment, one could use a writtable container and perform an
editable install of Earth-2 MIP but this does not work on many systems.
Alternatively, the following command can be used to create a local python virtual
environment in the bind directory and install Earth-2 MIP in edittable mode.
When running Jupyter lab, use the 'earth2mip' kernel to execute the notebooks.
This enables edits to the Earth-2 MIP module in the host directory that take immediate
effect on the notebooks running inside the singularity container.

```bash
singularity exec -B ${PWD}:/workspace/earth2mip --nv earth2mip.sif bash -c 'cd ~
    python3 -m venv --system-site-packages .venv --prompt earth2mip
    source .venv/bin/activate
    ipython kernel install --user --name=earth2mip
    pip3 install -e .
    jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=examples/'
```

### Conda Environment

For a more vanilla Pythonic experience, we suggest setting up Earth-2 MIP in a
[Conda enviroment](https://www.anaconda.com/download).
This will avoid the need for Docker/Singularity but will also require your system to be
configured correctly to support the required dependencies.
Start with creating a new Conda environment with Python 3.10:

```bash
conda create --name earth2mip python=3.10
conda activate earth2mip
```

Next we need to clone and install Earth-2 MIP:

```bash
git clone git@github.com:NVIDIA/earth2mip.git && cd earth2mip 
pip install .
```

On top of the dependencies installed with Earth-2 MIP, a few more will be needed for
running some of the models:

```bash
pip install tensorly tensorly-torch

git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

conda install -c dglteam dgl
```

Lastly we need to install and setup Jupyter lab with the following commands.
Pick a password to log into your Jupyter instance:

```bash
pip install jupyterlab
jupyter-lab --generate-config
jupyter-lab password
```

Be sure you are inside the Earth-2 repository folder and execute the following command
to launch the Jupyter lab session.

```bash
jupyter-lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=examples/
```

The Jupyter lab enviroment will be hosted at [http://localhost:8888/](http://localhost:8888/).

## Running workflows

To run the sample workflows, follow the same procedure as above to launch the docker
container / singularity enviroment and ensure Earth-2 MIP is installed.
The workflows can then be excuted using python, e.g.

```bash
cd workflows/
python pangu_24.py
```
