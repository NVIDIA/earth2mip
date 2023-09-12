# Earth-2 MIP Examples

Earth-2 MIP provide both Jupyter notebook and python workflow examples.
Generally speaking, new users are encouraged to use the notebooks which contain in depth information
on how to use Earth-2 MIP and the type of analysis that can be done out of the box.
The workflow scripts provide more advanced use cases which can be used to more involved analysis, extension and customization
in Earth-2 MIP.

## Running Notebooks

While Earth-2 MIP can be installed in a Python/Conda enviroment, its suggested that users run Earth 2 MIP inside the Modulus base container provided on NGC.
The following instructions can be used to run the examples notebooks using either docker container or apptainer/singularity sandbox.

To clone the repository to your local machine (if you haven't already) use the following commands:

```bash
git clone https://github.com/NVIDIA/earth2mip.git
cd earth2mip
```
For the following steps, make sure you are in the *root* directory of the Earth-2 MIP repository.

### Docker

Launch the Modulus docker container with port mapping as below:

```bash
docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia -p 8888:8888 -v ${PWD}:/earth2mip -it nvcr.io/nvidia/modulus/modulus:23.08
```

The above command maps the port `8888` inside the docker container to the port `8888` on local machine.

Once inside the container, navigate to the mounted folder `/earth2mip/`, install earth2mip package and launch the `jupyter lab` as shown below.

```bash
cd /earth2mip
pip install .
cd examples
jupyter lab --ip='*' --NotebookApp.token='' --NotebookApp.password=''
```

This will start the jupyter lab server inside the docker container. Once it is running, you can navigate to the `http://localhost:8888/` to access the jupyter environment and start using the notebooks.

### Apptainer / Singularity

For systems / compute enviroments that do not allow docker directly, a Apptainer definition file is also provided.
This will build a sandbox from the Modulus docker container and set up Jupyter lab inside.
To build the enviroment, run the following command:

```bash
apptainer build --fakeroot --sandbox earth2mip.sif examples/earth2mip.def
```

This should create `earth2mip.sif` in the root of the repository folder.
Next use the following command to launch Jupyter lab in the enviroment hosted at `http://localhost:8888/`:


```bash
apptainer run --writable --nv earth2mip.sif jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/examples/
```

## Running workflows

To run the sample workflows, follow the same procedure as above to launch the docker conatainer / apptainer enviroment and ensure Earth-2 MIP is installed.
The workflows can then be excuted using python, e.g.

```
cd workflows/
python pangu_24.py
```
