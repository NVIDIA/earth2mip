# Earth-2 MIP Examples

Earth-2 MIP provide both Jupyter notebook and python workflow examples.
Generally speaking, new users are encouraged to use the notebooks which contain in depth information
on how to use Earth-2 MIP and the type of analysis that can be done out of the box.
The workflow scripts provide more advanced use cases which can be used to more involved analysis, extension and customization
in Earth-2 MIP.



## Running Notebooks

To run the notebooks inside this folder inside the Modulus docker container, you can follow the below steps

First, clone this repo in your working directory

```bash
git clone <repo url>
cd earth2mip
```

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


## Running workflows

To run the sample workflows, follow the same procedure as above to launch the docker conatainer and install `earth2mip`. Once installed, run workflows using

```
cd workflows/
python pangu_24.py
```
