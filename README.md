# HENS

This codebase is a fork of the earth2mip repository and the modulus-makani codebase, developed by NVIDIA.  It is used to run huge ensemble (HENS) weather forecasts with the SFNO architecture. It serves as the codebase for the following two papers:

"Huge Ensembles Part II: Properties of a Huge Ensemble of Hindcasts Generated with Spherical Fourier Neural Operators" (submitted to Geoscientific Model Development): https://arxiv.org/abs/2408.03100

"Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators" (submitted to Geoscientific Model development): https://arxiv.org/abs/2408.01581

The license and copyright for the code for these two projects are at:
earth2mip/license_lbl.txt
earth2mip/copyright_notice.txt

There are two primary codebases for HENS: modulus-makani (used for training the ML models) and earth2mip (used for inference, scoring, and analysis).  These two codebases are developed by NVIDIA, and we use forks of the codebases for this project.

The codebases for this are `earth2mip-fork` and `modulus-makani-fork` at the DOI listed in the code and data availability section in the HENS preprints. They are also made available on Github for reference and convenience.
The `earth2mip-fork` is also on Github at : https://github.com/ankurmahesh/earth2mip-fork/tree/HENS.  On Github, please be sure to use the "HENS" branch.
The `modulus-makani-fork` is also on Github at : https://github.com/ankurmahesh/modulus-makani-fork

The rest of this README has the following sections:
- Setting up the environment
- Training SFNO (`modulus-makani-fork`)
- Access to our trained SFNOs used for HENS
- Ensemble Inference with SFNO (`earth2mip-fork`)
- Scoring SFNO (`earth2mip-fork`)
- Analysis Scripts (`earth2mip-fork`)

If you would like to train SFNO, access our trained model weights, or run inference with SFNO, please read the respective sections.  These sections can be modified to according to various config files (YAML files for SFNO hyperparameters that you would like to train) for training and earth2mip configs for the inference.

If you would like to run our models for inference, please see the section on "Ensemble Inference with SFNO".  You should be able to download our trained models and adapt the earth2mip configs to run your own ensembles, either on the same or different initial dates, with different numbers of ensemble members, and with different variables saved.

Our analysis scripts are in the form of (1) postprocessing scripts to reshape the data and (2) Jupyter notebooks to perform the analysis and make the plots: this is included for reproducibility but often is specific to our filepaths.
## Setting up the environment

We use the docker to create an environment used for training and inference.  The docker environment can be created in `modulus-makani-fork`:
```
docker build modulus-makani-fork/docker/build.sh -t YOUR_TARGET_NAME:23.11
```

The docker image used in this work is on Dockerhub (https://hub.docker.com/r/amahesh19/modulus-makani/tags)

```
docker pull amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint
```

## Training SFNO: `modulus-makani-fork`

`modulus-makani` is the codebase used for training SFNO.  We use a fork of the NVIDIA modulus-makani v0.1.0.  We conduct training on Perlmutter at the NERSC on NVIDIA A100 GPUs: https://docs.nersc.gov/systems/perlmutter/architecture/

Our config of SFNO is `sfno_linear_74chq_sc2_layers8_edim620_wstgl2` at `modulus-makani-fork/config/sfnonet.yaml`. Our job to submit the training was run via

```sbatch modulus-makani-fork/mp_submit_pm.sh```

The above script runs the job for inference using spatial model parallelism (`h_parallel_size=4`, which means that the model is split across 4 ranks in the `h` spatial dimension).  Spatial model parallelism allows larger models to be trained, which may not otherwise fit in the GPU memory of 1 GPU. This means that the final trained model checkpoint is saved in 4 chunks.  For inference, we combine these 4 chunks into one, so that the model can be run for inference on a single rank.

We use the ```modulus-makani-fork/convert_legacy_to_flexible.py` Python file, which is called from the script:

```
bash modulus-makani-fork/script_convert_legacy_to_flexible.sh
```

## Access to our trained SFNOs used for HENS

We open-source the learned weights of our models at 3 locations: (1) the DataDryad DOI listed in the paper, (2) (https://huggingface.co/datasets/maheshankur10/hens/tree/main) https://doi.org/10.57967/hf/4200, and (3) https://portal.nersc.gov/cfs/m4416/hens/earth2mip\_prod\_registry/.  Each model is trained with a different initial seed: therefore, the name of each trained model end with "seed[SEED]," where SEED corresponds to the Pytorch seed used for the weight initialization. Each model also includes a `config.json` file which specifies the configuration parameters used for training each SFNO checkpoint. Each model package includes the learned weights of the model and additional information necessary to run the model (e.g. the input to the model is normalized by the data in 'global_means.npy' and 'global_stds.npy').

As the data is hosted on the NERSC high-performance computing facility, it may occasionally be down for maintenance.  If the link above does not work, you can check https://www.nersc.gov/live-status/motd/ to see if the data portal is up.  In cases of downtime, the model checkpoints may also be downloaded at https://huggingface.co/datasets/maheshankur10/hens/tree/main (https://doi.org/10.57967/hf/4200) or the DataDryad repository listed in the Code and Data Availability Section of the HENS preprints.

## Ensemble Inference with SFNO: `earth2mip-fork`

### ERA5 data for inference

We use the files at `modulus-makani-fork/data_process/` to create the ERA5 mirror used for training and inference.

### Running the model for inference

#### Roadmap and important files

`earth2mip-fork` is used to run the ensemble weather forecasts.  In the HENS manuscripts, we use two methods to create an ensemble: bred vector initial condition perturbations (to represent initial condition uncertainty) and multiple checkpoints trained from scratch (to represent model uncertainty). Particular files of interest are our implementation of bred vectors for initial condition perturbations:

'earth2mip-fork/earth2mip/ensemble_utils.py'

Additionally, ensemble inference with the multiple checkpoints is at 

'earth2mip-fork/earth2mip/inference_ensemble.py'

#### Scripts and configs to run the ensemble

In order to run the ensemble, some paths to some dependent data files must be set as environment variables.  The script to set these paths is at earth2mip-fork/set_74ch_vars.sh.  The environment variables can be changed according to the location of the data dependencies on your machine. `prod_heat_index_lookup.zarr` is a table used for calculating the heat index, `percentile_95.tar` is a tar of the 95th percentile calculated for each hour of day and each month of ERA5 used for thresholding in the calculation of extreme statistics, and `percentile_99.tar` is the 99th percentile from ERA5.  `d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc` includes the data that stores the amplitudes used for the bred vector initial condition perturbation method.

This data for these dependent data files is available at https://huggingface.co/datasets/maheshankur10/hens/tree/main and at https://portal.nersc.gov/cfs/m4416/hens/.  You'll have to download this data to your local machine and change the paths `set_74ch_vars.sh` to point to this data.

The general documentation for earth2mip is included below.  For running ensembles, we recommend setting up earth2mip inference_ensemble to work on your local compute environment (ideally using the general earth2mip documentation).  Then, you can make modifications to the earth2mip config json to run ensembles with our trained model. In particular, earth2mip requires defining a config file which sets the parameters of the ensemble.  The config files used for generating huge ensemble forecasts in summer 2023 are at 

'earth2mip-fork/summer23_configs/*'

If you would like to run an ensemble with our models, you can adapt these configs to suit your local directory (e.g. change the save paths) and also to run ensembles that save different variable subsets and use different days as initial conditions.  See the earth2mip documentation for more information about specifying a config for ensemble inference.

The submit script for the huge ensemble is 

'sbatch earth2mip-fork/submit_HENS_summer23.sh' . This script creates a huge (7424 member ensemble) using `earth2mip/inference_ensemble.py` mentioned above

### Hyperparameter Tuning (Figures 2 and 3)

We perform minimal hyperparameter tuning (exploring a small model,  medium model, and large model).  We analyze the spectra of each model size.  And we assess the lagged ensemble performance of each model size.  This allows to assess ensemble performance without having to train multiple checkpoints of each model. You can read more about lagged ensembles in the preprints and at https://arxiv.org/pdf/2401.15305.

The code to view the results of the tuning is at `earth2mip-fork/earth2mip/scripts/spectra/SFNO_Hyperparameter_Spectra.ipynb` (HENS Part 1 Figure 2)

The code to view the results of resampling the number of checkpoints, to determine how many checkpoints is enough is at `earth2mip-fork/scripts/num_checkpoints/MakeFigure-Number_of_Checkpoints.ipynb`

### Scoring the ensemble: overall diagnostics

`earth2mip-fork/earth2mip/score_ensemble_outputs.py` and `earth2mip-fork/earth2mip/time_collection.py` is the file used to score the ensemble against the ERA5 dataset.  We score the dataset using 2018, 2020, and 2023.  This file used `inference_ensemble.py` to generate an ensemble and then uses the earth2mip scoring utils to score the ensemble.  The overall diagnostics include spread-error ratio, CPRS, and ensemble mean RMSE (Part 1: Figures 6-9).

We calculate the overall diagnostics across the entire year, using 730 initial times (00:00 UTC and 12:00 UTC for each day of the validation year).  This diagnostics calculation workflow is through `time_collection.py`, in which the ensemble is run, the output is saved to disk, and the scores are calculated.  If `save_ensemble` is set to True in the time_collection JSON below, then the ensemble is saved as a zarr file; otherwise, if it is false, then the ensemble is deleted.  There is a general template file for this time collection job: `earth2mip-fork/time_collection_config.json`: this config includes the parameters of the ensemble.  The `earth2mip/make_job.py` file converts this template into a config that can be used for the time_collection.

We have our scoring ensemble script here:

```
python earth2mip/make_job.py multicheckpoint time_collection_config.json /PATH/TO/SAVE/OUTPUT/
sbatch original_submit_time_collection.sh
```

### Scoring the ensemble: spectral diagnostics

We include diagnostics of the spectral properties of each member and the ensemble mean (Part 1: Figures 10-12).  The code for deterministic and perturbed spectra is at `earth2mip-fork/earth2mip/scripts/spectra/MakeFigure_Deterministic_Spectral_Analysis.ipynb` . The code for the ensemble spectra is at `earth2mip-fork/earth2mip/scripts/spectra/MakeFigure-EnsembleMeanSpectralAnalysis.ipynb`.

### Scoring the ensemble: extreme diagnostics

The diagnostics on extremes are located in `earth2mip/extreme_scoring_utils.py` and `earth2mip-fork/earth2mip/score_extreme_diagnostics.py`.  These include diagnostics like reliability diagrams, threshold-weighted CRPS, and ROC curves, and Extreme Forecast Index (Part 1: Figures 13-15).

####Extreme Forecast Index Prerequistites

IFS EFI values are directly downloaded from ECMWF (we do not calculate the IFS EFI values).  The SFNO-BVMC requires a model climatology and percentiles on this climatology.  

```
python earth2mip/make_job.py multicheckpoint time_collection_config_mclimate.json /PATH/TO/SAVE/MCLIMATE/OUTPUT/
sbatch submit_mclimate.sh

# After the job is completed, the percentiles are calculated with
sbatch earth2mip/scripts/cdf/mclimate-percentiles-slurmarray.sh
```
#### Calculating the extreme diagnostics

The IFS ensemble is downloaded from TIGGE. The SFNO-BVMC ensemble is generated using the time_collection workflow above (and save_ensemble=True in the time_collection_config).

The scripts that we used to calculate the extreme diagnostics are at

```
bash efi_extreme_scoring_ifs.sh #for IFS
bash efi_score_prerun_sfno.sh #for SFNO-BVMC
```

# Analysis of Huge Ensemble Run of Summer 2023 (figures in HENS Part 2)

We conducted a huge ensemble run with 7,424 ensemble members, initialized each day of summer 2023 and run forward in time for 15 days. In total, this creates almost 25 petabytes of data across all variables.  We save a subset of the variables (accounting for 2 petabytes).  We analyze the properties of this large dataset in HENS Part 2.

## Figures 2,4

The analysis code for Figures 2,4 is available at hens_gmd_reproducibility.zip

## Figure 5: Demo

We run the huge ensemble using using the submit_HENS_summer23.sh script above. We generate the demo for Kansas City using the code at `earth2mip-fork/earth2mip/scripts/demo/kc.py` and `earth2mip-fork/earth2mip/scripts/demo/KansasCity.ipynb`.  This code extracts the forecasts for the Kansas City heatwave initialized at the 10 initial times leading up to the heatwave, and it visualizes the 2D distribution t2m and d2m of one initialized forecast of the heatwave.

## Figures 3,6-12: Calculation of statistics on HENS Summer 2023

We calculated gain at each grid cell with `earth2mip-fork/earth2mip/scripts/HENS_stats/paralleltrials_gain_gridcell.py` and `earth2mip-fork/earth2mip/scripts/HENS_stats/submit_paralleltrials_gain_gridcell.sh` (Figure 3)

In order to calculate statistics on the HENS run for summer 2023, we first had to perform a matrix transpose, to change the way the data was stored in memory.  This transpose enables taking reductions (e.g. mean, min, max) over the huge ensemble dimension (7424).  See HENS Part II Appendix A for more information.

The code to perform this transpose is at `earth2mip-fork/earth2mip/scripts/h5_convert/h5_convert.py`.  We ran the job with

```
sbatch earth2mip-fork/earth2mip/scripts/h5_convert/alldates_submit.sh
```

After the data was transformed, we calculated statistics on the data with `earth2mip-fork/earth2mip/scripts/HENS_stats/reduce.py`.  These statistics tell us the ensemble mean, min, max, variance, and ensemble mean RMSE.  Additionally, we also calculated statistics on bootstrapped ensembles of different sizes (50, 100, 500, 1000, 5000, and 7424 members), including satisfactory ensemble size (ensemble size where 95% of bootstrapped ensembles include the true value), confidence interval of extreme 2m temperature, mean probability of extreme t2m, and RMSE of the best member (with a confidence interval).

```
sbatch earth2mip-fork/earth2mip/scripts/HENS_stats/submit_reduce.sh
```

We calculated Figure 6: number of ensemble members that exceed ERA5 (binned for different sigma values of ERA5) with `earth2mip-fork/earth2mip/scripts/HENS_stats/number_of_samples.ipynb`

We calculated Figure 7: minimum RMSE at `earth2mip-fork/earth2mip/scripts/HENS_stats/Minimum_RMSE.ipynb`

 We calculated the twCRPS, owCRPS and CRPS figures (Figure 8) at `earth2mip-fork/earth2mip/scripts/HENS_stats/Calculate_HENS_CRPS_related.ipynb`. We calculated the twCRPS, owCRPS, and CRPS walkthrough figures (Figure 9) at `earth2mip-fork/earth2mip/scripts/HENS_stats/Create_HENSpartII_walkthrough_figures.ipynb`. 

We calculated the width of the confidence intervals with `earth2mip-fork/earth2mip/scripts/HENS_stats/Confidence_Intervals.ipynb` (Figure 10), which uses the output of the reduce.py file above.

We calculated Figure 11: outlier statistic with `earth2mip-fork/earth2mip/scripts/HENS_stats/Necessary_ensemble_members.ipynb`

We calculated Figure 12 (for the cases where the true value is greater than the IFS ensemble max, what is the HENS max) with `earth2mip-fork/earth2mip/scripts/HENS_stats/HENSmax_IFSmax_ERA5ZScore.ipynb`



# Earth-2 MIP (Beta)

<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/earth2mip)](https://github.com/NVIDIA/earth2mip/blob/master/LICENSE.txt)
[![Documentstion](https://img.shields.io/website?up_message=online&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fnvidia.github.io%2Fearth2mip%2F&label=docs)](https://nvidia.github.io/earth2mip/)
[![codecov](https://codecov.io/gh/NickGeneva/earth2mip/graph/badge.svg?token=0PDBMHCH2C)](https://codecov.io/gh/NickGeneva/earth2mip/tree/main)
[![Python versionm: 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
)](https://github.com/NVIDIA/earth2mip)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->

Earth-2 Model Intercomparison Project (MIP) is a Python based AI framework that
enables climate researchers and scientists to explore and experiment with the use of AI
models for weather and climate.
It provides reference workflows for understanding how AI models capture the physics of
the Earth's atmosphere and how they can work with traditional numerical weather
forecasting models.
For instance, the repo provides a uniform interface for running inference using
pre-trained model checkpoints and scoring the skill of such models using certain
standard metrics.
This repository is meant to facilitate the weather and climate community to come up with
good reference baseline of events to test the models against and to use with a variety
of data sources.

## Installation

Earth-2 MIP will be installable on PyPi upon general release.
In the mean time, one can install from source:

```bash
git clone git@github.com:NVIDIA/earth2mip.git

cd earth2mip && pip install .
```

See [installation documentation](https://nvidia.github.io/earth2mip/userguide/install.html)
for more details and other options.

## Getting Started

Earth-2 MIP provides a set of examples which can be viewed on the [examples documentation](https://nvidia.github.io/earth2mip/examples/index.html)
page which can be used to get started with various workflows.
These examples can be downloaded both as Jupyer Notebooks and Python scripts.
The source Python scripts can be found in the [examples](./examples/) folders.

### Basic Inference

Earth-2 MIP provides high-level APIs for running inference with AI models.
For example, the following can be used to run Pangu weather using an initial state from
the climate data store (CDS):

```bash
python
>>> import datetime
>>> from earth2mip.networks import get_model
>>> from earth2mip.initial_conditions import cds
>>> from earth2mip.inference_ensemble import run_basic_inference
>>> time_loop  = get_model("e2mip://dlwp", device="cuda:0")
>>> data_source = cds.DataSource(time_loop.in_channel_names)
>>> ds = run_basic_inference(time_loop, n=10, data_source=data_source, time=datetime.datetime(2018, 1, 1))
>>> ds.chunk()
<xarray.DataArray (time: 11, history: 1, channel: 69, lat: 721, lon: 1440)>
dask.array<xarray-<this-array>, shape=(11, 1, 69, 721, 1440), dtype=float32, chunksize=(11, 1, 69, 721, 1440), chunktype=numpy.ndarray>
Coordinates:
  * lon      (lon) float32 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
  * lat      (lat) float32 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
  * time     (time) datetime64[ns] 2018-01-01 ... 2018-01-03T12:00:00
  * channel  (channel) <U5 'z1000' 'z925' 'z850' 'z700' ... 'u10m' 'v10m' 't2m'
Dimensions without coordinates: history
```

And you can get ACC/RMSE like this:
```
>>> from earth2mip.inference_medium_range import score_deterministic
>>> import numpy as np
>>> scores = score_deterministic(time_loop,
    data_source=data_source,
    n=10,
    initial_times=[datetime.datetime(2018, 1, 1)],
    # fill in zeros for time-mean, will typically be grabbed from data.
    time_mean=np.zeros((7, 721, 1440))
)
>>> scores
<xarray.Dataset>
Dimensions:        (lead_time: 11, channel: 7, initial_time: 1)
Coordinates:
  * lead_time      (lead_time) timedelta64[ns] 0 days 00:00:00 ... 5 days 00:...
  * channel        (channel) <U5 't850' 'z1000' 'z700' ... 'z300' 'tcwv' 't2m'
Dimensions without coordinates: initial_time
Data variables:
    acc            (lead_time, channel) float64 1.0 1.0 1.0 ... 0.9686 0.9999
    rmse           (lead_time, channel) float64 0.0 2.469e-05 0.0 ... 7.07 2.998
    initial_times  (initial_time) datetime64[ns] 2018-01-01
>>> scores.rmse.sel(channel='z500')
<xarray.DataArray 'rmse' (lead_time: 11)>
array([  0.        , 150.83014446, 212.07880612, 304.98592282,
       381.36510987, 453.31516952, 506.01464974, 537.11092269,
       564.79603347, 557.22871627, 586.44691243])
Coordinates:
  * lead_time  (lead_time) timedelta64[ns] 0 days 00:00:00 ... 5 days 00:00:00
    channel    <U5 'z500'
```

### Supported Models

These notebooks illustrate how-to-use with a few models and this can serve as reference
to bring in your own checkpoint as long as it's compatible. There may be additional work
to make it compatible with Earth-2 MIP.
Earth-2 MIP leverages the model zoo in [Modulus](https://github.com/NVIDIA/modulus) to
provide a reference set of base-line models.
The goal is to enable to community to grow this collection of models as shown in the
table below.

<!-- markdownlint-disable -->
| ID | Model | Architecture | Type | Reference | Source | Size |
|:-----:|:-----:|:-------------------------------------------:|:--------------:|:---------:|:-------:|:---:|
| fcn | FourCastNet | Adaptive Fourier Neural Operator  | global weather | [Arxiv](https://arxiv.org/abs/2202.11214)   | [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcn) | 300Mb |
| dlwp |  Deep Learning Weather Prediction  |  Convolutional Encoder-Decoder | global weather |   [AGU](https://doi.org/10.1029/2020MS002109)   | [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_dlwp_cubesphere) |  50Mb |
| pangu | Pangu Weather (Hierarchical 6 + 24 hr)  |  Vision Transformer | global weather |  [Nature](https://doi.org/10.1038/s41586-023-06185-3) | onnx | 2Gb |
| pangu_6 | Pangu Weather 6hr Model  |  Vision Transformer | global weather |  [Nature](https://doi.org/10.1038/s41586-023-06185-3) | onnx | 1Gb |
| pangu_24 | Pangu Weather 24hr Model |  Vision Transformer | global weather |  [Nature](https://doi.org/10.1038/s41586-023-06185-3) | onnx | 1Gb |
| fcnv2_sm |  FourCastNet v2 | Spherical Harmonics Fourier Neural Operator | global weather |  [Arxiv](https://arxiv.org/abs/2306.03838)  | [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_fcnv2_sm) | 3.5Gb |
| graphcast |  Graphcast, 37 levels, 0.25 deg | Graph neural network | global weather |  [Science](https://www.science.org/doi/10.1126/science.adi2336)  | [github](https://github.com/google-deepmind/graphcast) | 145MB |
| graphcast_small |  Graphcast, 13 levels, 1 deg | Graph neural network | global weather |  [Science](https://www.science.org/doi/10.1126/science.adi2336)  | [github](https://github.com/google-deepmind/graphcast) | 144MB |
| graphcast_operational |  Graphcast, 13 levels, 0.25 deg| Graph neural network | global weather |  [Science](https://www.science.org/doi/10.1126/science.adi2336)  | [github](https://github.com/google-deepmind/graphcast) | 144MB |
| precipitation_afno | FourCastNet Precipitation | Adaptive Fourier Neural Operator  | diagnostic | [Arxiv](https://arxiv.org/abs/2202.11214)   | [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_diagnostics) | 300Mb |
| climatenet | ClimateNet Segmentation Model | Convolutional Neural Network | diagnostic | [GMD](https://doi.org/10.5194/gmd-14-107-2021)   | [modulus](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_diagnostics) | 2Mb |
<!-- markdownlint-enable -->

\* = coming soon

Some models require additional dependencies not installed by default.
Refer to the [installation instructions](https://nvidia.github.io/earth2mip/userguide/install.html)
for details.

*Note* : Each model checkpoint may have its own unique license. We encourage users to
familiarize themselves with each to understand implications for their particular use
case.

We want to integrate your model into the scoreboard to show the community!
The best way to do this is via [NVIDIA Modulus](https://github.com/NVIDIA/modulus).
You can contribute your model (both the training code as well as model checkpoint) and
we can ensure that it is maintained as part of the reference set.

## Contributing

Earth-2 MIP is an open source collaboration and its success is rooted in community
contribution to further the field.
Thank you for contributing to the project so others can build on your contribution.
For guidance on making a contribution to Earth-2 MIP, please refer to the
[contributing guidelines](./CONTRIBUTING.md).

## More About Earth-2 MIP

This work is inspired to facilitate similar engagements between teams here at
NVIDIA - the ML experts developing new models and the domain experts in Climate science
evaluating the skill of such models.
For instance, often necessary input data such as normalization constants and
hyperparameter values are not packaged alongside the model weights.
Every model typically implements a slightly different interface. Scoring routines are
specific to the model being scored and may not be consistent across groups.

Earth-2 MIP addresses  these challenges and bridges the gap between the domain experts
who most often are assessing ML models, and the ML experts producing them.
Compared to other projects in this space, Earth-2 MIP focuses on scoring models
on-the-fly.
It has python APIs suitable for rapid iteration in a jupyter book, CLIs for scoring
models distributed over many GPUs, and a flexible
plugin framework that allows anyone to use their own ML models.
More importantly Earth-2 MIP aspires to facilitate exploration and collaboration within
the climate research community to evaluate the potential of AI models in climate and
weather simulations.

Please see the [documentation page](https://nvidia.github.io/earth2mip/) for in depth
information about Earth-2 MIP, functionality, APIs, etc.

## Communication

- Github Discussions: Discuss new ideas, model integration, support etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.

## License

Earth-2 MIP is provided under the Apache License 2.0, please see
[LICENSE.txt](./LICENSE.txt) for full license text.

### Additional Resources

- [Earth-2 Website](https://www.nvidia.com/en-us/high-performance-computing/earth-2/)
- [NVIDIA Modulus](https://github.com/NVIDIA/modulus)
