# Earth2-MIP

## Description

Earth 2 Model Intercomparison Project (MIP) is python inference framework for data-driven weather and climate models.
It aims to provide a uniform interface for running and scoring any such model with a variety of data sources.
It is inspired by the difficulty of sharing and running such models between teammates here at NVIDIA.
Often necessary input data such as normalization constants and hyperparameter values are not packaged alongside the model weights.
Every model typically implements a slightly different interface.
Scoring routines are specific to the model being scored and often not shared between groups.

earth2mip addresses  these points and bridge the gap between the domain experts
who most often are assessing ML models, and the ML experts producing them.

Compared to other projects in this space, earth2mip focuses on scoring models on-the-fly.
It has python APIs suitable for rapid iteration in a jupyter book, CLIs for scoring models distributed over many GPUs, and a flexible
plugin framework that allows anyone to use their own ML models.

## Badges

TODO

On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals

Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation

TODO add these instructions

## Usage

Basic inference example:

TODO make sure this example runs (need CDS api for pangu)

```python
>>> import datetime
>>> from earth2mip.networks import get_model
>>> from earth2mip.initial_conditions import cds
>>> from earth2mip.inference_ensemble import run_basic_inference
>>> time_loop  = get_model("pangu_weather_6", device="cuda:0")
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

Deterministic scoring:

```
>>> time_mean = np.zeros([69, 721, 1440])
>>> score_deterministic(time_loop, data_source=data_source, n=10, initial_times=[datetime.datetime(2018, 1, 1)], time_mean=time_mean)
<xarray.Dataset>
Dimensions:        (lead_time: 11, channel: 69, initial_time: 1)
Coordinates:
  * lead_time      (lead_time) timedelta64[ns] 0 days 00:00:00 ... 2 days 12:...
  * channel        (channel) <U5 'z1000' 'z925' 'z850' 'z700' ... 'u10m' 'v10m' 't2m'
Dimensions without coordinates: initial_time
Data variables:
    acc            (lead_time, channel) float64 1.0 1.0 1.0 ... 0.9923 0.9964
    rmse           (lead_time, channel) float64 1.204e-07 .214e-07 ... 6.548
    initial_times  (initial_time) datetime64[ns] 2018-01-01
```


## Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

We are open to contributions! Please open a PR. Here are some guidelines:

Linting. Manually run the linting:

    make lint

Run it before every commit:

- (optional) install pre-commit hooks and git lfs with `make setup-ci`.
  After this command is run, you will need to fix any lint errors before
  commiting. This needs to be done once per local clone of this repository.

Pull requests (contributor instructions):
- Try to test your code (your reviewer may request you to write some).
- finish your work and run `make lint test`. Fix any errors that come up.
- target MRs to main
- either fork or push a feature branch to this repo directly
- open the MR, and then slack Yair or Noah to review it.

Pull requests (reviewer instructions)
- The reviewer is responsible for merging
- Avoid "squash merge"
- The CI is currently broken. So use the "merge immediately" button.

To run the test suite:

    pytest

To run quick tests (takes 10 seconds):

  pytest -m 'not slow'

To reset the regression data:

  pytest --regtest-reset

and then check in the changes.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.