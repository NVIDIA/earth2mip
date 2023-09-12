# Earth-2 MIP (Alpha)

<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![GitHub](https://img.shields.io/github/license/NVIDIA/earth2mip)](https://github.com/NVIDIA/earth2mip/blob/master/LICENSE.txt)
[![Documentstion](https://img.shields.io/badge/docs-online-green)](https://nvidia.github.io/earth2mip/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->

Earth-2 Model Intercomparison Project (MIP) is python based AI framework that
enables climate researchers and scientists explore and experiment with the use of AI
models for for weather and climate.
It provides reference workflows for understanding how AI models capture the understanding
of our  climate system and how they can work with traditional numerical weather
forecasting models.
For instance, the repo provides a uniform interface for running inference using
pre-trained model checkpoints and scoring the skill of such models using certain
standard metrics.
This repository is meant to facilitate the weather and climate community to come up with
good reference baseline of events to test the models against and to use with a variety
of data sources.

## Installation

Earth-2 MIP will be installable on PyPi upon general release.
Install from source in the mean time:

```bash
git clone git@github.com:NVIDIA/earth2mip.git

cd earth2mip && pip install .
```

To verify installation run:

```bash
make pytest
```

## Getting Started

The [notebooks](./examples/notebooks/) folder contains reference Jupyter notebooks for a
few common workflows and usecases.
These are the best way to start exploring Earth-2 MIP :

1. [Ensemble Inference](./examples/notebooks/01_ensemble_inference.ipynb) - This
notebook will demonstrate how to perturb the initial conditions to generate ensemble of
inference outputs for a given pre-trained model and generate some standard metrics to
evaluate the skill of the model.

2. [Model Inter Comparison](./examples/notebooks/02_model_comparison.ipynb) - This
notebook will showcase how to generate forecasts from multiple models with Earth-2 MIP
and how to compare their predictions.

### Supported Models

These notebooks illustrate how-to-use with a few models and this can serve as reference
to bring in your own checkpoint as long as its compatible. There may be additional work
to make it compatible with Earth-2 MIP.
Earth-2 MIP leverages the model zoo in Modulus framework to provide a reference set of
models. The goal is to enable to community to grow this collection of models as showin
the table below.

<!-- markdownlint-disable -->
| ID | Model | Architecture | Type | Reference | Source | Size |
|:-----:|:-----:|:-------------------------------------------:|:--------------:|:---------:|:-------:|:---:|
| fcn* | FourCastNet |    Adaptive Fourier Neural Operator  | global weather |   [Arxiv](https://arxiv.org/abs/2202.11214)   | modulus |
| dlwp |  Deep Learning Weather Prediction  |  Convolution Encoder-Decoder | global weather |   [AGU](https://doi.org/10.1029/2020MS002109)   | modulus |  50Mb |
| pangu | Pangu Weather  |  Vision Transformer | global weather |  [Nature](https://doi.org/10.1038/s41586-023-06185-3) | onnx | 2Gb |
| fcnv2_sm |  FourCastNet v2 | Spherical Harmonics Fourier Neural Operator | global weather |  [Arxiv](https://arxiv.org/abs/2306.03838)  | modulus | 3.5Gb |
<!-- markdownlint-enable -->

\* = coming soon

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

- Github Discussions: Discuss new ideas, model integraiton, support etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.

## License

Earth-2 MIP is provided under the Apache License 2.0, please see
[LICENSE.txt](./LICENSE.txt) for full license text.

### Additional Resources

- [Earth-2 Website](https://www.nvidia.com/en-us/high-performance-computing/earth-2/)
- [NVIDIA Modulus](https://github.com/NVIDIA/modulus)
