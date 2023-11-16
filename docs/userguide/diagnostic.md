# Diagnostics

The term diagnostic model refers here to models that do not make a timestep predictions,
but simply predict some new quantities from existing ones at the same time.
By combining a diagnostic and a forecast (prognostic) model we can forecast additional variables without training the latter.
Functionally diagnostics are :py:class:`earth2mip.geo_operator.GeoOperator`, models can be viewed as post-processers in the inference process.
Earth-2 MIP presently provides several built-in diagnostics.
Those that require a model package files are hosted on [NVIDIA's model registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_diagnostics).

## Motivation

A natural question is: what is the point of developing a diagnostic model over a new weather/climate model that performs time integration?
Training AI weather models is a very expensive process, typically requiring a large set training data and compute with a team that has deep learning + weather expertise which is not always available.
The advantage of diagnostic models is that they can be trained invariant of time, on a reduced set of data / variables and produce quantities that are of interest (versus the ones needed to represent the state of the Earth's weather).
This opens the door to extending existing foundational models to a vast range of
downstream tasks both in academia and industry.
The built in ClimateNet diagnostic is a great example of how a diagnostic can be used to produce a very useful quantity (identification of tropical cyclones). Combined with ensemble inference workflows this can generate probabilistic fields of hurricane trajectories with little effort.

## Using Diagnostics

To load / instantiate a diagnostic model, there are two API provided as part of :py:class:`earth2mip.diagnostic.base.DiagnosticBase`. The `load_package` can be called first to create a built in model package, alternatively users can manually create this package for custom registry set ups.
By default diagnostics with checkpoints will have their model package files stored in `${MODEL_REGISTRY}/diagnostics`.
The `load_diagnostic` function can then be used to instantiate the diagnostic model.

Diagnostics are integrated into Earth-2 MIP workflows by wrapping a model time loop (iterator).
The diagnostic :py:class:`earth2mip.diagnostic.DiagnosticTimeLoop` is a class that can be used to attach diagnostic models onto existing Earth-2 MIP workflows.
This iterator will pull data from the provided weather model time loop, feed this output data through each diagnostic model provided and concatenating the outputs together.

The following demonstrates how to use the diagnostic time loop:
```python
from earth2mip.diagnostic import  DiagnosticTimeLoop

# Instantiate weather model
model = get_model(model_name, device=device)

# Instantiate diagnostic model
package = DiagnosticModel.load_package()
diagnostic = DiagnosticModel.load_diagnostic(package)

# Create diagnostic time loop to attach diagnostic models
model_diagnostic = DiagnosticTimeLoop(diagnostics=[diagnostic], model=model)
```

## Custom Diagnostics

Adding a custom diagnostic is simple so long as one meets the functional requirements of :py:class:`earth2mip.diagnostic.base.DiagnosticBase` and :py:class:`earth2mip.geo_operator.GeoOperator`.
The following provides a simple demonstration of how to create a custom diagnostic that just transforms surface temperature from Kelvin to Celcius.

```python
class Kelvin2Celcius(DiagnosticBase):
    def __init__(self, grid: grid.LatLonGrid):
        super().__init__()
        self.grid = grid

    @property
    def in_channel_names(self) -> list[str]:
        return ['t2m']

    @property
    def out_channel_names(self) -> list[str]:
        return ['t2m_c']

    @property
    def in_grid(self) -> grid.LatLonGrid:
        return self.grid

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - 273.15

    @classmethod
    def load_diagnostic(cls):
        pass

    @classmethod
    def load_diagnostic(
        cls, package: Optional[Package], grid: grid.LatLonGrid
    ):
        return cls(grid)
```

## Known Limitations

- The diagnostics used must operate on grids and use channels consistent with the weather model it is attached to.