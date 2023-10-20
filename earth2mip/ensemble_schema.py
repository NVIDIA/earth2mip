from typing import Optional
from pydantic import BaseModel
from earth2mip.schema import Grid, PerturbationStrategy
from earth2mip import weather_events
from earth2mip.diagnostic import DIAGNOSTIC_TYPES


class EnsembleRun(BaseModel):
    """A configuration for running an ensemble weather forecast

    Attributes:
        weather_model: The name of the fully convolutional neural network (FCN) model to use for the forecast.
        ensemble_members: The number of ensemble members to use in the forecast.
        noise_amplitude: The amplitude of the Gaussian noise to add to the initial conditions.
        noise_reddening: The noise reddening amplitude, 2.0 was the defualt set by A.G. work.
        simulation_length: The length of the simulation in timesteps.
        output_frequency: The frequency at which to write the output to file, in timesteps.
        use_cuda_graphs: Whether to use CUDA graphs to optimize the computation.
        seed: The random seed for the simulation.
        ensemble_batch_size: The batch size to use for the ensemble.
        autocast_fp16: Whether to use automatic mixed precision (AMP) with FP16 data types.
        perturbation_strategy: The strategy to use for perturbing the initial conditions.
        perturbation_channels: channel(s) perturbed by the initial condition perturbation strategy, None = all channels
        forecast_name (optional): The name of the forecast to use (alternative to `weather_event`).
        weather_event (optional): The weather event to use for the forecast (alternative to `forecast_name`).
        output_dir (optional): The directory to save the output files in (alternative to `output_path`).
        output_path (optional): The path to the output file (alternative to `output_dir`).
        restart_frequency: if provided save at end and at the specified frequency. 0 = only save at end.
        grf_noise_alpha: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details
        grf_noise_sigma: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details
        grf_noise_tau: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details

    """  # noqa

    weather_model: str
    simulation_length: int
    diagnostic: Optional[list[DIAGNOSTIC_TYPES]] = []
    # TODO make perturbation_strategy an Enum (see ChannelSet)
    perturbation_strategy: PerturbationStrategy = PerturbationStrategy.correlated
    perturbation_channels: Optional[list[str]] = None
    noise_reddening: float = 2.0
    noise_amplitude: float = 0.05
    output_frequency: int = 1
    output_grid: Optional[Grid] = None
    ensemble_members: int = 1
    seed: int = 1
    ensemble_batch_size: int = 1
    # alternatives for specifiying forecast
    forecast_name: Optional[str] = None
    weather_event: Optional[weather_events.WeatherEvent] = None
    # alternative for specifying output
    output_dir: Optional[str] = None
    output_path: Optional[str] = None
    restart_frequency: Optional[int] = None
    grf_noise_alpha: float = 2.0
    grf_noise_sigma: float = 5.0
    grf_noise_tau: float = 2.0

    def get_weather_event(self) -> weather_events.WeatherEvent:
        if self.forecast_name:
            return weather_events.read(self.forecast_name)
        else:
            return self.weather_event
