import argparse
import logging
import os
import json
from functools import partial
import torch
from modulus.distributed.manager import DistributedManager

from earth2mip.inference_ensemble import run_inference, get_perturbator
from earth2mip.ensemble_utils import brown_noise
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun
# from earth2mip.time_loop import TimeLoop


# TODO change all noise funcs to update x and not return noise and unify functions
def generate_model_noise_correlated(x,
                                    time_step,
                                    reddening,
                                    device,
                                    noise_injection_amplitude,
                                    ):
    shape = x.shape
    dt = torch.tensor(time_step.total_seconds()) / 3600.0
    noise = noise_injection_amplitude * dt * brown_noise(shape, reddening).to(device)
    return x * (1.0 + noise)


def get_source(
    device,
    config,
):
    
    source = partial(
        generate_model_noise_correlated,
        reddening=config.noise_reddening,
        device=device,
        noise_injection_amplitude=0.003)
    return source


def main(config=None):
    logging.basicConfig(level=logging.INFO)

    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        parser.add_argument("--weather_model", default=None)
        args = parser.parse_args()
        config = args.config

    # If config is a file
    if os.path.exists(config):
        config: EnsembleRun = EnsembleRun.parse_file(config)
    # If string, assume JSON string
    elif isinstance(config, str):
        config: EnsembleRun = EnsembleRun.parse_obj(json.loads(config))
    # Otherwise assume parsable obj
    else:
        raise ValueError(
            f"Passed config parameter {config} should be valid file or JSON string"
        )

    # if args and args.weather_model:
    #     config.weather_model = args.weather_model

    # Set up parallel
    DistributedManager.initialize()
    device = DistributedManager().device
    group = torch.distributed.group.WORLD

    logging.info(f"Earth-2 MIP config loaded {config}")
    logging.info(f"Loading model onto device {device}")
    model = get_model(config.weather_model, device=device)
    logging.info(f"Constructing initializer data source")
    perturb = get_perturbator(
        model,
        config,
    )
    model.source = get_source(device, config)
    logging.info(f"Running inference")
#     time_loop = pangu.PanguInference(perturb=perturb_gaussian) # make time loop
    run_inference(model, config, perturb, group)


if __name__ == "__main__":
    main()
