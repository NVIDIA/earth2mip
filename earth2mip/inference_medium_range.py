import logging

import os
import xarray as xr
import torch
import argparse
import numpy as np
import datetime
import sys
from earth2mip import config
from earth2mip import schema, time_loop
from earth2mip.initial_conditions.era5 import HDF5DataSource
from earth2mip import _cli_utils
from modulus.distributed.manager import DistributedManager


__all__ = ["score_deterministic"]


def get_times():
    # the IFS data Jaideep downloaded only has 668 steps (up to end of november 2018)
    nsteps = 668
    times = [
        datetime.datetime(2018, 1, 1) + k * datetime.timedelta(hours=12)
        for k in range(nsteps)
    ]
    return times


class RMSE:
    def __init__(self, weight=None):
        self._xy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xy = self._mean((truth - pred) ** 2)
        return xy.cpu()

    def gather(self, seq):
        return torch.sqrt(sum(seq) / len(seq))


class ACC:
    def __init__(self, mean, weight=None):
        self.mean = mean
        self._xy = {}
        self._xx = {}
        self._yy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xx = self._mean((truth - self.mean) ** 2).cpu()
        yy = self._mean((pred - self.mean) ** 2).cpu()
        xy = self._mean((pred - self.mean) * (truth - self.mean)).cpu()
        return xx, yy, xy

    def gather(self, seq):
        """seq is an iterable of (xx, yy, xy) tuples"""
        # transpose seq
        xx, yy, xy = zip(*seq)

        xx = sum(xx)
        xy = sum(xy)
        yy = sum(yy)
        return xy / torch.sqrt(xx) / torch.sqrt(yy)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("inference")


def flat_map(func, seq, *args):
    for x in seq:
        yield from func(x, *args)


def run_forecast(
    model: time_loop.TimeLoop, n, initial_times, device, data_source, mean
):
    mean = mean.squeeze()
    assert mean.ndim == 3

    nlat = {schema.Grid.grid_720x1440: 720, schema.Grid.grid_721x1440: 721}[model.grid]
    channels = [
        data_source.channel_names.index(name) for name in model.out_channel_names
    ]
    mean = mean[channels, :nlat]
    mean = torch.from_numpy(mean).to(device)

    ds = data_source[initial_times[0]]
    lat = np.deg2rad(ds.lat).values
    assert lat.ndim == 1
    weight = np.cos(lat)[:, np.newaxis]
    weight_torch = torch.from_numpy(weight).to(device)

    if model.grid == schema.Grid.grid_720x1440:
        weight_torch = weight_torch[:720, :]

    acc = ACC(mean, weight=weight_torch)
    metrics = {"acc": acc, "rmse": RMSE(weight=weight_torch)}

    def process(initial_time):
        logger.info(f"Running {initial_time}")
        initial_condition = data_source[initial_time]
        logger.debug("Initial Condition Loaded.")
        x = torch.from_numpy(initial_condition.values[None, :, channels]).to(device)
        i = -1
        for valid_time, data, _ in model(x=x, time=initial_time):
            assert data.shape[1] == len(model.out_channel_names)
            i += 1
            if i > n:
                break

            lead_time = valid_time - initial_time
            logger.debug(f"{valid_time}")
            # TODO may need to fix n_history here
            v = data_source[valid_time]
            verification = v.values[:, channels, :nlat, :]
            verification_torch = torch.from_numpy(verification).to(device)

            output = {}
            for name, metric in metrics.items():
                output[name] = metric.call(verification_torch, data)
            yield (initial_time, lead_time), output

    # collect outputs for lead_times
    my_channels = np.array(model.out_channel_names)
    return metrics, my_channels, list(flat_map(process, initial_times))


def gather(seq, metrics, model_name, channels):
    outputs_by_lead_time = {}
    initial_times = set()
    for (initial_time, lead_time), metric_values in seq:
        forecasts_at_lead_time = outputs_by_lead_time.setdefault(lead_time, [])
        forecasts_at_lead_time.append(metric_values)
        initial_times.add(initial_time)

    def to_dataset(metric, name):
        outputs = {
            k: [v[name] for v in snapshots]
            for k, snapshots in outputs_by_lead_time.items()
        }
        times, accs = zip(*outputs.items())
        times = list(times)
        acc_arr = [metric.gather(acc) for acc in accs]
        stacked = torch.stack(acc_arr, 0)
        stacked = stacked.cpu().numpy()
        return xr.DataArray(
            stacked,
            dims=["lead_time", "channel"],
            coords={"lead_time": times, "channel": channels},
        ).to_dataset(name=name)

    ds = xr.merge(to_dataset(metric, name) for name, metric in metrics.items())
    ds = ds.assign(
        initial_times=xr.DataArray(list(initial_times), dims=["initial_time"])
    )

    return ds


def score_deterministic(
    model: time_loop.TimeLoop, n: int, initial_times, data_source, time_mean
):
    """Compute deterministic accs and rmses

    Args:
        model: the inference class
        n: the number of lead times
        initial_times: the initial_times to compute over
        data_source: a mapping from time to dataset, used for the initial
            condition and the scoring
        time_mean: a (channel, lat, lon) numpy array containing the time_mean.
            Used for ACC.

    Returns:
        metrics

    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = f"cuda:{rank % world_size}"
    else:
        rank = 0
        world_size = 1
        device = "cuda:0"

    local_initial_times = initial_times[rank::world_size]

    metrics, channels, seq = run_forecast(
        model,
        n=n,
        device=device,
        initial_times=local_initial_times,
        data_source=data_source,
        mean=time_mean,
    )

    if world_size > 1:
        output_list = [None] * world_size
        torch.distributed.all_gather_object(output_list, seq)
    else:
        output_list = [seq]

    if rank == 0:
        seq = []
        for item in output_list:
            seq.extend(item)
        return gather(
            seq,
            metrics=metrics,
            model_name=model,
            channels=channels,
        )


def main():
    parser = argparse.ArgumentParser()
    _cli_utils.add_model_args(parser, required=True)
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--test", action="store_true")
    # TODO refactor this to a shared place
    parser.add_argument(
        "--data", type=str, help="path to hdf5 root directory containing data.json"
    )

    args = parser.parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()

    initial_times = get_times()
    if args.test:
        initial_times = initial_times[-dist.world_size :]

    model = _cli_utils.model_from_args(args, dist.device)

    data_source = HDF5DataSource.from_path(args.data or config.ERA5_HDF5_73)

    # time mean
    ds = score_deterministic(
        model, args.n, initial_times, data_source, time_mean=data_source.time_means
    )

    if dist.rank == 0:
        ds.attrs["model"] = args.model
        ds.attrs["history"] = " ".join(sys.argv)
        output = os.path.abspath(args.output)
        dirname = os.path.dirname(args.output)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        ds.to_netcdf(output)


if __name__ == "__main__":
    main()
