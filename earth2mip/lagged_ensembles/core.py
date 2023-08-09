"""
# Lagged Average Forecasting

Noah Brenowitz
July 3, 2023

This notebook contains the first comparison of probablistic IFS and SFNO forecasts.

I try the concept of Lagged Average Forecating (Hoffman and Kalnay, 1982) for generating ensembles. This is perhaps the earliest ensemble technique that was applied to operational weather prediction. The number of ensemble members is limited, but because they are drawn directly from the data distribution no new tuning knobs are added. This technique is valuable for comparing ML and physical models since only deterministic forecasts are required. This method may not be ideal for forecast applications, but it can tell us if a deterministic model (SFNO, IFS, etc) is "too predictable". "Too predictable" determinstic models improve deterministic accuracy at the cost of having a calibrated ensemble.

I computed spread errors based on an ensemble from 9 rolling lags of lead time.
All the “ensemble members” have the same valid_time, but their initializations differ by 12 hours. Specifically,
let $x(t_0, t)$ be the forecast valid at time $t$ but starting at $t_0 \leq t$. The lagged-ensemble of size $m$ is $S_m(t_0, t)=\{ x(t_0 - (m-1)/2 h ,t ),\ldots,x(t_0, t),\ldots, x(t_0+(m-1)/2, t) \}$. $h$ is the time step of the forecast/validation data. Note that the ensemble contains future members so is not practical for real-time forecast application, but by having the same number of future and past lags it ensures that the average lead-time of the ensemble is $t-t_0$. Without this, some further post-processing would be required to ensure that long lead time simulations are weighted less, an approach that Hoffman and Kalnay call "tempering". For forecast applications, we should be able to get hourly outputs from ECWMF, so can in theory generate up to 24 ensemble members from only 1 day of lag.


Preliminary results with LAF are similar to our other initialization methods. The ensemble mean is more skillful than the deterministic model.

It is intriguing that z500, our "worst" channel relative to IFS also has the best ratio between spread and skill. t2m. It's not surprise since this fields have much sharper features. Since the models are dissipative the spread is too small, but the error of any individual deterministic member is higher.

A key question: **Do we expect spread-error to be 1:1 for LAF?**. Also, do we need to consider a simple model calibration to allow comparing models "maximizing sharpness subject to calibration" (Raftery et. al., 2005).

## Caveats

- This is with /lustre/fsw/sw_climate_fno/nbrenowitz/model_packages/sfno_coszen, an older, non-finetuned model. Would be interesting to retry with a better model.
- For computational simplicity only show scores over the initial 10 valid_times
- Unlike with initial condition perturbations, we know that some ensemble members---the longer lead-times---are less accurate. These ensemble members should be weighted less.

## References


Hoffman, R. N., & Kalnay, E. (1983). Lagged average forecasting, an alternative to Monte Carlo forecasting. Tellus A Dynamic Meteorology and Oceanography, 35A(2), 100–118. https://doi.org/10.1111/j.1600-0870.1983.tb00189.x

Raftery, A. E., Gneiting, T., Balabdaoui, F., & Polakowski, M. (2005). Using Bayesian Model Averaging to Calibrate Forecast Ensembles. Monthly Weather Review, 133(5), 1155–1174. https://doi.org/10.1175/MWR2906.1

e[j][l][m]  = x(j - l + m, j)
len(e[j][l]) = #(m s.t. n > j - l + m  >= 0 and -L <= m <= L)
             = #(m  s.t. n + l - j > m >= l - j and  -L <= m <= L)
             = #(m  s.t. n + l - j - 1 >= m >= l - j and  -L <= m <= L)

len(e[j][l]) = min(n + l - j - 1, L) - max(l - j, -L)

# proof that filling it up works
e[j][l+m][m] = x(j-l, j),
0 <= l <= j, l <= n

e[j][l'] = x(j- l' + m, j),
0 <= l' - m <= j
l' -j <= m <= l'

l <= n
l' -m <= n
l' -n <= m

-L <= m <= L

lower_bound = max(l' - j, l' - n, -L)
upper_bound = min(l', L)


Setup the dask distributed::

    dask-scheduler --scheduler-file /tmp/scheduler.json &
    export PYTHONPATH=/root/fcn-mip/workflows/scoring_tools:/root/fcn-mip:$PYTHONPATH
    dask-worker --nthreads 1 --nworkers 32 --scheduler-file  /tmp/scheduler.json
"""  # noqa
import torch
import torch.distributed
from collections import deque


async def yield_lagged_ensembles(
    *,
    observations,
    forecast,
    lags: int = 2,
    n: int = 10,
):
    """Yield centered lagged ensembles

    The forecast array has shape (len(observations), n)

    The ensemble consist of runs initialized with an offset of (-lags, ..., 0,
    ...lags). The ensemble size is therefore ``2*lags + =`` for points within
    the interior of the array.

    Supports running in parallel using the ``rank`` and ``world_size`` flags
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    # example one. no garbage collection
    nt = len(observations)
    assert n < nt

    # work trackers that will be used to determine when an ensemble is finished,
    # and ensure that all data is processed
    finished = set()
    ensemble = {}

    obs_buffer = deque([])

    for i in range(n + world_size):
        obs_buffer.append(await observations[i])

    n_iter = int(nt // world_size)
    assert nt % world_size == 0

    buffers = None

    for i0 in range(n_iter):

        for k in range(world_size):
            i = world_size * i0 + k
            if i + n + 1 < nt:
                obs_buffer.append(await observations[i + n + 1])

        i = world_size * i0 + rank
        nsteps = min(nt - world_size * i0 - 1, n)

        lead_time = -1
        async for y in forecast[i]:
            lead_time += 1
            j = i + lead_time
            if lead_time > nsteps:
                break

            if torch.distributed.is_initialized():
                buffers = [torch.empty_like(y) for _ in range(world_size)]
                # TODO only gather from needed ranks (i - m)
                torch.distributed.all_gather(buffers, y)
                if y.device != torch.device("cpu"):
                    cpu_buffers = [
                        torch.empty_like(b, device="cpu", pin_memory=True)
                        for b in buffers
                    ]
                    for cpu, gpu in zip(cpu_buffers, buffers):
                        cpu.copy_(gpu, non_blocking=True)
                else:
                    cpu_buffers = buffers
            else:
                cpu_buffers = [y]

            lead_time = j - i
            # need to loop over ranks to ensure that number of iterations
            # per rank is the same
            for r in range(world_size):
                for m in range(-lags, lags + 1):
                    ii = i0 * world_size + r
                    jj = ii + lead_time

                    if jj >= nt:
                        break

                    # Should this rank process the data or not?
                    i_owner = jj - lead_time - m
                    if i_owner % world_size != rank:
                        continue

                    k = (jj, lead_time + m)

                    store_me = cpu_buffers[r]
                    # ensemble[k][m]
                    ensemble.setdefault(k, {})[m] = store_me
                    # There are two options for this finishing criteria
                    # 1. if it work is not done in the next iteration, then we know
                    # we are done this would be implemented by
                    #
                    #       if not done(j, lead_time + m, i + 1):
                    #
                    # 2. if the ensemble has the expected number of members
                    # 2 seems easier to parallelize and less subject to the
                    # looping we take, so is what we do here:
                    expected = num(n=n, ell=lead_time + m, j=jj, L=lags)
                    if jj < nt and len(ensemble[k]) == expected:

                        # sanity check that a single ensemble is not
                        # processed multiple times
                        if k in finished:
                            assert False, k
                        finished.add(k)
                        # need to synchronize to ensure cpu buffers are filled
                        # before yielding the complete ensemble
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        yield k, ensemble.pop(k), await observations[jj]

        for _ in range(world_size):
            obs_buffer.popleft()

    assert not ensemble, len(ensemble)


def num(n, ell, j, L):
    a = max(ell - j, ell - n, -L)
    b = min(ell, L)
    return b - a + 1


def done(j, ell, i, lags, n):
    """Unused helper function wich says if lag ell and valid_time j are written
    to in a given iteration `i` of the loop in lagged_average_simple

    This is one way to implement the done criteria which is less easily
    parallelized. I am leaving it in the code for educational value only.
    """
    #
    a = j - i - lags <= ell <= j - i + lags
    b = n >= j - i >= 0
    return a & b
