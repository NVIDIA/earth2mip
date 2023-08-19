# %%
from earth2mip.networks.graphcast.inference import Graphcast
from earth2mip.networks.graphcast.channels import pack, unpack
import xarray
from graphcast import data_utils
import dataclasses

# %%

root = "/lustre/fsw/sw_earth2_ml/graphcast/"
model = Graphcast(root)

# %%
# load dataset
dataset_filename = os.path.join(
    root, "dataset", "source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

# get eval data
eval_steps = 12
(
    eval_inputs,
    eval_targets,
    eval_forcings,
) = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(model.task_config),
)


predictions = model(eval_inputs, eval_targets, eval_forcings)
print(predictions)

# %%
# simple demo of how to do some time steps
import jax
from earth2mip.networks.graphcast.rollout import _get_next_inputs
rng = jax.random.PRNGKey(0)


x = eval_inputs

for i in range(10):

    times = eval_forcings.time[0:1]

    y = eval_targets.isel(time=[i]).assign_coords(time=times)
    f = eval_forcings.isel(time=[i]).assign_coords(time=times)


    next_inputs = model.run_forward_jitted(
        rng=rng,
        inputs=x,
        targets_template=y, 
        forcings=f
    )

    next_inputs = next_inputs.merge(f)

    x = _get_next_inputs(x, next_inputs)


# %% integrate with 
# TODO use pack/unpack here
