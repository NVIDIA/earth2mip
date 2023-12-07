Extensive verifications of graphcast. since these are slow, I did not want to
include them in the tests suite.

This directory includes a reference implementation of graphcast that generates
a time series.

Run `reference_inference.py` to generate a file `predictions.nc`

Running `multi_step.py` verifies that the earth2mip implementation produces a
close enough time series to `predictions.nc`.

Note that graphcast is not deterministic from run to run, since jax uses
non-determinstic gpu operations. Especially given the use of fp16, this can lead
to some noise-like changes in the predictions from run to run. Therefore the
checks in `multi_step.py` have some tolerances.

You can use tensorboard to look at some more diagnostics of `multi_step.py`

	tensorboard --logdir runs

