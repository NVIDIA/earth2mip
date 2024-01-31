
# Run a 1 year simulation with several models

```
# to avoid OOM. jax/xla is greedy by default
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
python3 1_year_run.py
```

Outputs are stored in `year_runs/`
