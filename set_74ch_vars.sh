export MASTER_ADDR=$(hostname)
#export MODEL_REGISTRY=/pscratch/sd/a/amahesh/earth2mip_klugestats_registry/
export MODEL_REGISTRY=/pscratch/sd/a/amahesh/earth2mip_prod_registry/
export MASTER_PORT=29500
export ERA5_HDF5=/pscratch/sd/p/pharring/74var-6hourly/staging/
export HEAT_INDEX_LOOKUP_TABLE=/pscratch/sd/a/amahesh/hens/prod_heat_index_lookup.zarr
export DETERMINISTIC_RMSE=/pscratch/sd/a/amahesh/hens/optimal_perturbation_targets/means/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc
