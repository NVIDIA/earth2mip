<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release latest (released Aug MM, YYYY)

### Added

- `earth2mip.initial_conditions.cds.CDSDataSource`
- `earth2mip.initial_conditions.cds.get`

### Changed

- `earth2mip.time_loop.TimeLoop.grid` and
`earth2mip.initial_conditions.base.DataSource.grid` are now
`earth2mip.grid.LatLonGrid` objects. Allows more flexible usage.
- `earth2mip.schema.ChannelSet` removed and
`earth2mip.schema.Model.in_channel_names` added. **Some older model packages
will be broken and can be fixed by adding .in_channel_names and
.out_channel_names attributes to any metadata.json files**.
- The `ERA5_HDF_73` and `TIME_MEAN_73` configurations have been removed. Used
`ERA5_HDF` and `TIME_MEAN` instead to control the data used by the "era5" or
"hrmip" initial condition sources.
- Add `--model-metadata` flag to `earth2mip.inference_medium_range` and
  `earth2mip.inference_ensemble`.
- Add `metadata` argument to `earth2mip.networks.get_model`.
- Save year in name of the folder of ensemble predictions
- Sort the ensemble files before opening them with `glob` in `earth2mip/score_ensemble_outputs.py`
- Add deepmind graphcast

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

- Added eccodes, ecmwflibs, h5netcdf
