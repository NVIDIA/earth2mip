<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0a0] - 2024-xx-xx

### Added

- Local Xarray data source
- Diagnostic precipitation model example
- `yield_lagged_ensembles` has max_lags, min_lags options. These allow for
  non-centered lagged windows.

### Changed

- Refactored initial conditions / data source API
- Updated GFS data source to pull from AWS
- Updated IFS data source to use ECMWF's open data package
- Change DLWP inferencer class to now work on 6 hr time-steps
- Moving perturbation methods from inference ensemble into submodule

### Deprecated

### Removed

- `earth2mip.xarray`

### Fixed

- Fixed Graphcast implementation
- Lagged ensembles now can run in parallel on arbitrary ranks
- Fixed default cache location of file system
- Fixed fs.glob search for H5 files
- Corrected DLWP intial condition fetch with history

### Security

### Dependencies

## [0.1.0] - 2023-11-17

### Added

- Initial public release of Earth-2 MIP.
