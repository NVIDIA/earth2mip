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

- Add `--model-metadata` flag to `earth2mip.inference_medium_range` and
  `earth2mip.inference_ensemble`.
- Add `metadata` argument to `earth2mip.networks.get_model`.
- Sort the ensemble files before opening them with `glob` in `earth2mip/score_ensemble_outputs.py`

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

- Added eccodes
