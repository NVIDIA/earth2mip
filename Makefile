# stages called by modulus ci library
# setup-ci black interrogate lint license install pytest coverage doctest
# https://gitlab-master.nvidia.com/modulus/modulus-launch/-/blob/main/Makefile
# ALL THESE TARGETS NEED to be here for blossom-ci

.PHONY: install
install:
	apt-get install -y libeccodes-dev
	pip install --upgrade pip
	pip install .[pangu,graphcast]
	pip install -r requirements.txt

.PHONY: setup-ci
setup-ci:
	pip install .[dev]
	pre-commit install

.PHONY: format
format:
	pre-commit run black -a

.PHONY: lint
lint:
	echo "TODO: add interrogate"
	pre-commit run check-added-large-files -a
	pre-commit run ruff -a
	pre-commit run mypy -a

.PHONY: license
license:
	python test/_license/header_check.py

.PHONY: doctest
doctest:
	echo "TODO"
	true

.PHONY: pytest_parallel
pytest_parallel:
	torchrun -r 0:3,1:0,2:3 --nproc_per_node 3 -m  pytest test/lagged_ensembles/test_lagged_averaged_forecast.py

.PHONY: pytest
pytest:
	coverage run -m pytest test/

.PHONY: coverage
coverage:
	coverage combine
	coverage report

.PHONY: report
report:
	coverage xml
	curl -Os https://uploader.codecov.io/latest/linux/codecov
	chmod +x codecov
	./codecov -v -f e2mip.coverage.xml $(COV_ARGS)

.PHONY: docs
docs:
	pip install .[docs]
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

.PHONY: docs-full
docs-full:
	pip install .[docs]
	$(MAKE) -C docs clean
	rm -rf examples/outputs
	PLOT_GALLERY=True RUN_STALE_EXAMPLES=True $(MAKE) -C docs html
