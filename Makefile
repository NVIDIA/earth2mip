# stages called by modulus ci library
# setup-ci black interrogate lint license install pytest coverage doctest
# https://gitlab-master.nvidia.com/modulus/modulus-launch/-/blob/main/Makefile
# ALL THESE TARGETS NEED to be here for blossom-ci

install:
	apt-get install -y libeccodes-dev
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install .[pangu,graphcast]

setup-ci:
	pip install .[dev]
	pre-commit install

format:
	pre-commit run black -a

lint:
	echo "TODO: add interrogate"
	pre-commit run check-added-large-files -a
	pre-commit run ruff -a

license:
	python test/_license/header_check.py

doctest:
	echo "TODO"
	true

pytest:
	coverage run -m pytest test/

coverage:
	coverage combine
	coverage report

report:
	coverage xml
	echo $(COV_ARGS)
	curl -Os https://uploader.codecov.io/latest/linux/codecov
	chmod +x codecov
	./codecov -v -f e2mip.coverage.xml $(COV_ARGS)

docs:
	$(MAKE) -C docs html
	open docs/_build/html/index.html
.PHONY: docs
