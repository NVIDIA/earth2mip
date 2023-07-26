# stages called by modulus ci library
# setup-ci black interrogate lint license install pytest coverage doctest
# https://gitlab-master.nvidia.com/modulus/modulus-launch/-/blob/main/Makefile
# ALL THESE TARGETS NEED to be here for blossom-ci

install:
	pip install --upgrade pip && \
		pip install -e .

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	true

lint:
	pre-commit run -a

license:
	true

doctest:
	true

pytest:
	coverage run -m pytest

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --fail-under=20 && \
		coverage html
