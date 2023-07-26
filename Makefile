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
	echo "TODO"
	true

lint:
	pre-commit run check-added-large-files -a
	pre-commit run flake8 -a

license:
	echo "TODO"
	true

doctest:
	echo "TODO"
	true

pytest:
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest 

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --fail-under=20 && \
		coverage html
