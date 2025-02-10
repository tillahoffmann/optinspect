.PHONY : docs doctests lint tests

all : lint tests doctests docs

docs :
	rm -rf docs/_build
	sphinx-build docs docs/_build

doctests :
	sphinx-build -b doctest docs docs/_build

tests :
	pytest -v --cov=optinspect --cov-report=term-missing

lint:
	black --check .
