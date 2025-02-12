.PHONY : docs doctests lint tests

all : lint tests doctests docs

docs :
	rm -rf docs/_build
	sphinx-build docs docs/_build

doctests :
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS README.md
	sphinx-build -b doctest docs docs/_build

tests :
	pytest -v --cov=optinspect --cov-report=term-missing --doctest-modules

lint:
	black --check .
	flake8 optinspect
	mypy --ignore-missing-imports optinspect
