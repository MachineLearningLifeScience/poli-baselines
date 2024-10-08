# Contributing to `poli`

![Formatting: black](https://img.shields.io/badge/Formatting-black-black)
![Formatting: isort](https://img.shields.io/badge/Formatting-isort-black)
![Linting: ruff](https://img.shields.io/badge/Linting-ruff-black)
![Testing: pytest](https://img.shields.io/badge/Testing-pytest-blue)
![Testing: tox](https://img.shields.io/badge/Testing-tox-blue)
![Main branch: main](https://img.shields.io/badge/Pull_request_to-main-blue)

This note details how to contribute to `poli-baselines`.

## Setting up your dev environment

We recommend creating a fresh environment (with Python 3.10 for most solvers), installing the `requirements-dev.txt`, and the pre-commit hooks

```bash
pip install requirements-dev.txt
pre-commit install
```

## Forking and making pull requests

The main development branch is `main`. To contribute, we recommend creating a fork of this repository and making changes on your version. Once you are ready to contribute, we expect you to document, lint and test.

## Documentation standards

We follow [numpy's documentation standards](https://numpydoc.readthedocs.io/en/latest/format.html).

## Linting your changes

Each commit will lint and format your changes using `ruff`, `black` and `isort`.

## Testing your changes

Since we are multiple environments, we settled for using a combination of `tox` and `pytest`. We encourage you to add tests for your solver, and create a new testing environment inside `tox.ini`.

```bash
# To test both linting and logic (from the root of the project)
tox
```

If you want to test a specific environment, you can pass it with the `-e` flag. For example

```bash
tox -e lint
```

## Create a pull request to main

Once all tests pass and you are ready to share your changes, create a pull request to the `main` branch.