# Contributing to `poli`

![Linting: black](https://img.shields.io/badge/Linting-black-black)
![Testing: pytest](https://img.shields.io/badge/Testing-pytest-blue)
![Testing: tox](https://img.shields.io/badge/Testing-tox-blue)
![Main branch: main](https://img.shields.io/badge/Pull_request_to-main-blue)

This note details how to contribute to `poli-baselines`.

## Forking and making pull requests

The main development branch is `main`. To contribute, we recommend creating a fork of this repository and making changes on your version. Once you are ready to contribute, we expect you to document, lint and test.

## Documentation standards

We follow [numpy's documentation standards](https://numpydoc.readthedocs.io/en/latest/format.html).

## Linting your changes

We expect you to lint the code you write or modify using `black`.

```bash
pip install black
black ./path/to/files
```

## Testing your changes for `dev``

Since we are testing multiple conda environments, we settled for using a combination of `tox` and `pytest`.

```bash
pip install tox

# To test both linting and logic (from the root of the project)
tox
```

## Create a pull request to main

Once all tests pass and you are ready to share your changes, create a pull request to the `main` branch.