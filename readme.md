# `poli-baselines`, a library of optimizers of black-box functions

[![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)

> [!WARNING]  
> This package is a work in progress.

`poli-baselines` is a collection of **black box optimization algorithms**, aimed mostly at optimizing discrete sequences. These optimization algorithms are meant to optimize objective functions defined using [`poli`](https://github.com/MachineLearningLifeScience/poli), a tool for instantiating complex, difficult-to-query functions.

## Installation

Create a fresh conda environment by running

```bash
conda create -n poli-baselines python=3.9
conda activate poli-baselines
```

After which, you can download this repository and install it locally

```bash
git clone git@github.com:MachineLearningLifeScience/poli-baselines.git
cd ./poli-baselines
```

Now install the requirements, as well as the library:

```bash
pip install -r requirements.txt
pip install -e .
```

After this, you could test you installation by running (inside your `poli-baselines` environment):

```bash
python -c "import poli_baselines ; print('Everything went well!')"
```

## Your first optimization using `poli-baselines`

As mentioned above, this library interoperates well with the discrete objective functions defined in [`poli`](https://github.com/MachineLearningLifeScience/poli). One such objective function is the ALOHA problem, in which we search the space of 5-letter sequences of the word "ALOHA". The following is a simple example of how one could use the `RandomMutation` solver inside `poli-baselines` to solve this problem:

```python
from poli import objective_factory
from poli_baselines.solvers import RandomMutation

# Creating an instance of the problem
problem_info, f, x0, y0, run_info = objective_factory.create(name="aloha")

# Creating an instance of the solver
solver = RandomMutation(
    black_box=f,
    x0=x0,
    y0=y0,
)

# Running the optimization for 1000 steps,
# breaking if we find a performance above 5.0.
solver.solve(max_iter=1000, break_at_performance=5.0)

# Checking if we got the solution we were waiting for
print(solver.get_best_solution())  # Should be [["A", "L", "O", "H", "A"]]
```

## More examples

The `examples` folder includes the optimization of more complex objective functions such as `foldx` stability (if you have `foldx` installed in your computer), and the use of advanced black box optimizers like (Line) Bayesian Optimization, or NSGA-2.

## Want to contribute?

Take a look at our open issues, and check our [guide to contributing](https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/CONTRIBUTING.md).
