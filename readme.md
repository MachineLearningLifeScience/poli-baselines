# `poli-baselines`, a library of optimizers of black-box functions

[![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)

> [!WARNING]  
> This package is a work in progress. Some solvers have not been properly tested. For an authoritative list of the stable solvers, check below or [our documentation](https://machinelearninglifescience.github.io/poli-docs/#black-box-optimization-algorithms).

`poli-baselines` is a collection of **black box optimization algorithms**, aimed mostly at optimizing discrete sequences. These optimization algorithms are meant to optimize objective functions defined using [`poli`](https://github.com/MachineLearningLifeScience/poli), a tool for instantiating complex, difficult-to-query functions.

If the dependencies get too specific, we provide replicable conda environments for each solver.

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

## Solvers available

Some solvers run on specific conda environments. We provide
the `environment.yml` files for each, so you can replicate
the environment in your computer.

These can be found [in the folder of each solver](src/poli_baselines/solvers).

| Name | Status | Reference |
|----------|----------|----------|
| **Random Mutations**   |  [![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)  | N/A   |
| **Random hill-climbing**   |  [![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)  | N/A   |
| **CMA-ES**   |  [![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)  | [pycma](https://github.com/CMA-ES/pycma)   |
| **(Fixed-length) Genetic Algorithm**   |  [![Test (conda, python 3.9)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing.yml)  | [pymoo's implementation](https://pymoo.org/algorithms/soo/ga.html)  |
| **Hvarfner's Vanilla BO** |  [![Ax (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml)  | [Hvarfner et al. 2024](https://arxiv.org/abs/2402.02229) |
| **Bounce** |  [![Bounce (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-bounce.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-bounce.yml)  | [Papenmeier et al. 2023](https://arxiv.org/abs/2307.00618) |
| **BAxUS** |  [![BAxUS (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-baxus.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-baxus.yml)  | [Papenmeier et al. 2022](https://arxiv.org/abs/2304.11468) |
| **Probabilistic Reparametrization** |  [![Prob. Rep. (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-pr.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-pr.yml)  | [Daulton et al. 2022](https://arxiv.org/abs/2210.10199) |
| **SAASBO** |  [![Ax (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml)  | [Eriksson and Jankowiak 2021](https://arxiv.org/abs/2103.00349) |
| **ALEBO** |  [![Ax (py3.10 in conda)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml/badge.svg)](https://github.com/MachineLearningLifeScience/poli-baselines/actions/workflows/python-tox-testing-ax.yml)  | [Lentham et al. 2020](https://proceedings.neurips.cc/paper/2020/file/10fb6cfa4c990d2bad5ddef4f70e8ba2-Paper.pdf) |

In the case of Probabilistic Reparametrization, we rely on a slightly modified version of the `run_one_replication.py` script from [the original repository](https://github.com/facebookresearch/bo_pr).

## Your first optimization using `poli-baselines`

As mentioned above, this library interoperates well with the discrete objective functions defined in [`poli`](https://github.com/MachineLearningLifeScience/poli). One such objective function is the ALOHA problem, in which we search the space of 5-letter sequences of the word "ALOHA". The following is a simple example of how one could use the `RandomMutation` solver inside `poli-baselines` to solve this problem:

```python
from poli.objective_repository import AlohaProblemFactory
from poli_baselines.solvers import RandomMutation

# Creating an instance of the problem
problem = AlohaProblemFactory().create()
f, x0 = problem.black_box, problem.x0
y0 = f(x0)

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

The `examples` folder includes the optimization of more complex objective functions such as `foldx` stability (if you have `foldx` installed in your computer), and the use of advanced black box optimizers like (Line) Bayesian Optimization.

## Want to contribute?

Take a look at our open issues, and check our [guide to contributing](https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/CONTRIBUTING.md).
