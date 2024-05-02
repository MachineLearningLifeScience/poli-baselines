"""This module tests the BAXUS solver."""

import pytest

pytestmark = pytest.mark.baxus

import warnings
import numpy as np


warnings.filterwarnings("ignore")


def test_baxus_instantiates():
    """Tests that BAXUS instantiates."""
    from poli import objective_factory

    pytest.importorskip("baxus")
    from poli_baselines.solvers.bayesian_optimization.baxus import BAxUS

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    black_box, x0 = problem.black_box, problem.x0
    y0 = black_box(x0)

    solver = BAxUS(black_box, x0, y0, lower_bound=-3.0, upper_bound=3.0, noise_std=0.0)

    assert solver is not None


def test_baxus_runs():
    from poli import objective_factory

    from poli_baselines.solvers.bayesian_optimization.baxus import BAxUS

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    black_box, x0 = problem.black_box, problem.x0
    y0 = black_box(x0)

    x0 = np.random.uniform(-1, 1, size=10).reshape(1, 10)
    y0 = black_box(x0)

    solver = BAxUS(black_box, x0, y0, lower_bound=-3.0, upper_bound=3.0, noise_std=0.0)

    solver.solve(max_iter=5)


if __name__ == "__main__":
    test_baxus_runs()
