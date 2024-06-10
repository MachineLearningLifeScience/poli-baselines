"""This module tests the TURBO solver."""

import warnings
import numpy as np

from poli_baselines.solvers.bayesian_optimization.amortized.amortized_bo_wrapper import AmortizedBOWrapper

warnings.filterwarnings("ignore")


def test_amortized_bo_runs():
    from poli import objective_factory

    problem = objective_factory.create(
        name="aloha", observer_name=None
    )
    black_box, x0 = problem.black_box, problem.x0
    y0 = black_box(x0)

    solver = AmortizedBOWrapper(black_box, x0, y0)

    solver.solve(max_iter=5)


if __name__ == "__main__":
    test_amortized_bo_runs()
