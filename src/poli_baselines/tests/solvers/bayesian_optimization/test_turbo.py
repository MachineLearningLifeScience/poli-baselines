"""This module tests the TURBO solver."""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def test_turbo_runs():
    from poli import objective_factory
    from poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper import (
        TurboWrapper,
    )

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    black_box, x0 = problem.black_box, problem.x0
    x0 = np.concatenate([x0, np.random.rand(1, x0.shape[1])])
    y0 = black_box(x0)

    bounds = np.concatenate([-np.ones([x0.shape[1], 1]), np.ones([x0.shape[1], 1])], axis=-1)

    solver = TurboWrapper(black_box, x0, y0, bounds=bounds)

    solver.solve(max_iter=5)


if __name__ == "__main__":
    test_turbo_runs()
