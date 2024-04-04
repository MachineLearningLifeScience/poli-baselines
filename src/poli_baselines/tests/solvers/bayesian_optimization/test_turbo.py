"""This module tests the BAXUS solver."""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def test_turbo_runs():
    from poli import objective_factory
    from poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper import TurboWrapper

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    black_box, x0 = problem.black_box, problem.x0
    y0 = black_box(x0)

    x0 = np.random.uniform(-1, 1, size=10).reshape(1, 10)
    y0 = black_box(x0)

    solver = TurboWrapper(black_box, x0, y0)

    solver.solve(max_iter=5)


if __name__ == "__main__":
    test_turbo_runs()
