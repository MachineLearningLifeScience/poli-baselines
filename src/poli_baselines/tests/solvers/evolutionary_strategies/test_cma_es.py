"""This module implements tests for the CMA-ES solver."""

import numpy as np


def test_running_cma_es():
    from poli_baselines.solvers import CMA_ES
    from poli import objective_factory

    _, f, _, _, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=3,
    )
    x0 = np.random.normal(size=3).reshape(1, -1)
    y0 = f(x0)

    solver = CMA_ES(
        black_box=f,
        x0=x0,
        y0=y0,
        initial_sigma=1.0,
        population_size=10,
    )

    solver.solve(max_iter=2)
