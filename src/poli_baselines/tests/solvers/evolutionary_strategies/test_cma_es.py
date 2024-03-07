"""This module implements tests for the CMA-ES solver."""

import numpy as np


def test_running_cma_es():
    from poli_baselines.solvers import CMA_ES
    from poli.objective_repository import ToyContinuousBlackBox

    n_dimensions = 3
    population_size = 10

    f = ToyContinuousBlackBox(
        function_name="ackley_function_01",
        n_dimensions=n_dimensions,
    )
    x0 = np.random.normal(size=n_dimensions * population_size).reshape(
        population_size, -1
    )
    y0 = f(x0)

    initial_mean = np.random.normal(size=n_dimensions)
    solver = CMA_ES(
        black_box=f,
        x0=x0,
        y0=y0,
        initial_mean=initial_mean,
        initial_sigma=1.0,
        population_size=population_size,
    )

    solver.solve(max_iter=50, verbose=True)


def test_running_cma_es_docs():
    from poli_baselines.solvers import CMA_ES
    from poli.objective_repository import ToyContinuousBlackBox

    n_dimensions = 3
    population_size = 10

    f = ToyContinuousBlackBox(
        function_name="ackley_function_01",
        n_dimensions=n_dimensions,
    )

    x0 = np.random.normal(size=(population_size, n_dimensions))
    y0 = f(x0)

    initial_mean = np.random.normal(size=n_dimensions)
    solver = CMA_ES(
        black_box=f,
        x0=x0,
        y0=y0,
        initial_mean=initial_mean,
        initial_sigma=1.0,
        population_size=population_size,
    )

    solver.solve(max_iter=50)


if __name__ == "__main__":
    test_running_cma_es()
